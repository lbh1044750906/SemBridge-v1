# BridgeMatching_train ---> Train the matching prediction model
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from typing import List, Dict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import re
import sys
from tqdm import tqdm
import random
import os
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd

# ======================
# Configuration parameters
# ======================
MODEL_NAME = "./Deepseek/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
BRIDGES_FILE = "../bridge_items.json"
TRAIN_FILE = "train_bridge_data.json"
OUTPUT_BASE_DIR = "./deepseek14_bridge_tarin"  # Base path; each fold will create a fold_X subdirectory.
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
SEED = 42
N_SPLITS = 4  # 4-fold cross-validation

# Set the random seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ======================
# Load bridging item
# ======================
def load_bridges(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        bridges = json.load(f)
    lines = [f"- {b['ID']}: {b['SD']}" for b in bridges]
    return "\n".join(lines)

BRIDGE_LIST_TEXT = load_bridges(BRIDGES_FILE)


# ======================
# Prompt Template
# ======================
# English
PROMPT_TEMPLATE = """[System Prompt]
Please determine which bridge items are relevant to the following event description, and for each relevant bridge item, assess whether the behavior complies with its semantic requirements.

[Input]
List of Bridge Items: {bridge_list_text}
Event Description: {behavior}

[Output Format]
Please strictly output the evaluation result in the following format:
[{{"bridge_id": "B-01", "SJ": 1}}, {{"bridge_id": "B-02", "SJ": 0}}]

[Constraints & Rules]
1. The output must be a JSON list. Each element is a dictionary containing exactly two required keys: "bridge_id" (the bridge item ID, e.g., "B-01", which must exactly match an ID in the bridge item list) and "SJ"; this value must be an integer and cannot be omitted).
2. Include only bridge items that are relevant to the behavior. If no bridge items are relevant, output [].
3. Assign SJ = 1 if the behavior fully satisfies the semantic description of the bridge item; assign SJ = 0 if it does not.
4. Output only the JSON list with no additional text, commentary, or explanation.

[Examples]
Event Description: A certain application utilises users' browsing history for personalised advertising without their explicit consent.
Output Result: [{{"bridge_id": "B-01", "SJ": 0}}]

Output Result: """

# Chinese
PROMPT_TEMPLATE = """请根据以下行为描述，判断其与哪些桥接项相关，并对每个相关的桥接项判断是否符合其语义描述要求。

桥接项列表：
{bridge_list_text}

行为描述：
{behavior}

请按以下格式严格输出判断结果：
[{{"bridge_id": "B-01", "SJ": 1}}, {{"bridge_id": "B-02", "SJ": 0}}]

示例-1：
行为描述：某公司未经用户同意，将用户的浏览记录用于个性化广告推送。

示例-1判断结果：
[{{"bridge_id": "B-01", "SJ": 0}}]

说明：
1. 必须输出 JSON 列表，每个列表元素是包含 两个必填键的字典：
   - "bridge_id"：桥接项 ID（如 "B-01"，需与桥接项列表中的 ID 完全一致）
   - "SJ"：合规性判断结果（1=符合，0=不符合，必须为整数，不可省略）
2. 只输出行为相关的桥接项，若无相关的桥接项则输出"[]"
3. SJ=1 表示行为符合该桥接项的描述，SJ=0 表示不符合
4. 输出为 JSON 列表格式，不要额外解释

判断结果："""


# ======================
# Customised Dataset
# ======================
class BridgeMatchDataset(Dataset):
    def __init__(self, data: List[str], tokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        print(f"\n📂 Building {len(data)} samples...")
        for line_idx, item in enumerate(data):
            behavior = item.get("behavior", "").strip()
            if not behavior:
                continue

            labels = item.get("compliance_labels", [])
            if not isinstance(labels, list):
                labels = []

            prompt = PROMPT_TEMPLATE.format(
                bridge_list_text=BRIDGE_LIST_TEXT,
                behavior=behavior
            )

            label_entries = []
            for label in labels:
                if isinstance(label, dict) and "bridge_id" in label and "SJ" in label:
                    label_entries.append({"bridge_id": label["bridge_id"], "SJ": label["SJ"]})

            target = json.dumps(label_entries, ensure_ascii=False)
            full_text = f"{prompt}\n{target}".strip()

            if len(full_text) == 0:
                continue

            self.examples.append(full_text)
            if line_idx < 3:
                print(f"✅ Example: {behavior[:30]}...")

        print(f"✅ Successfully constructed {len(self.examples)} valid samples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        full_text = self.examples[idx]
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None
        )
        input_ids = inputs["input_ids"]

        try:
            sep_token = "Output Result："
            full_str = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            sep_pos = full_str.find(sep_token)
            if sep_pos != -1:
                sep_pos += len(sep_token)
                sep_pos_tokens = len(self.tokenizer.encode(full_str[:sep_pos], add_special_tokens=False))
            else:
                sep_pos_tokens = 0
        except:
            sep_pos_tokens = 0

        labels = [-100] * sep_pos_tokens + input_ids[sep_pos_tokens:]
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }


# ======================
# Data collator
# ======================
def collate_fn(batch, tokenizer):
    batch = [b for b in batch if isinstance(b, dict) and "input_ids" in b]
    if not batch:
        return {}
    return tokenizer.pad(batch, padding=True, return_tensors="pt")


# ======================
# JSON Security Analysis
# ======================
def safe_parse_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r'\[\s*\{.*\}\s*\]', s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return []


# ======================
# Establish a mapping from bridge_id to the index
# ======================
def build_id_to_index(bridge_list_text: str) -> Dict[str, int]:
    lines = bridge_list_text.strip().split('\n')
    id_to_idx = {}
    for i, line in enumerate(lines):
        if ':' in line:
            bid_part = line.split(':')[0].strip()
            bid = bid_part.replace('- ', '')
            id_to_idx[bid] = i
    return id_to_idx


# ======================
# Load raw data
# ======================
def load_data(file_path: str):
    data = []
    print(f"📄 Loading data file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                behavior = item.get("behavior", "").strip()
                labels = item.get("compliance_labels", [])
                if not behavior:
                    print(f"⚠️ Line {line_idx}: Missing behaviour, skipped")
                    continue
                if not isinstance(labels, list):
                    labels = []
                data.append({"behavior": behavior, "compliance_labels": labels})
            except Exception as e:
                print(f"⚠️ Parsing failed on line {line_idx}: {e}")
    print(f"✅ A total of {len(data)} data entries have been loaded.")
    return data


# ======================
# Evaluation function (for a single fold)
# ======================
def evaluate_fold(model_path: str, val_data: List[Dict], id_to_idx: Dict[str, int]) -> Dict[str, float]:
    print(f"🔍 Evaluation model: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Tokenizer failed to load: {e}")
        return {}

    try:
        config = PeftConfig.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=False, 
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return {}

    num_classes = len(id_to_idx)
    print(f"✅ A total of {num_classes} bridge item categories")

    # Statistical container
    bridge_stats = defaultdict(lambda: {"tp_rel": 0, "fp_rel": 0, "fn_rel": 0, "correct_sj": 0, "total_sj": 0})
    sample_exact_match = 0

    for item in tqdm(val_data, desc="Evaluating"):
        behavior = item["behavior"]
        true_labels = {lbl["bridge_id"]: lbl["SJ"] for lbl in item["compliance_labels"] if isinstance(lbl, dict)}
        true_bridge_ids = set(true_labels.keys())

        prompt = PROMPT_TEMPLATE.format(
            bridge_list_text=BRIDGE_LIST_TEXT,
            behavior=behavior
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False 
            )

        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_list = safe_parse_json(response)
        # pred_labels = {item["bridge_id"]: item["SJ"] for item in pred_list if isinstance(item, dict) and "bridge_id" in item}
        pred_labels = {}
        for item in pred_list:
            if isinstance(item, dict):
                bid = item.get("bridge_id")
                sj_val = item.get("SJ")
                if bid is not None and sj_val is not None:
                    # Ensure that SJ is an integer of either 0 or 1.
                    if isinstance(sj_val, (int, float)) and sj_val in (0, 1):
                        pred_labels[bid] = int(sj_val)
                    else:
                        print(f"⚠️ Invalid SJ value: {sj_val}, bridge_id={bid}")
                else:
                    print(f"⚠️ Missing bridge_id or SJ: {item}")
            else:
                print(f"⚠️ Non-dictionary entries are ignored.: {item}")

        # Statistics each bridge item
        all_bridges = set(true_bridge_ids) | set(pred_labels.keys())
        for bid in all_bridges:
            is_true_rel = 1 if bid in true_bridge_ids else 0
            is_pred_rel = 1 if bid in pred_labels else 0

            if is_true_rel and is_pred_rel:
                bridge_stats[bid]["tp_rel"] += 1
                if pred_labels.get(bid) == true_labels.get(bid):
                    bridge_stats[bid]["correct_sj"] += 1
                bridge_stats[bid]["total_sj"] += 1
            elif not is_true_rel and is_pred_rel:
                bridge_stats[bid]["fp_rel"] += 1
            elif is_true_rel and not is_pred_rel:
                bridge_stats[bid]["fn_rel"] += 1

        # Sample-level exact match
        pred_set = {(k, v) for k, v in pred_labels.items()}
        true_set = {(k, v) for k, v in true_labels.items()}
        if pred_set == true_set:
            sample_exact_match += 1

    # micro & macro
    overall_tp_rel = sum(s["tp_rel"] for s in bridge_stats.values())
    overall_fp_rel = sum(s["fp_rel"] for s in bridge_stats.values())
    overall_fn_rel = sum(s["fn_rel"] for s in bridge_stats.values())
    overall_correct_sj = sum(s["correct_sj"] for s in bridge_stats.values())
    overall_total_sj = sum(s["total_sj"] for s in bridge_stats.values())

    micro_recall = overall_tp_rel / (overall_tp_rel + overall_fn_rel) if (overall_tp_rel + overall_fn_rel) > 0 else 0
    micro_precision = overall_tp_rel / (overall_tp_rel + overall_fp_rel) if (overall_tp_rel + overall_fp_rel) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    micro_sj_acc = overall_correct_sj / overall_total_sj if overall_total_sj > 0 else 0

    per_bridge_recalls = [s["tp_rel"] / (s["tp_rel"] + s["fn_rel"]) if (s["tp_rel"] + s["fn_rel"]) > 0 else 0 for s in bridge_stats.values()]
    per_bridge_precisions = [s["tp_rel"] / (s["tp_rel"] + s["fp_rel"]) if (s["tp_rel"] + s["fp_rel"]) > 0 else 0 for s in bridge_stats.values()]
    per_bridge_f1s = [2*p*r/(p+r) if (p+r)>0 else 0 for p,r in zip(per_bridge_precisions, per_bridge_recalls)]
    per_bridge_sj_accs = [s["correct_sj"] / s["total_sj"] if s["total_sj"] > 0 else 0 for s in bridge_stats.values()]

    macro_recall = np.mean(per_bridge_recalls) if per_bridge_recalls else 0
    macro_precision = np.mean(per_bridge_precisions) if per_bridge_precisions else 0
    macro_f1 = np.mean(per_bridge_f1s) if per_bridge_f1s else 0
    macro_sj_acc = np.mean(per_bridge_sj_accs) if per_bridge_sj_accs else 0

    sample_exact_match_rate = sample_exact_match / len(val_data)

    return {
        'micro_recall': micro_recall,
        'micro_precision': micro_precision,
        'micro_f1': micro_f1,
        'micro_sj_acc': micro_sj_acc,
        'macro_recall': macro_recall,
        'macro_precision': macro_precision,
        'macro_f1': macro_f1,
        'macro_sj_acc': macro_sj_acc,
        'sample_exact_match': sample_exact_match_rate
    }


# ======================
# Cross-Validation Main Function
# ======================
def train_kfold():
    print(f"🔁 Commencing {N_SPLITS} fold cross-validation...")

    # Loading data
    all_data = load_data(TRAIN_FILE)
    if len(all_data) == 0:
        raise ValueError("❌ The loaded data is empty. Please check the data file path and format.")

    # Construct the bridge_id mapping
    id_to_idx = build_id_to_index(BRIDGE_LIST_TEXT)
    all_bridge_ids = sorted(id_to_idx.keys())
    num_classes = len(all_bridge_ids)
    print(f"✅ A total of {num_classes} bridging items: {all_bridge_ids}")

    # Constructing a multi-label matrix y
    X = all_data
    y = []
    for item in all_data:
        vec = [0] * num_classes
        for lbl in item.get("compliance_labels", []):
            if isinstance(lbl, dict) and "bridge_id" in lbl:
                bid = lbl["bridge_id"]
                if bid in id_to_idx:
                    vec[id_to_idx[bid]] = 1
        y.append(vec)
    y = np.array(y, dtype=int)

    # Initialising result storage
    fold_results = []

    # Stratified K-Fold on multi-label using iterative split
    for fold in range(N_SPLITS):
        print(f"\n" + "="*60)
        print(f"📌 Start training the {fold+1}/{N_SPLITS} fold")
        print("="*60)

        OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"fold_{fold}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Manually implementing K-fold cross-validation: partitioning using a fixed random seed
        indices = np.arange(len(X))
        np.random.RandomState(SEED + fold).shuffle(indices)

        # Split Index
        n_val = len(X) // N_SPLITS
        val_start = fold * n_val
        val_end = (fold + 1) * n_val if fold < N_SPLITS - 1 else len(X)
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        # Retrieve a subset of data
        train_data = [X[i] for i in train_indices]
        val_data = [X[i] for i in val_indices]

        print(f"✅ Fold {fold+1}: {len(train_data)} records in the training set, {len(val_data)} records in the validation set")

        # Distribute each bridge_id across the training and validation sets
        def count_bridge_distribution(data, id_to_idx):
            counts = defaultdict(int)
            for item in data:
                for lbl in item.get("compliance_labels", []):
                    if isinstance(lbl, dict) and "bridge_id" in lbl:
                        bid = lbl["bridge_id"]
                        if bid in id_to_idx:
                            counts[bid] += 1
            return dict(counts)
    
        train_counts = count_bridge_distribution(train_data, id_to_idx)
        val_counts = count_bridge_distribution(val_data, id_to_idx)
    
        print("\n📊 Distribution of bridging items across the training and validation sets：")
        print(f"{'Bridging item':<8} {'Training set':<8} {'Validation set':<8} {'Total':<8}")
        print("-" * 40)
        total_all = 0
        for bid in all_bridge_ids:
            if bid not in id_to_idx:
                continue
            train_cnt = train_counts.get(bid, 0)
            val_cnt = val_counts.get(bid, 0)
            total = train_cnt + val_cnt
            total_all += total
            print(f"{bid:<8} {train_cnt:<8} {val_cnt:<8} {total:<8}")
        print("-" * 40)
        print(f"{'Total':<8} {sum(train_counts.values()):<8} {sum(val_counts.values()):<8} {total_all:<8}")

        # Retain data for evaluation
        with open(os.path.join(OUTPUT_DIR, "train_data.json"), 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(os.path.join(OUTPUT_DIR, "val_data.json"), 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Load the tokeniser and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=False,
            # offload_folder=None,
            # offload_state_dict=False,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Dataset
        train_dataset = BridgeMatchDataset(train_data, tokenizer)
        val_dataset = BridgeMatchDataset(val_data, tokenizer)

        # Training args
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=False,
            bf16=True,
            optim="adamw_torch",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=lambda x: collate_fn(x, tokenizer),
        )

        # Training
        print(f"🚀 Start training from the {fold+1} fold onwards...")
        trainer.train()

        # Save
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Evaluation
        results = evaluate_fold(OUTPUT_DIR, val_data, id_to_idx)
        fold_results.append(results)

        print(f"✅ Fold {fold+1} completed, result:")
        for k, v in results.items():
            print(f"   {k}: {v:.4f}")

    # ======================
    # Summarise all fold results
    # ======================
    print("\n" + "="*70)
    print("📊 Summary of Final Cross-Validation Results")
    print("="*70)

    df = pd.DataFrame(fold_results)
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:<20} : {mean_val:.4f} ± {std_val:.4f}")

    print("\n📋 Detailed results for each fold:")
    print(df.round(4))

    # Save results to file
    result_file = os.path.join(OUTPUT_BASE_DIR, "cv_results.csv")
    df.to_csv(result_file, index=False, float_format='%.4f')
    print(f"\n✅ All results have been saved to: {result_file}")


# ======================
# Main function
# ======================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        print("❌ Cross-validation mode does not support standalone evaluation. Please run the full training process.")
    else:
        train_kfold()