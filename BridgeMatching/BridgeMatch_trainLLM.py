# train_model.py
# 个人信息保护合规审计：基于语义桥接的模型训练与评估
# 使用 5 折交叉验证

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
# 配置参数
# ======================
MODEL_NAME = "./Deepseek/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
BRIDGES_FILE = "../bridge_items.json"
TRAIN_FILE = "train_bridge_data.json"
OUTPUT_BASE_DIR = "./deepseek14_bridge_structured_cv"  # 基础路径，每折会创建 fold_X 子目录
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
SEED = 42
N_SPLITS = 4  # 4折交叉验证

# 设置随机种子
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
# 加载桥接项
# ======================
def load_bridges(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        bridges = json.load(f)
    lines = [f"- {b['ID']}: {b['SD']}" for b in bridges]
    return "\n".join(lines)

BRIDGE_LIST_TEXT = load_bridges(BRIDGES_FILE)


# ======================
# 提示模板
# ======================
# PROMPT_TEMPLATE = """请根据以下行为描述，判断其与哪些桥接项相关，并对每个相关的桥接项判断是否符合其语义描述要求。

# 桥接项列表：
# {bridge_list_text}

# 行为描述：
# {behavior}

# 请按以下格式严格输出判断结果：
# [{{"bridge_id": "B-01", "SJ": 1}}, {{"bridge_id": "B-02", "SJ": 0}}]

# 说明：
# 1. 必须输出 JSON 列表，每个列表元素是包含 两个必填键的字典：
#    - "bridge_id"：桥接项 ID（如 "B-01"，需与桥接项列表中的 ID 完全一致）
#    - "SJ"：合规性判断结果（1=符合，0=不符合，必须为整数，不可省略）
# 2. 只输出行为相关的桥接项，若无相关的桥接项则输出"[]"
# 3. SJ=1 表示行为符合该桥接项的描述，SJ=0 表示不符合
# 4. 输出为 JSON 列表格式，不要额外解释

# # 判断结果："""

PROMPT_TEMPLATE = """请根据以下行为描述，判断其与哪些桥接项相关，并对每个相关的桥接项判断是否符合其语义描述要求。

# 桥接项列表：
{bridge_list_text}

# 行为描述：
{behavior}

# 请按以下格式严格输出判断结果：
[{{"bridge_id": "B-01", "SJ": 1}}, {{"bridge_id": "B-02", "SJ": 0}}]

# 示例-1：
行为描述：某公司未经用户同意，将用户的浏览记录用于个性化广告推送。

# 示例-1判断结果：
[{{"bridge_id": "B-01", "SJ": 0}}]

# 说明：
1. 必须输出 JSON 列表，每个列表元素是包含 两个必填键的字典：
   - "bridge_id"：桥接项 ID（如 "B-01"，需与桥接项列表中的 ID 完全一致）
   - "SJ"：合规性判断结果（1=符合，0=不符合，必须为整数，不可省略）
2. 只输出行为相关的桥接项，若无相关的桥接项则输出"[]"
3. SJ=1 表示行为符合该桥接项的描述，SJ=0 表示不符合
4. 输出为 JSON 列表格式，不要额外解释

# 判断结果："""

# PROMPT_TEMPLATE = """请根据以下行为描述，严格按照两步流程进行判断，最终输出标准 JSON 结果。

# ### 桥接项列表
# {bridge_list_text}

# ### 行为描述
# {behavior}

# ### 第一步：相关性判断
# 1. 逐一分析桥接项列表，识别出与行为描述语义相关的桥接项 ID（如 "B-01"）。
# 2. 只保留相关项，无关项不列入。
# 3. 若无相关项，此步结果为 []。

# ### 第二步：合规性判断
# 1. 针对第一步识别出的每个桥接项，判断该行为是否符合其描述要求：
#    - 符合 → 标记为 1
#    - 不符合 → 标记为 0
# 2. 生成最终结果，必须为标准 JSON 数组，格式如下：
#    [{{"bridge_id": "B-01", "SJ": 1}}, {{"bridge_id": "B-02", "SJ": 0}}]
   
# ### 说明：
# 1. 每个对象必须包含 "bridge_id" 和 "SJ" 两个字段，禁止省略任何字段
# 2. "SJ" 值必须为整数 0 或 1，不可为字符串或 null
# 3. 若无相关桥接项，输出 []

# # 最终结果："""



# ======================
# 自定义 Dataset
# ======================
class BridgeMatchDataset(Dataset):
    def __init__(self, data: List[str], tokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        print(f"\n📂 正在构建 {len(data)} 条样本...")
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
                print(f"  ✅ 示例: {behavior[:30]}...")

        print(f"✅ 成功构建 {len(self.examples)} 条有效样本")

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
            sep_token = "判断结果："
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
# 数据整理器
# ======================
def collate_fn(batch, tokenizer):
    batch = [b for b in batch if isinstance(b, dict) and "input_ids" in b]
    if not batch:
        return {}
    return tokenizer.pad(batch, padding=True, return_tensors="pt")


# ======================
# 安全解析 JSON
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
# 构建 bridge_id 到索引的映射
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
# 加载原始数据
# ======================
def load_data(file_path: str):
    data = []
    print(f"📄 正在加载数据文件: {file_path}")
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
                    print(f"⚠️ 第{line_idx}行: 缺少 behavior，跳过")
                    continue
                if not isinstance(labels, list):
                    labels = []
                data.append({"behavior": behavior, "compliance_labels": labels})
            except Exception as e:
                print(f"⚠️ 第{line_idx}行解析失败: {e}")
    print(f"✅ 共加载 {len(data)} 条数据")
    return data


# ======================
# 评估函数（针对单个 fold）
# ======================
def evaluate_fold(model_path: str, val_data: List[Dict], id_to_idx: Dict[str, int]) -> Dict[str, float]:
    print(f"🔍 评估模型: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
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
        print(f"❌ 模型加载失败: {e}")
        return {}

    num_classes = len(id_to_idx)
    print(f"✅ 共 {num_classes} 个桥接项类别")

    # 统计容器
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
                    # 确保 SJ 是整数 0 或 1
                    if isinstance(sj_val, (int, float)) and sj_val in (0, 1):
                        pred_labels[bid] = int(sj_val)
                    else:
                        print(f"⚠️ 无效的 SJ 值: {sj_val}, bridge_id={bid}")
                else:
                    print(f"⚠️ 缺失 bridge_id 或 SJ: {item}")
            else:
                print(f"⚠️ 非字典项被忽略: {item}")

        # 统计 per-bridge
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

        # 样本级 exact match
        pred_set = {(k, v) for k, v in pred_labels.items()}
        true_set = {(k, v) for k, v in true_labels.items()}
        if pred_set == true_set:
            sample_exact_match += 1

    # 计算 micro & macro
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
# 五折交叉验证主函数
# ======================
def train_kfold():
    print(f"🔁 开始 {N_SPLITS} 折交叉验证...")

    # 加载数据
    all_data = load_data(TRAIN_FILE)
    if len(all_data) == 0:
        raise ValueError("❌ 加载的数据为空，请检查数据文件路径和格式")

    # 构建 bridge_id 映射
    id_to_idx = build_id_to_index(BRIDGE_LIST_TEXT)
    all_bridge_ids = sorted(id_to_idx.keys())
    num_classes = len(all_bridge_ids)
    print(f"✅ 共 {num_classes} 个桥接项: {all_bridge_ids}")

    # 构建多标签矩阵 y
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

    # 初始化结果存储
    fold_results = []

    # Stratified K-Fold on multi-label using iterative split
    for fold in range(N_SPLITS):
        print(f"\n" + "="*60)
        print(f"📌 开始第 {fold+1}/{N_SPLITS} 折训练")
        print("="*60)

        OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"fold_{fold}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 手动实现 K 折：使用固定随机种子划分
        indices = np.arange(len(X))
        np.random.RandomState(SEED + fold).shuffle(indices)

        # 分割索引
        n_val = len(X) // N_SPLITS
        val_start = fold * n_val
        val_end = (fold + 1) * n_val if fold < N_SPLITS - 1 else len(X)
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        # 获取数据子集
        train_data = [X[i] for i in train_indices]
        val_data = [X[i] for i in val_indices]

        print(f"✅ 第 {fold+1} 折: 训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")

        # ✅ 统计每个 bridge_id 在训练集和验证集中的分布
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
    
        print("\n📊 桥接项在训练集和验证集中的分布情况：")
        print(f"{'桥接项':<8} {'训练集':<8} {'验证集':<8} {'合计':<8}")
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
        print(f"{'总计':<8} {sum(train_counts.values()):<8} {sum(val_counts.values()):<8} {total_all:<8}")

        # 保存数据用于评估
        with open(os.path.join(OUTPUT_DIR, "train_data.json"), 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with open(os.path.join(OUTPUT_DIR, "val_data.json"), 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 加载 tokenizer 和模型
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            low_cpu_mem_usage=False,        # ✅ 关键
            # offload_folder=None,           # ❌ 确保不要启用 offload（除非你明确需要）
            # offload_state_dict=False,      # ❌ 禁用 offload
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
            save_total_limit=1,               # 🔥 只保留最后一个 checkpoint
            eval_strategy="epoch",
            load_best_model_at_end=True,      # 加载最佳模型
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

        # 训练
        print(f"🚀 第 {fold+1} 折开始训练...")
        trainer.train()

        # 保存
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # 评估
        results = evaluate_fold(OUTPUT_DIR, val_data, id_to_idx)
        fold_results.append(results)

        print(f"✅ 第 {fold+1} 折完成，结果:")
        for k, v in results.items():
            print(f"    {k}: {v:.4f}")

    # ======================
    # 汇总所有 fold 结果
    # ======================
    print("\n" + "="*70)
    print("📊 4 折交叉验证最终结果汇总")
    print("="*70)

    df = pd.DataFrame(fold_results)
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col:<20} : {mean_val:.4f} ± {std_val:.4f}")

    print("\n📋 详细每折结果:")
    print(df.round(4))

    # 可选：保存结果到文件
    result_file = os.path.join(OUTPUT_BASE_DIR, "cv_results.csv")
    df.to_csv(result_file, index=False, float_format='%.4f')
    print(f"\n✅ 所有结果已保存至: {result_file}")


# ======================
# 主入口
# ======================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        print("❌ 交叉验证模式下不支持单独 evaluate。请运行完整训练。")
    else:
        train_kfold()