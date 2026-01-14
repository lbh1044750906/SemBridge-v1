# BridgeMatching ---> Load the matching prediction model
import json
import torch
import re
import os
from datetime import datetime
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import load_behaviors

def load_behaviors_from_json(data_source: str, is_file: bool = True) -> List[str]:
    """
    Load behaviour descriptions (the behaviour field) from JSON files or JSON strings.
    Args:
        data_source (str): File path or JSON string content
        is_file (bool): Whether it is a file path. True indicates a file, False indicates a JSON string.
    Returns:
        List[str]: The list comprising all behaviour fields
    """
    try:
        # Step 1: Retrieve JSON data
        if is_file:
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(data_source)

        # Step 2: Verify that the data format is a list
        if not isinstance(data, list):
            raise ValueError("The JSON data must be a list.")

        # Step 3: Extract the behaviour field
        behaviors = []
        for item in data:
            if isinstance(item, dict) and 'behavior' in item:
                behavior_text = item['behavior'].strip()
                if behavior_text:
                    behaviors.append(behavior_text)
                else:
                    print("⚠️ Skip empty behaviour fields")
            else:
                print(f"⚠️ Ignore invalid data: {item}")

        return behaviors

    except Exception as e:
        raise RuntimeError(f"Failed to load or parse data: {str(e)}")

class BehaviorPredictor:
    def __init__(self, config_path="config.json"):
        """load configuration and model"""
        print("🔄 Initialise BehaviorPredictor...")
        self.config = self._load_config(config_path)
        self.bridge_list_text = self._load_bridges()
        self.model, self.tokenizer = self._load_model()
        self.prompt_template = self._get_prompt_template()
        self.max_length = self.config.get("MAX_LENGTH", 1024)
        print("✅ Initialisation complete")

    def _load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"❌ Failed to load configuration file: {str(e)}")

    def _load_bridges(self):
        """Load the list of bridging items"""
        bridges_file = self.config["BRIDGES_FILE"]
        try:
            with open(bridges_file, 'r', encoding='utf-8') as f:
                bridges = json.load(f)
            return "\n".join([f"- {b['ID']}: {b['SD']}" for b in bridges])
        except Exception as e:
            raise ValueError(f"❌ Failed to load bridge item file: {str(e)}")

    def _get_prompt_template(self):
        """Return prompt template"""
        # English
        prompt_template = """[System Prompt]
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
        prompt_template = """请根据以下行为描述，判断其与哪些桥接项相关，并对每个相关的桥接项判断是否符合其语义描述要求。
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

        return prompt_template
   

    def _load_model(self):
        """Load the model and tokeniser"""
        try:
            # Obtain the model path
            model_name = self.config["MODEL_PATHS"][self.config["MATCHING_MODEL_NAME"]]
            finetuned_dir = self.config["FINETUNED_MODEL_DIR"]

            print(f"🔍 Load the base model: {model_name}")
            print(f"🔧 Loading LoRA weights: {finetuned_dir}")

            # Load LoRA configuration
            peft_config = PeftConfig.from_pretrained(finetuned_dir)

            # Load the word segmenter
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load the base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Load LoRa adapter
            model = PeftModel.from_pretrained(base_model, finetuned_dir)
            model.eval()

            print("✅ Model loaded successfully!")
            return model, tokenizer

        except Exception as e:
            raise RuntimeError(f"❌ Model loading failed: {str(e)}")

    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON results from model outputs, supporting multi-line formatting, indentation, and other conventions."""
        # Match any content beginning with [ and ending with ] (non-greedy)
        json_pattern = r'\[\s*(?:\{[^}]*\}\s*,?\s*)*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if not match:
            print("⚠️ No JSON array found in the output.")
            return []

        try:
            result = json.loads(match.group())
            return result if isinstance(result, list) else []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
            return []

    def predict(self, behavior: str) -> List[Dict[str, Any]]:
        """
        Predicted matching results for bridging items corresponding to event text
        :param behavior: event text to be predicted
        :return: List of matching results
        """
        if not behavior or not isinstance(behavior, str):
            raise ValueError("The input event text must be a non-empty string.")

        # Prompt construction
        prompt = self.prompt_template.format(
            bridge_list_text=self.bridge_list_text,
            behavior=behavior.strip()
        )

        # Processing input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)

        print("🧠 Generating prediction...")
        # Model generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                # temperature=0.1,
                # top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        print(f"📝 Model raw output:\n{response}")

        # Extract JSON
        result = self._extract_json_from_text(response)
        print(f"✅ Analysis results: {result}")
        return result

    def predict_batch(self, behaviors: List[str], output_file="predict_result.jsonl") -> List[Dict[str, Any]]:
        """
        Batch predict and save each result
        """
        results = []
        for i, behavior in enumerate(behaviors):
            print(f"\n📌 Processing the {i+1}/{len(behaviors)}th entry: {behavior}")
            try:
                # Call the encapsulated save method
                self.predict_and_save(behavior, output_file=output_file)
                # Standardised output structure
                result = self.predict(behavior)
                results.append({
                    "behavior": behavior,
                    "compliance_labels": result
                })
            except Exception as e:
                print(f"❌ Prediction failure: {str(e)}")
                results.append({
                    "behavior": behavior,
                    "compliance_labels": [],
                    "error": str(e)
                })
                # Save error
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "behavior": behavior,
                        "compliance_labels": [],
                        "error": str(e)
                    }, ensure_ascii=False) + '\n')
        return results

    def predict_and_save(self, behavior: str, output_file="predict_audit_result.jsonl") -> str:
        """
        Predict behavioural text and append the results to a file in JSONL format.
        Each call writes one line
        """
        # Implementing the prediction
        try:
            result = self.predict(behavior)
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            result = []

        # Constructing output data
        save_data = {
            "behavior": behavior,
            "compliance_labels": result
        }

        # Write to JSONL files in append mode
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

        print(f"💾 The prediction results have been saved to: {output_file}")
        return output_file

