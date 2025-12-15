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
    从 JSON 文件或 JSON 字符串中加载行为描述（behavior 字段）

    Args:
        data_source (str): 文件路径 或 JSON 字符串内容
        is_file (bool): 是否为文件路径。True 表示是文件，False 表示是 JSON 字符串

    Returns:
        List[str]: 所有 behavior 字段组成的列表
    """
    try:
        # 步骤1: 获取 JSON 数据
        if is_file:
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = json.loads(data_source)

        # 步骤2: 验证数据格式是否为列表
        if not isinstance(data, list):
            raise ValueError("JSON 数据必须是一个列表")

        # 步骤3: 提取 behavior 字段
        behaviors = []
        for item in data:
            if isinstance(item, dict) and 'behavior' in item:
                behavior_text = item['behavior'].strip()
                if behavior_text:
                    behaviors.append(behavior_text)
                else:
                    print("⚠️ 跳过空 behavior 字段")
            else:
                print(f"⚠️ 忽略无效条目: {item}")

        return behaviors

    except Exception as e:
        raise RuntimeError(f"加载或解析数据失败: {str(e)}")

class BehaviorPredictor:
    def __init__(self, config_path="config.json"):
        """初始化预测器，加载配置和模型"""
        print("🔄 初始化 BehaviorPredictor...")
        self.config = self._load_config(config_path)
        self.bridge_list_text = self._load_bridges()
        self.model, self.tokenizer = self._load_model()
        self.prompt_template = self._get_prompt_template()
        self.max_length = self.config.get("MAX_LENGTH", 1024)
        print("✅ 初始化完成")

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"❌ 加载配置文件失败: {str(e)}")

    def _load_bridges(self):
        """加载桥接项列表"""
        bridges_file = self.config["BRIDGES_FILE"]
        try:
            with open(bridges_file, 'r', encoding='utf-8') as f:
                bridges = json.load(f)
            return "\n".join([f"- {b['ID']}: {b['SD']}" for b in bridges])
        except Exception as e:
            raise ValueError(f"❌ 加载桥接项文件失败: {str(e)}")

    def _get_prompt_template(self):
        """返回提示模板"""
        return """请根据以下行为描述，判断其与哪些桥接项相关，并对每个相关的桥接项判断是否符合其语义描述要求。

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

   

    def _load_model(self):
        """加载模型和分词器"""
        try:
            # 获取模型路径
            model_name = self.config["MODEL_PATHS"][self.config["MATCHING_MODEL_NAME"]]
            finetuned_dir = self.config["FINETUNED_MODEL_DIR"]

            print(f"🔍 加载基础模型: {model_name}")
            print(f"🔧 加载LoRA权重: {finetuned_dir}")

            # 加载LoRA配置
            peft_config = PeftConfig.from_pretrained(finetuned_dir)

            # 加载分词器
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # 加载LoRA适配器
            model = PeftModel.from_pretrained(base_model, finetuned_dir)
            model.eval()

            print("✅ 模型加载成功")
            return model, tokenizer

        except Exception as e:
            raise RuntimeError(f"❌ 模型加载失败: {str(e)}")

    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从模型输出中提取JSON结果，支持多行、缩进等格式"""
        # 匹配以 [ 开头、] 结尾的任意内容（非贪婪）
        json_pattern = r'\[\s*(?:\{[^}]*\}\s*,?\s*)*\]'
        match = re.search(json_pattern, text, re.DOTALL)
        if not match:
            print("⚠️ 未在输出中找到 JSON 数组")
            return []

        try:
            result = json.loads(match.group())
            return result if isinstance(result, list) else []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析失败: {e}")
            return []

    def predict(self, behavior: str) -> List[Dict[str, Any]]:
        """
        预测行为文本对应的桥接项匹配结果
        :param behavior: 待预测的行为文本
        :return: 匹配结果列表
        """
        if not behavior or not isinstance(behavior, str):
            raise ValueError("输入的行为文本必须是非空字符串")

        # 构建提示
        prompt = self.prompt_template.format(
            bridge_list_text=self.bridge_list_text,
            behavior=behavior.strip()
        )

        # 处理输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)

        print("🧠 正在生成预测...")
        # 模型生成
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

        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        print(f"📝 模型原始输出:\n{response}")

        # 提取 JSON
        result = self._extract_json_from_text(response)
        print(f"✅ 解析结果: {result}")
        return result

    # def predict_batch(self, behaviors: List[str]) -> List[Dict[str, Any]]:
    #     """
    #     批量预测多个行为文本（逐条处理，如需并行可进一步优化）
    #     :param behaviors: 行为文本列表
    #     :return: 预测结果列表
    #     """
    #     results = []
    #     for i, behavior in enumerate(behaviors):
    #         print(f"\n📌 处理第 {i+1}/{len(behaviors)} 条: {behavior}")
    #         try:
    #             result = self.predict(behavior)
    #             results.append({
    #                 "behavior": behavior,
    #                 "compliance_labels": result
    #             })
    #         except Exception as e:
    #             print(f"❌ 预测失败: {str(e)}")
    #             results.append({
    #                 "behavior": behavior,
    #                 "compliance_labels": [],
    #                 "error": str(e)
    #             })
    #     return results

    def predict_batch(self, behaviors: List[str], output_file="predict_result.jsonl") -> List[Dict[str, Any]]:
        """
        批量预测并保存每条结果
        """
        results = []
        for i, behavior in enumerate(behaviors):
            print(f"\n📌 处理第 {i+1}/{len(behaviors)} 条: {behavior}")
            try:
                # 直接调用封装好的保存方法
                self.predict_and_save(behavior, output_file=output_file)
                # 为了统一返回结构，我们也构建 result 记录
                result = self.predict(behavior)  # 注意：这里重复预测了！
                results.append({
                    "behavior": behavior,
                    "compliance_labels": result
                })
            except Exception as e:
                print(f"❌ 预测失败: {str(e)}")
                results.append({
                    "behavior": behavior,
                    "compliance_labels": [],
                    "error": str(e)
                })
                # 也保存错误
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "behavior": behavior,
                        "compliance_labels": [],
                        "error": str(e)
                    }, ensure_ascii=False) + '\n')
        return results

    def predict_and_save(self, behavior: str, output_file="predict_audit_resultwoese.jsonl") -> str:
        """
        预测行为文本并将结果以 JSONL 格式追加保存到文件
        每次调用写入一行
        """
        # 执行预测
        try:
            result = self.predict(behavior)
        except Exception as e:
            print(f"❌ 预测出错: {str(e)}")
            result = []

        # 构建输出数据
        save_data = {
            "behavior": behavior,
            "compliance_labels": result
        }

        # 以追加模式写入 JSONL 文件
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

        print(f"💾 预测结果已追加保存至: {output_file}")
        return output_file


# ========================
# 🚀 示例用法
# ========================
# if __name__ == "__main__":
#     try:
#         # 初始化预测器
#         predictor = BehaviorPredictor(config_path="config.json")

#         behaviors = load_behaviors.load_behaviors("test_data.json")
#         # behaviors = load_behaviors_from_json("text_audit_test.json", is_file=True)
#         print(f"✅ 行为文本加载结果: {behaviors}")
#         # 测试单条预测
#         # test_behavior = "用户注册时未提供隐私政策链接。"
#         # result_file = predictor.predict_and_save(test_behavior)
#         # print(f"✅ 单条预测结果已保存: {result_file}")

#         # 测试批量预测
#         # behaviors = [
#         #     "App在后台持续收集位置信息，未征得用户同意。",
#         #     "当平台计划公开用户“XX经验”时，弹窗提示“您的XX记录将被脱敏展示，是否同意？”，用户需单独点击“同意”按钮"
#         # ]

        
#         batch_results = predictor.predict_batch(behaviors)
#         print("\n📊 批量预测结果:")
#         for item in batch_results:
#             print(item)

#     except Exception as e:
#         print(f"❌ 执行出错: {str(e)}")