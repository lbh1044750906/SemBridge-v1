import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from tqdm import tqdm
import os
import re
import traceback
from json_repair import repair_json

# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


class LocalLLMBridgeLawMatcher:
    def __init__(self, model_path: str):
        """
        初始化本地LLM法律关联分析器
        :param model_path: 本地LLM模型路径
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)  # 提取模型文件夹名作为名称
        print(f"🔄 正在加载模型：{self.model_name} 来自 {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",  # 改为 auto 更通用
            trust_remote_code=True
        )
        self.model.eval()
        self.related_types = ["直接对应", "包含关系", "补充关系", "无关联"]

    def extract_json_strict(self, text: str) -> dict:
        """
        使用 json-repair 健壮提取并修复 JSON
        """
        # 优先提取 ```json ... ``` 中的内容
        code_block_match = re.search(r"```json\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            try:
                repaired = repair_json(code_block_match.group(1), return_objects=True)
                if isinstance(repaired, dict):
                    return repaired
            except:
                pass
    
        # 尝试直接修复整个文本
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception as e:
            pass
    
        # 最后 fallback：查找 {.*?} 并尝试修复
        json_match = re.search(r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{.*?\}", text, re.DOTALL)
        if json_match:
            try:
                repaired = repair_json(json_match.group(), return_objects=True)
                if isinstance(repaired, dict):
                    return repaired
            except:
                pass
    
        return None

    def generate_relation(self, bridge_text: str, law_text: str) -> dict:
        if not bridge_text.strip() or not law_text.strip():
            return {"关联类型": "无关联", "推理依据": "输入文本为空"}
    
        system_prompt = "你是专注于法律文本语义匹配的资深专家..."
        user_prompt = f"""..."""
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
        try:
            # === 1. Tokenize 输入，获取 input_ids ===
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(device)
    
            input_length = inputs["input_ids"].shape[1]
    
            # === 2. 自动获取模型最大上下文长度 ===
            model_max_length = self.tokenizer.model_max_length
            # 更可靠的方式：从 config 获取
            if hasattr(self.model.config, "max_position_embeddings"):
                model_max_length = self.model.config.max_position_embeddings
            elif hasattr(self.model.config, "seq_length"):
                model_max_length = self.model.config.seq_length
            elif hasattr(self.tokenizer, "model_max_length"):
                model_max_length = self.tokenizer.model_max_length
    
            # 安全兜底
            model_max_length = min(model_max_length, 131072)  # 防止某些 config 返回 1000000
            model_max_length = max(model_max_length, 2048)
    
            # === 3. 计算最大可生成 token 数 ===
            max_possible_new_tokens = model_max_length - input_length
    
            if max_possible_new_tokens <= 0:
                return {
                    "关联类型": "无关联",
                    "推理依据": f"输入过长，超出模型上下文限制（模型最大长度: {model_max_length}, 输入长度: {input_length}）"
                }
    
            # 设置实际生成上限：最多生成 2048 个新 token，且不超过剩余空间
            actual_max_new_tokens = min(2048, max_possible_new_tokens)
    
            # === 4. 生成输出 ===
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=actual_max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
    
            llm_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
            # === 5. 检查是否被截断 ===
            generated_tokens = outputs.shape[1] - input_length
            is_truncated = (generated_tokens >= actual_max_new_tokens)
    
            # === 6. 提取 JSON ===
            relation = self.extract_json_strict(llm_output)
    
            if relation is None:
                reasoning = "未能解析出有效JSON"
                if is_truncated:
                    reasoning += f"，且输出已被截断（生成 {generated_tokens}/{actual_max_new_tokens} tokens）"
                reasoning += f"，原始输出片段：{llm_output[:300]}..."
                return {"关联类型": "无关联", "推理依据": reasoning}
    
            # === 7. 验证字段 ===
            if not isinstance(relation, dict):
                return {"关联类型": "无关联", "推理依据": "解析结果不是字典"}
    
            if "关联类型" not in relation or "推理依据" not in relation:
                return {"关联类型": "无关联", "推理依据": "JSON缺少必要字段"}
    
            if relation["关联类型"] not in self.related_types:
                return {
                    "关联类型": "无关联",
                    "推理依据": f"模型输出类型异常（{relation['关联类型']}），仅允许：{self.related_types}"
                }
    
            # ✅ 成功返回
            result = {
                "关联类型": relation["关联类型"],
                "推理依据": relation["推理依据"]
            }
    
            # 可选：添加长度警告
            if generated_tokens >= actual_max_new_tokens * 0.9:
                result["推理依据"] += f" [⚠️ 接近生成上限({generated_tokens}/{actual_max_new_tokens})]"
    
            return result
    
        except Exception as e:
            return {
                "关联类型": "无关联",
                "推理依据": f"LLM生成失败：{str(e)}\n{traceback.format_exc()}"
            }
        
    def generate_relation(self, bridge_text: str, law_text: str) -> dict:
        """
        调用本地LLM生成桥接项与法律条文的关联分析（带健壮JSON解析）
        """
        if not bridge_text.strip() or not law_text.strip():
            return {"关联类型": "无关联", "推理依据": "输入文本为空"}
    
        system_prompt = (
            "你是专注于法律文本语义匹配的资深专家，核心任务是基于法律条文与桥接项的文本内容，判定二者的语义关联类型。"
            "请基于事实分析，避免主观推测。若无法确定关联，请优先归类为无关联。"
        )
        
        # user_prompt = f"""
        # 请根据以下规则判断关联类型，并严格按指定格式输出JSON结果：
        
        # ### 一、关联类型定义（唯一判定标准）
        # 1. 直接对应：桥接项的核心语义与法律条文某一条款的核心要求完全匹配（允许表述措辞略有差异，但核心约束/规则无偏差）；
        # 2. 包含关系：桥接项的核心要求仅覆盖法律条文多个条款/部分内容中的某一项，即桥接项语义范围是法律条文的子集；
        # 3. 补充关系：桥接项的要求未在法律条文中直接提及，但可通过法律条文的基本原则、隐含逻辑推导得出，属于条文的场景化细化或操作层延伸；
        # 4. 无关联：桥接项与法律条文在语义主题、约束对象、核心逻辑上均无交集。
        
        # ### 二、输出要求（必须严格遵守）
        # 1. 关联类型仅可选上述四类之一，不允许新增或修改类型名称；
        # 2. 最终输出为JSON格式内容，中文表述，JSON格式为：{{"关联类型": "类型名称", "推理依据": "简要描述语义和逻辑对比"}}
        
        # ### 三、当前判定任务
        # 桥接项：{bridge_text}
        # 法律条文：{law_text}
        # 输出："""
        
        # user_prompt = f"""
        # 请根据以下规则判断关联类型，严格按以下步骤执行，并输出JSON格式结果：
        
        # 注意：
        # - 只输出一个 JSON 对象，格式为：{{"关联类型": "...", "推理依据": "..."}}
        # - JSON 中的“推理依据”必须基于你的分析总结，禁止复制示例中的句子
        # - 如果无法确定，输出 {{"关联类型": "无关联", "推理依据": "无法确定关联性"}}
    
        # ### 关联类型定义
        # 1. 直接对应：桥接项的核心语义与法律条文的核心要求基本完全匹配（允许表述措辞略有差异，但核心约束/规则无偏差）；
        # 2. 包含关系：桥接项的核心要求仅覆盖法律条文的部分条款/部分内容中的某一项，即桥接项语义范围是法律条文的子集；
        # 3. 补充关系：桥接项的要求未在法律条文中直接提及，但可通过法律条文的基本原则、隐含逻辑推导得出，属于条文的场景化细化或操作层延伸；
        # 4. 无关联：桥接项与法律条文在语义主题、约束对象、核心逻辑上均无交集。
    
        # ### 示例（Few-Shot）
        # 示例1（直接对应）：
        # 桥接项：应采取相应安全技术措施实现个人信息的保密性、完整性、可用性
        # 法律条文：《个人信息保护法》第9条：个人信息处理者应当采取加密、去标识化等安全技术措施，确保个人信息安全。
        # 输出：{{"关联类型": "直接对应", "推理依据": "桥接项'采取相应安全措施'与法律条文'采取加密、去标识化等安全技术措施'完全匹配"}}
    
        # 示例2（包含关系）：
        # 桥接项：基于个人同意处理个人信息时，应该取得个人同意，且该同意由个人在充分知情的前提下自愿、明确作出；
        # 法律条文：《个人信息保护法》第13条：符合下列情形之一的，个人信息处理者方可处理：（一）取得个人的同意；（二）为订立、履行个人作为一方当事人的合同所必需...（包含多个条款）
        # 输出：{{"关联类型": "包含关系", "推理依据": "桥接项'取得个人同意'是法律条文第13条（一）的子集，法律条文还包含其他情形"}}
    
        # 示例3（补充关系）：
        # 桥接项：基于个人同意处理个人信息时，应该取得个人同意，且该同意由个人在充分知情的前提下自愿、明确作出；
        # 法律条文：《个人信息保护法》第5条：处理个人信息应当遵循合法、正当、必要和诚信原则，不得通过误导、欺诈、胁迫等方式处理个人信息。
        # 输出：{{"关联类型": "补充关系", "推理依据": "法律条文强调了处理个人信息时应遵循的原则，包括合法、正当、必要和诚信，而桥接项具体说明了取得个人同意且该同意由个人在充分知情的前提下自愿、明确作出，这体现了法律条文中的合法、正当、诚信原则，是该条文上的要求延伸和细化"}}
        
        # ### 当前任务
        # 桥接项：{bridge_text}
        # 法律条文：{law_text}
        # 输出："""
        

        user_prompt = f"""
        请根据以下规则判断关联类型，严格按以下规则执行，并输出JSON格式结果：

        ### 关联类型定义
        1. 直接对应：桥接项的核心语义与法律条文某一条款的核心要求完全匹配（允许表述措辞略有差异，但核心约束/规则无偏差）；
        2. 包含关系：桥接项的核心要求仅覆盖法律条文多个条款/部分内容中的某一项，即桥接项语义范围是法律条文的子集；
        3. 补充关系：桥接项的要求未在法律条文中直接提及，但可通过法律条文的基本原则、隐含逻辑推导得出，属于条文的场景化细化或操作层延伸；
        4. 无关联：桥接项与法律条文在语义主题、约束对象、核心逻辑上均无交集。
    
        ### 分析步骤
        ### 步骤1：分析
        请先分析桥接项与法律条文的语义关系，回答以下问题：
        1). 桥接项的核心要求是什么？
        2). 法律条文的关键内容是什么？
        
        ### 步骤2：判断
        根据分析，对照上述定义判断桥接项与法律条文的关联类型（直接对应/包含关系/补充关系/无关联）。
        
        ### 步骤3：输出
        - 只输出一个 JSON 对象，格式为：{{"关联类型": "...", "推理依据": "..."}}
        - 如果无法确定，输出 {{"关联类型": "无关联", "推理依据": "无法确定关联性"}}

        ### 示例（Few-Shot）
        示例1（直接对应）：
        桥接项：应采取相应安全技术措施实现个人信息的保密性、完整性、可用性
        法律条文：《个人信息保护法》第9条：个人信息处理者应当采取加密、去标识化等安全技术措施，确保个人信息安全。
        输出：{{"关联类型": "直接对应", "推理依据": "桥接项'采取相应安全措施'与法律条文'采取加密、去标识化等安全技术措施'完全匹配"}}
    
        示例2（包含关系）：
        桥接项：基于个人同意处理个人信息时，应该取得个人同意，且该同意由个人在充分知情的前提下自愿、明确作出；
        法律条文：《个人信息保护法》第13条：符合下列情形之一的，个人信息处理者方可处理：（一）取得个人的同意；（二）为订立、履行个人作为一方当事人的合同所必需...（包含多个条款）
        输出：{{"关联类型": "包含关系", "推理依据": "桥接项'取得个人同意'是法律条文第13条（一）的子集，法律条文还包含其他情形"}}
    
        示例3（补充关系）：
        桥接项：基于个人同意处理个人信息时，应该取得个人同意，且该同意由个人在充分知情的前提下自愿、明确作出；
        法律条文：《个人信息保护法》第5条：处理个人信息应当遵循合法、正当、必要和诚信原则，不得通过误导、欺诈、胁迫等方式处理个人信息。
        输出：{{"关联类型": "补充关系", "推理依据": "法律条文强调了处理个人信息时应遵循的原则，包括合法、正当、必要和诚信，而桥接项具体说明了取得个人同意且该同意由个人在充分知情的前提下自愿、明确作出，这体现了法律条文中的合法、正当、诚信原则，是该条文上的要求延伸和细化"}}
    
        ### 当前任务
        桥接项：{bridge_text}
        法律条文：{law_text}
        输出："""

    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
    
            outputs = self.model.generate(
                inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.0,
                top_p=1
            )
    
            llm_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(f"📝 LLM 原始输出:\n{llm_output}")
    
            # 【关键修复】使用健壮的 JSON 提取函数
            relation = self.extract_json_strict(llm_output)
    
            if relation is None:
                return {
                    "关联类型": "无关联",
                    "推理依据": f"未能解析出有效JSON，原始输出：{llm_output[:200]}..."
                }
    
            # 验证字段
            if not isinstance(relation, dict):
                return {"关联类型": "无关联", "推理依据": "解析结果不是字典"}
    
            if "关联类型" not in relation or "推理依据" not in relation:
                return {
                    "关联类型": "无关联",
                    "推理依据": f"JSON缺少必要字段，原始输出：{llm_output[:200]}..."
                }
    
            if relation["关联类型"] not in self.related_types:
                relation["关联类型"] = "无关联"
                relation["推理依据"] = f"模型输出类型异常（{relation['关联类型']}），已修正为无关联"
                
            return relation
    
        except Exception as e:
            return {
                "关联类型": "无关联",
                "推理依据": f"LLM生成或解析失败：{str(e)}\n{traceback.format_exc()}"
            }

    def batch_analyze(self, bridge_items: list, law_articles: list) -> list:
        """
        批量分析桥接项与法律条文的关联（带进度和耗时）
        """
        results = []
        total_start_time = datetime.now()

        print(f"🔍 开始批量分析：共 {len(bridge_items)} 个桥接项 × {len(law_articles)} 条法律条文")

        # 外层：遍历桥接项
        for idx, bridge in enumerate(tqdm(bridge_items, desc="📊 分析桥接项", unit="项")):
            bridge_id = bridge.get("ID") or bridge.get("id")
            bridge_text = bridge.get("SD") or bridge.get("sd") or ""

            if not bridge_id or not bridge_text.strip():
                tqdm.write(f"⚠️ 跳过无效桥接项（ID缺失或内容为空）: {bridge}")
                continue

            start_time = datetime.now()
            tqdm.write(f"✅ 开始分析桥接项: {bridge_id} ({idx + 1}/{len(bridge_items)})")

            bridge_result = {
                "bridge_id": bridge_id,
                "bridge_sd": bridge_text,
                "relations": []
            }

            # 内层：遍历所有法律条文
            for law in law_articles:
                law_id = law["CID"]
                law_text = law["Cont"]
                relation = self.generate_relation(bridge_text, law_text)
                bridge_result["relations"].append({
                    "law_id": law_id,
                    "law_text": law_text,
                    "relation_type": relation["关联类型"],
                    "reasoning": relation["推理依据"]
                })

            # 排序：直接对应 > 包含关系 > 补充关系 > 无关联
            bridge_result["relations"] = sorted(
                bridge_result["relations"],
                key=lambda x: self.related_types.index(x["relation_type"]) if x["relation_type"] in self.related_types else 3
            )

            results.append(bridge_result)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            tqdm.write(f"⏱️  桥接项 {bridge_id} 分析完成，耗时 {duration:.2f} 秒\n")

        total_duration = (datetime.now() - total_start_time).total_seconds()
        print(f"✅ 模型 [{self.model_name}] 分析完成！总计 {len(results)} 个桥接项，总耗时 {total_duration:.2f} 秒")
        print(f"🚀 平均每项耗时: {total_duration / len(results):.2f} 秒")

        return results

    def save_results(self, results: list, save_path: str):
        """保存分析结果到JSON文件"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 分析结果已保存至：{save_path}")


# ================================
# 主程序：依次测试多个模型
# ================================
if __name__ == "__main__":
    # 定义模型路径列表
    model_paths = [
        "./Qwen/Qwen/Qwen2.5-7B-Instruct",
        # "./Deepseek/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        # "./Qwen/Qwen/Qwen3-8B",
        # "./Qwen/Qwen/Qwen3-14B",
        # "./Llama/LLM-Research/Meta-Llama-3.1-8B-Instruct",
        # "./Llama/FlagAlpha/Llama3-Chinese-8B-Instruct",
        # "./Qwen/Qwen/Qwen2.5-14B-Instruct",
        # "./Deepseek/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        # " ./Llama/luohuan02/ShengbinYue-LawLLM-7B"
    ]

    # 加载桥接项和法律条文
    with open('../bridge_items.json', 'r', encoding='utf-8') as file:
        bridge_items = json.load(file)
    with open('policies_all.json', 'r', encoding='utf-8') as  file:
        law_articles = json.load(file)

    # 加载案例
  #   bridge_items = [
  #       {
  #         "ID": "B-01",
  #         "L1": "处理个人信息合法性基础",
  #         "L2": "知情同意",
  #         "SD": "基于个人同意处理个人信息时，应该取得个人同意，且该同意由个人在充分知情的前提下自愿、明确作出。",
  #         "SJ": ""
  #       }]
  #   law_articles = [  {
  #       "CID": "PIPL-12",
  #       "PName": "《中华人民共和国个人信息保护法》",
  #       "Cont": "国家积极参与个人信息保护国际规则的制定，促进个人信息保护方面的国际交流与合作，推动与其他国家、地区、国际组织之间的个人信息保护规则、标准等互认。"
  #     },
  #     {
  #       "CID": "PIPL-13",
  #       "PName": "《中华人民共和国个人信息保护法》",
  #       "Cont": "符合下列情形之一的，个人信息处理者方可处理个人信息： （一）取得个人的同意； （二）为订立、履行个人作为一方当事人的合同所必需，或者按照依法制定的劳动规章制度和依法签订的集体合同实施人力资源管理所必需； （三）为履行法定职责或者法定义务所必需； （四）为应对突发公共卫生事件，或者紧急情况下为保护自然人的生命健康和财产安全所必需； （五）为公共利益实施新闻报道、舆论监督等行为，在合理的范围内处理个人信息； （六）依照本法规定在合理的范围内处理个人自行公开或者其他已经合法公开的个人信息； （七）法律、行政法规规定的其他情形。 依照本法其他有关规定，处理个人信息应当取得个人同意，但是有前款第二项至第七项规定情形的，不需取得个人同意。"
  #     },  {
  #   "CID": "PIPL-17",
  #   "PName": "《中华人民共和国个人信息保护法》",
  #   "Cont": "个人信息处理者在处理个人信息前，应当以显著方式、清晰易懂的语言真实、准确、完整地向个人告知下列事项： （一）个人信息处理者的名称或者姓名和联系方式； （二）个人信息的处理目的、处理方式，处理的个人信息种类、保存期限； （三）个人行使本法规定权利的方式和程序； （四）法律、行政法规规定应当告知的其他事项。 前款规定事项发生变更的，应当将变更部分告知个人。 个人信息处理者通过制定个人信息处理规则的方式告知第一款规定事项的，处理规则应当公开，并且便于查阅和保存。"
  # },  {
  #   "CID": "PIPL-14",
  #   "PName": "《中华人民共和国个人信息保护法》",
  #   "Cont": "基于个人同意处理个人信息的，该同意应当由个人在充分知情的前提下自愿、明确作出。法律、行政法规规定处理个人信息应当取得个人单独同意或者书面同意的，从其规定。 个人信息的处理目的、处理方式和处理的个人信息种类发生变更的，应当重新取得个人同意。"
  # },]
    
    print(f"📌 共加载 {len(bridge_items)} 个桥接项 和 {len(law_articles)} 条法律条文")

    # 依次遍历每个模型
    for model_path in model_paths:
        try:
            # 创建匹配器
            matcher = LocalLLMBridgeLawMatcher(model_path=model_path)

            # 批量分析
            print("🚀 开始本地LLM批量分析桥接项-法律条文关联...")
            results = matcher.batch_analyze(bridge_items, law_articles)

            # 生成输出文件名：Qwen2.5-7B-Instruct_result.json
            model_name = os.path.basename(model_path)
            output_filename = f"./{model_name}_COT_mapping_result.json"
            matcher.save_results(results, save_path=output_filename)

            print(f"✅ [{model_name}] 测试完成，结果已保存为：{output_filename}\n")
            print("-" * 60)

        except Exception as e:
            print(f"❌ 模型 [{model_path}] 测试失败：{str(e)}\n")
            continue

    print("🎉 所有模型测试完成！")