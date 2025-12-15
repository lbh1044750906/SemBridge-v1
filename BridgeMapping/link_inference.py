import json
import re
from collections import defaultdict
import sys

# ========== 1. 加载简化版知识图谱并构建 bridge → law_id 映射 ==========

def load_law_clauses_from_kg(kg_path: str) -> dict:
    """
    从简化版知识图谱中提取：
        bridge_id -> sorted list of law_id (by compliance_degree)
    只保留 relation_type in ["直接对应", "包含关系"]
    """
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)

    bridge_to_clauses = {}

    for item in kg["compliance_graph"]:
        bridge_id = item["bridge_id"]
        laws = item["related_laws"]

        # 过滤只保留“直接对应”和“包含关系”
        filtered = [
            {
                "law_id": f"{extract_abbr_from_law_name(law['law'])}-{law['article'].replace('第', '').replace('条', '')}",
                "compliance_degree": law["compliance_degree"]
            }
            for law in laws
            if law["relation"] in ["直接对应", "包含关系"]
        ]

        # 按合规关联度降序排序
        sorted_clauses = sorted(filtered, key=lambda x: x["compliance_degree"], reverse=True)

        # 提取 law_id 字符串列表
        clause_ids = [c["law_id"] for c in sorted_clauses]
        bridge_to_clauses[bridge_id] = clause_ids

    print(f"✅ 已从知识图谱加载 {len(bridge_to_clauses)} 个桥接项的法律条文映射")
    return bridge_to_clauses


import json
import re
from collections import defaultdict

def build_bridge_to_law_ids(kg_path: str) -> dict:
    """
    从 compliance_RKG_CoT.json（简化版图谱）构建：
        bridge_id → 按 compliance_degree 排序的 law_id 列表
    仅保留 relation_type in ["直接对应", "包含关系"]

    由于图谱中没有原始 law_id（如 PIPL-14），需通过：
        law_name + article → 反向推断 law_id
    """

    # ===== 1. 法律名称到缩写的反向映射表 =====
    LAW_NAME_TO_ABBR = {
        "《个人信息保护法》": "PIPL",
        "《个人信息保护合规审计管理办法》": "PIPCA",
        "《中华人民共和国网络安全法》": "CSL",
        "《网络数据安全管理条例》": "NDSMR",
        "《App违法违规收集使用个人信息行为认定方法》": "MICUPIA",
        "《中华人民共和国数据安全法》": "DSL",
        "《未成年人网络保护条例》": "RPMC",
        "《中华人民共和国宪法》": "Constitution",
        "GB/T 35273-2020": "GBT35273",
        "《中华人民共和国民法典》": "Cvilcode"
        # 可继续补充
    }

    # ===== 2. 加载简化版知识图谱 =====
    with open(kg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bridge_to_clauses = {}

    # ===== 3. 遍历 compliance_graph =====
    for item in data["compliance_graph"]:
        bridge_id = item["bridge_id"]
        related_laws = item["related_laws"]

        clause_list = []

        for law in related_laws:
            relation_type = law["relation"]
            if relation_type not in ["直接对应", "包含关系"]:
                continue

            law_name = law["law"].strip()
            article_id = str(law["article"]).strip()

            # 查找对应缩写
            abbr = LAW_NAME_TO_ABBR.get(law_name)
            if not abbr:
                print(f"⚠️ 未识别的法律名称：{law_name}，跳过")
                continue

            # 构造原始 law_id 格式：缩写-条款号
            # 注意：article_id 可能带“第X条”，需清理
            cleaned_article = re.sub(r"第|条|款|项", "", article_id).strip()
            if not cleaned_article:
                cleaned_article = "通用"

            law_id = f"{abbr}-{cleaned_article}"

            # 使用 compliance_degree 排序
            compliance_degree = law["compliance_degree"]

            clause_list.append({
                "law_id": law_id,
                "compliance_degree": compliance_degree
            })

        # 按合规度降序排序
        sorted_clauses = sorted(clause_list, key=lambda x: x["compliance_degree"], reverse=True)
        # 提取 law_id 列表
        bridge_to_clauses[bridge_id] = [c["law_id"] for c in sorted_clauses]

    print(f"✅ 已从 {kg_path} 构建 bridge → law_id 映射，共 {len(bridge_to_clauses)} 个桥接项")
    return bridge_to_clauses


# ========== 2. 处理输入 JSONL 文件 ==========
def process_behavior_jsonl(input_path: str, output_path: str, bridge_to_clauses: dict):
    """
    读取输入 JSONL，输出带 related_clauses 的 JSONL
     """
    output_data = []

    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"❌ 无法解析行：{line}, 错误: {e}")
                continue

            behavior = record.get("behavior", "")
            compliance_list = record.get("compliance_labels", [])

            # 处理空compliance_labels情况
            if not compliance_list:
                output_data.append({
                    "behavior": behavior,
                    "compliance_labels": 1,  # 默认设为合规
                    "related_clauses": []  # 注意：这里修正了拼写错误
                })
                continue

            # 提取 SJ 值，取最小值作为最终 compliance_labels
            try:
                sj_values = [item.get("SJ", 1) for item in compliance_list]  # 默认1（合规）
                final_compliance_label = min(sj_values)  # 只要有一个 SJ=0，就为0
            except (KeyError, TypeError) as e:
                print(f"⚠️ 处理SJ值失败: {e}, 行为: {behavior[:50]}, 数据: {compliance_list}")
                final_compliance_label = 1  # 默认设为合规

            # 收集所有 bridge_id 对应的 law_id
            all_clauses = []
            for item in compliance_list:
                try:
                    bridge_id = item.get("bridge_id")
                    if bridge_id:  # 确保bridge_id存在
                        clauses = bridge_to_clauses.get(bridge_id, [])
                        all_clauses.extend(clauses)
                except AttributeError:
                    continue

            # 去重，保持排序（按首次出现的顺序）
            seen = set()
            unique_clauses = []
            for cid in all_clauses:
                if cid and cid not in seen:  # 确保cid不为空且不重复
                    seen.add(cid)
                    unique_clauses.append(cid)

            output_data.append({
                "behavior": behavior,
                "compliance_labels": final_compliance_label,
                "related_clauses": unique_clauses  # 注意：这里修正了拼写错误
            })

    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in output_data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 已处理完成，输出文件：{output_path}")
    print(f"   共 {len(output_data)} 条记录")


# ========== 3. 主函数 ==========

# if __name__ == "__main__":
#     # 输入输出路径
#     RELATION_DATA_PATH = "compliance_RKG_CoT.json"        # 构建的RKG数据
#     INPUT_JSONL_PATH = "predict_result.json"       # 预测的桥接项结果文件
#     OUTPUT_JSONL_PATH = "output_link_compliance_result.json"   # 结果输出文件
#
#     print("=== 开始加载桥接项到法律条文的映射 ===")
#     bridge_to_clauses = build_bridge_to_law_ids(RELATION_DATA_PATH)
#
#     print("\n=== 开始处理行为数据 ===")
#     process_behavior_jsonl(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH, bridge_to_clauses)
#
#     print("\n🎉 处理完成！")
