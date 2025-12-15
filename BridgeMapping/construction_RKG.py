import json
import re
from collections import defaultdict
from datetime import datetime

# ========== 1. 核心配置 ==========

# 1.1 法律层级权重
LAW_HIERARCHY_CONFIG = {
    "宪法": {"level": 5, "rule_weight": 1.0},
    "法律": {"level": 4, "rule_weight": 0.8},
    "行政法规": {"level": 3, "rule_weight": 0.6},
    "地方性法规": {"level": 2, "rule_weight": 0.4},
    "其他": {"level": 1, "rule_weight": 0.3}
}

# 1.2 关系类型权重
RELATION_WEIGHT = {
    "直接对应": 1.0,
    "包含关系": 0.8,
    "补充关系": 0.6,
    "无关联": 0.0
}

# 1.3 【关键】法律缩写映射表（你只需维护这个）
LAW_ABBR_MAP = {
    "PIPL": {
        "name": "《个人信息保护法》",
        "level": "法律",
        "effective_date": "2021-11"
    },
    "PIPCA": {
        "name": "《个人信息保护合规审计管理办法》",
        "level": "其他",
        "effective_date": "2025-05"
    },
    "CSL": {
        "name": "《中华人民共和国网络安全法》",
        "level": "法律",
        "effective_date": "2017-06"
    },
    "NDSMR": {
        "name": "《网络数据安全管理条例》",
        "level": "行政法规",
        "effective_date": "2025-01"
    },
    "MICUPIA": {
        "name": "《App违法违规收集使用个人信息行为认定方法》",
        "level": "行政法规",
        "effective_date": "2019-11"
    },
    "DSL": {
        "name": "《中华人民共和国数据安全法》",
        "level": "法律",
        "effective_date": "2021-09"
    },
    "RPMC": {
        "name": "《未成年人网络保护条例》",
        "level": "行政法规",
        "effective_date": "2024-01"
    },
    "Constitution": {
        "name": "《中华人民共和国宪法》",
        "level": "宪法",
        "effective_date": "2018-11"
    },
    "GBT35273": {
        "name": "GB/T 35273-2020",
        "level": "其他",
        "effective_date": "2020-01"
    },
    "Cvilcode": {
        "name": "《中华人民共和国民法典》",
        "level": "法律",
        "effective_date": "2021-01"
    }
}

# 1.4 桥接项合规属性
BRIDGE_COMPLIANCE_ATTR = {
    "B001": "不合规",
    "B002": "合规"
}

# 正则模式：匹配 LAW-123 格式
# LAW_ID_PATTERN = re.compile(r"^([A-Z]+)-(\d+)$")


# ========== 2. 核心函数：从 law_id 解析并提取法律信息 ==========

# 更新正则：支持更广泛的 law_id 格式
LAW_ID_PATTERN = re.compile(r"^([A-Za-z0-9]+)-(.+)$")


def extract_law_info_by_id(law_id: str) -> dict:
    """
    从 law_id（如 GBT35273-11.5 或 Cvilcode-1038）中提取：
      - 前缀（用于查表）
      - 后缀（直接作为 article_id）
    """
    law_id = law_id.strip()
    match = LAW_ID_PATTERN.match(law_id)
    if not match:
        print(f"⚠️ 警告：law_id '{law_id}' 无法解析，格式应为 '前缀-后缀' 形式。")
        return {
            "name": "未知法律",
            "article_id": "未知条款",
            "level": "其他",
            "effective_date": "2000-01-01",
            "rule_weight": 0.2
        }

    abbr, article_suffix = match.groups()
    article_id = f"{article_suffix}"  # 直接使用后缀作为条款标识

    # 查表获取法律元数据（基于前缀）
    law_meta = LAW_ABBR_MAP.get(abbr)
    if not law_meta:
        print(f"⚠️ 警告：未识别的法律缩写 '{abbr}'，使用默认值。")
        return {
            "name": f"《{abbr}》",
            "article_id": article_id,
            "level": "其他",
            "effective_date": "2000-01-01",
            "rule_weight": 0.2
        }

    level = law_meta["level"]
    rule_weight = LAW_HIERARCHY_CONFIG.get(level, LAW_HIERARCHY_CONFIG["其他"])["rule_weight"]

    return {
        "name": law_meta["name"],
        "article_id": article_id,
        "level": level,
        "effective_date": law_meta["effective_date"],
        "rule_weight": rule_weight
    }

# ========== 3. 构建知识图谱（保持不变）==========
def build_compliance_knowledge_graph(relation_data: list) -> dict:
    kg = {
        "nodes": {
            "bridge": [],
            "law": []
        },
        "edges": []
    }
    law_node_id_map = {}  # (name + article_id) → node_id

    for bridge_item in relation_data:
        bridge_id = bridge_item["bridge_id"]
        bridge_ssd = bridge_item["bridge_sd"]
        bridge_compliance = BRIDGE_COMPLIANCE_ATTR.get(bridge_id, "未知")

        # 添加桥接项节点（去重）
        if not any(b["id"] == bridge_id for b in kg["nodes"]["bridge"]):
            kg["nodes"]["bridge"].append({
                "id": bridge_id,
                "sd": bridge_ssd,
                "compliance_attr": bridge_compliance
            })

        for relation in bridge_item["relations"]:
            relation_type = relation["relation_type"]
            if relation_type == "无关联":
                continue

            law_id = relation["law_id"]
            relation_weight = RELATION_WEIGHT[relation_type]

            # 使用新函数解析 law_id
            law_info = extract_law_info_by_id(law_id)
            law_name = law_info["name"]
            law_article_id = law_info["article_id"]
            law_node_key = f"{law_name}_{law_article_id}"

            # 添加法律节点（去重）
            if law_node_key not in law_node_id_map:
                law_node_id = f"law_{len(kg['nodes']['law']) + 1}"
                law_node_id_map[law_node_key] = law_node_id
                kg["nodes"]["law"].append({
                    "id": law_node_id,
                    "name": law_name,
                    "article_id": law_article_id,
                    "level": law_info["level"],
                    "effective_date": law_info["effective_date"],
                    "rule_weight": law_info["rule_weight"]
                })
            else:
                law_node_id = law_node_id_map[law_node_key]

            # 计算合规关联度
            compliance_degree = round(relation_weight * law_info["rule_weight"] * 5, 2)

            # 添加边（去重）
            if not any(e["source"] == bridge_id and e["target"] == law_node_id for e in kg["edges"]):
                kg["edges"].append({
                    "source": bridge_id,
                    "target": law_node_id,
                    "relation_type": relation_type,
                    "compliance_degree": compliance_degree
                })

    return kg


# ========== 4. 输出简化版图谱（保持不变）==========
def save_simplified_kg_by_bridge(kg: dict, output_path: str):
    law_node_map = {node["id"]: node for node in kg["nodes"]["law"]}
    bridge_edges_map = defaultdict(list)
    for edge in kg["edges"]:
        bridge_edges_map[edge["source"]].append(edge)

    simplified_kg = {
        "metadata": {
            "description": "合规关系知识图谱（简化版）",
            "filter": "仅保留直接对应、包含关系、补充关系",
            "sort_rule": "按合规关联度降序排列",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "compliance_graph": []
    }

    for bridge in kg["nodes"]["bridge"]:
        bridge_id = bridge["id"]
        edges = sorted(
            bridge_edges_map[bridge_id],
            key=lambda x: x["compliance_degree"],
            reverse=True
        )

        related_laws = []
        for edge in edges:
            law_node = law_node_map[edge["target"]]
            related_laws.append({
                "law": law_node["name"],
                "article": law_node["article_id"],
                "level": law_node["level"],
                "effective_date": law_node["effective_date"],
                "relation": edge["relation_type"],
                "compliance_degree": edge["compliance_degree"]
            })

        simplified_kg["compliance_graph"].append({
            "bridge_id": bridge_id,
            "bridge_sd": bridge["sd"],
            "compliance_attr": bridge["compliance_attr"],
            "related_laws": related_laws
        })

    def sort_key(item):
        max_deg = max([l["compliance_degree"] for l in item["related_laws"]] or [0])
        is_non_compliant = 1 if item["compliance_attr"] == "不合规" else 0
        return (-is_non_compliant, -max_deg)

    simplified_kg["compliance_graph"].sort(key=sort_key)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified_kg, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存简化版知识图谱至：{output_path}")
    print(f"   - 涉及桥接项数量：{len(simplified_kg['compliance_graph'])}")
    total_relations = sum(len(item["related_laws"]) for item in simplified_kg["compliance_graph"])
    print(f"   - 有效法律关联总数：{total_relations} 条")


# ========== 5. 主流程 ==========
if __name__ == "__main__":
    print("=== 开始加载合规分析数据 ===")
    try:
        # 输入为LLM关联分析的结果文件
        with open('COT_mapping_result.json', 'r', encoding='utf-8') as file:
            relation_data = json.load(file)
        print(f"加载完成，共 {len(relation_data)} 个桥接项的关联分析数据\n")
    except FileNotFoundError:
        print("❌ 错误：未找到 relation_data.json 文件，请确保文件存在。")
        exit(1)

    # 构建图谱
    print("=== 开始构建合规关系知识图谱 ===")
    compliance_kg = build_compliance_knowledge_graph(relation_data)
    print(f"图谱构建完成：")
    print(f"   - 桥接项节点：{len(compliance_kg['nodes']['bridge'])} 个")
    print(f"   - 法律条文节点：{len(compliance_kg['nodes']['law'])} 个")
    print(f"   - 有效关联边（已过滤无关联）：{len(compliance_kg['edges'])} 条\n")

    # 保存简化版
    print("=== 正在生成并保存知识图谱RKG ===")
    save_simplified_kg_by_bridge(compliance_kg, "compliance_RKG_CoT.json")

    print("\n🎉 合规知识图谱生成与导出完成！")
    print("文件：compliance_RKG_CoT.json")