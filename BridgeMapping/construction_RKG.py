import json
import re
from collections import defaultdict
from datetime import datetime

# ========== 1. Core configuration ==========

# 1.1 Legal Hierarchy Weighting
LAW_HIERARCHY_CONFIG = {
    "宪法": {"level": 5, "rule_weight": 1.0},
    "法律": {"level": 4, "rule_weight": 0.8},
    "行政法规": {"level": 3, "rule_weight": 0.6},
    "地方性法规": {"level": 2, "rule_weight": 0.4},
    "其他": {"level": 1, "rule_weight": 0.3}
}

# 1.2 Relationship Type Weighting
RELATION_WEIGHT = {
    "直接对应": 1.0,
    "包含关系": 0.8,
    "补充关系": 0.6,
    "无关联": 0.0
}

# 1.3 Legal Abbreviation Mapping Table
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

# 1.4 Bridging Item Compliance Attributes
BRIDGE_COMPLIANCE_ATTR = {
    "B001": "不合规",
    "B002": "合规"
}



# ========== 2. Core function: Parse and extract legal information from law_id ==========

# Support for a broader range of law_id formats
LAW_ID_PATTERN = re.compile(r"^([A-Za-z0-9]+)-(.+)$")


def extract_law_info_by_id(law_id: str) -> dict:
    """
    Extracted from law_id (e.g. GBT35273-11.5 or Civil Code-1038):
      - Prefix (for lookup purposes)
      - Suffix (used directly as article_id)
    """
    law_id = law_id.strip()
    match = LAW_ID_PATTERN.match(law_id)
    if not match:
        print(f"⚠️ Warning: law_id '{law_id}' cannot be parsed; the format should be 'prefix-suffix'.")
        return {
            "name": "未知法律",
            "article_id": "未知条款",
            "level": "其他",
            "effective_date": "2000-01-01",
            "rule_weight": 0.2
        }

    abbr, article_suffix = match.groups()
    article_id = f"{article_suffix}"  # Use suffixes directly as clause identifiers

    # Look up legal metadata (based on prefix)
    law_meta = LAW_ABBR_MAP.get(abbr)
    if not law_meta:
        print(f"⚠️ Warning: Unrecognised legal abbreviation '{abbr}', using default value.")
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

# ========== 3. Constructing knowledge graphs ==========
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

        # Add bridge item node
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

            # Analyse law_id
            law_info = extract_law_info_by_id(law_id)
            law_name = law_info["name"]
            law_article_id = law_info["article_id"]
            law_node_key = f"{law_name}_{law_article_id}"

            # Add legal node
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

            # Calculate compliance correlation ---> LKI
            compliance_degree = round(relation_weight * law_info["rule_weight"] * 5, 2)

            # Add edge
            if not any(e["source"] == bridge_id and e["target"] == law_node_id for e in kg["edges"]):
                kg["edges"].append({
                    "source": bridge_id,
                    "target": law_node_id,
                    "relation_type": relation_type,
                    "compliance_degree": compliance_degree
                })

    return kg


# ========== 4. Output graph structure ==========
def save_simplified_kg_by_bridge(kg: dict, output_path: str):
    law_node_map = {node["id"]: node for node in kg["nodes"]["law"]}
    bridge_edges_map = defaultdict(list)
    for edge in kg["edges"]:
        bridge_edges_map[edge["source"]].append(edge)

    simplified_kg = {
        "metadata": {
            "description": "合规关系知识图谱",
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

    print(f"\n✅ The knowledge graph has been saved to：{output_path}")
    print(f"  - Number of bridged items involved：{len(simplified_kg['compliance_graph'])}")
    total_relations = sum(len(item["related_laws"]) for item in simplified_kg["compliance_graph"])
    print(f"  - Total number of valid legal connections：{total_relations} clauses")


# ========== 5. Main process ==========
if __name__ == "__main__":
    print("=== Loading LLM association analysis data ===")
    try:
        # Input: Result file for LLM association analysis
        with open('COT_mapping_result.json', 'r', encoding='utf-8') as file:   # COT_mapping_result.json is the input path
            relation_data = json.load(file)
        print(f"Loading complete. Correlation analysis data for {len(relation_data)} bridging items.\n")
    except FileNotFoundError:
        print("❌ Error: relation_data.json file not found. Please ensure the file exists.")
        exit(1)

    # Constructing a map
    print("=== Start building a compliance relationship knowledge graph ===")
    compliance_kg = build_compliance_knowledge_graph(relation_data)
    print(f"The map has been constructed.：")
    print(f"  - Bridge item node：{len(compliance_kg['nodes']['bridge'])} 个")
    print(f"  - Legal Provisions Node：{len(compliance_kg['nodes']['law'])} 个")
    print(f"  - Validly associated edges (irrelevant edges filtered)：{len(compliance_kg['edges'])} 条\n")

    # Save results
    print("=== Generating and saving the Knowledge Graph RKG ===")
    save_simplified_kg_by_bridge(compliance_kg, "compliance_RKG_CoT.json")   # compliance_RKG_CoT.json is the output path

    print("\n🎉 Generation and export of the compliance knowledge graph completed！")
    print("Result file：compliance_RKG_CoT.json")
