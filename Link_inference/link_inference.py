import json
import re
from collections import defaultdict
import sys

# ========== 1. Load the knowledge graph and establish the bridge → law_id mapping ==========

def load_law_clauses_from_kg(kg_path: str) -> dict:
    """
    Extracted from the relational knowledge graph:
        bridge_id -> sorted list of law_id (by compliance_degree)
    Retain only relation_type in ["直接对应", "包含关系"]
    Retain only relation_type in ["Direct Alignment", "Partial Alignment"]
        """
    with open(kg_path, 'r', encoding='utf-8') as f:
        kg = json.load(f)

    bridge_to_clauses = {}

    for item in kg["compliance_graph"]:
        bridge_id = item["bridge_id"]
        laws = item["related_laws"]

        # Filtering
        filtered = [
            {
                "law_id": f"{extract_abbr_from_law_name(law['law'])}-{law['article'].replace('第', '').replace('条', '')}",
                "compliance_degree": law["compliance_degree"]
            }
            for law in laws
            if law["relation"] in ["直接对应", "包含关系"]
        ]

        # Sorted in descending order by compliance relevance (i.e., LKI)
        sorted_clauses = sorted(filtered, key=lambda x: x["compliance_degree"], reverse=True)

        # Extract the list of law_id strings
        clause_ids = [c["law_id"] for c in sorted_clauses]
        bridge_to_clauses[bridge_id] = clause_ids

    print(f"✅ The mapping of legal provisions to {len(bridge_to_clauses)} bridging items has been loaded from the knowledge graph.")
    return bridge_to_clauses



def build_bridge_to_law_ids(kg_path: str) -> dict:
    """
    Built from compliance_RKG_CoT.json:
        bridge_id → List of law_id sorted by compliance_degree
    Retain only relation_type in ["Direct Alignment", "Partial Alignment"]
    Retain only relation_type in ["直接对应", "包含关系"]

    As the original law_id (e.g., PIPL-14) is absent from the graph, it is necessary to:
        law_name + article → law_id
    """

    # ===== 1. Reverse mapping table from legal names to abbreviations =====
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
        # Further additions may be made
    }

    # ===== 2. Load the relational knowledge graph =====
    with open(kg_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bridge_to_clauses = {}

    # ===== 3. Iterate through compliance_graph =====
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

            # Look up the corresponding abbreviation
            abbr = LAW_NAME_TO_ABBR.get(law_name)
            if not abbr:
                print(f"⚠️ Unrecognised legal name: {law_name}, skipped")
                continue

            # Construct the original law_id format: Abbreviation-Clause number
            # Note: article_id may include “第X条”, requiring removal.
            cleaned_article = re.sub(r"第|条|款|项", "", article_id).strip()
            if not cleaned_article:
                cleaned_article = "通用"

            law_id = f"{abbr}-{cleaned_article}"

            # Sort by compliance_degree
            compliance_degree = law["compliance_degree"]

            clause_list.append({
                "law_id": law_id,
                "compliance_degree": compliance_degree
            })

        # Sort in descending order by compliance relevance (i.e., LKI)
        sorted_clauses = sorted(clause_list, key=lambda x: x["compliance_degree"], reverse=True)
        # Extract the list of law_id
        bridge_to_clauses[bridge_id] = [c["law_id"] for c in sorted_clauses]

    print(f"✅ A bridge → law_id mapping has been constructed from {kg_path}, comprising {len(bridge_to_clauses)} bridge entries.")
    return bridge_to_clauses


# ========== 2. Processing input JSONL/JSON files ==========
def process_behavior_jsonl(input_path: str, output_path: str, bridge_to_clauses: dict):
    """
    Read input JSONL, output JSONL with related_clauses
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
                print(f"❌ Unable to parse line: {line}, error: {e}")
                continue

            behavior = record.get("behavior", "")
            compliance_list = record.get("compliance_labels", [])

            # Handling empty compliance_labels
            if not compliance_list:
                output_data.append({
                    "behavior": behavior,
                    "compliance_labels": 1,
                    "related_clauses": []
                })
                continue

            # Extract the SJ values and take the minimum value as the final compliance_labels.
            try:
                sj_values = [item.get("SJ", 1) for item in compliance_list]
                final_compliance_label = min(sj_values)  # Whenever there is an SJ=0, it is 0.
            except (KeyError, TypeError) as e:
                print(f"⚠️ Failed to process SJ value: {e}, behavior: {behavior[:50]}, data: {compliance_list}")
                final_compliance_label = 1

            # Collect all law_id corresponding to bridge_id
            all_clauses = []
            for item in compliance_list:
                try:
                    bridge_id = item.get("bridge_id")
                    if bridge_id:
                        clauses = bridge_to_clauses.get(bridge_id, [])
                        all_clauses.extend(clauses)
                except AttributeError:
                    continue

            # Remove duplicates, maintain sorting (in order of first occurrence)
            seen = set()
            unique_clauses = []
            for cid in all_clauses:
                if cid and cid not in seen:  # Ensure that the CID is not empty and does not contain duplicates.
                    seen.add(cid)
                    unique_clauses.append(cid)

            output_data.append({
                "behavior": behavior,
                "compliance_labels": final_compliance_label,
                "related_clauses": unique_clauses
            })

    # Write to the output file
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in output_data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Processing completed. Output file:{output_path}")
    print(f"   A total of {len(output_data)} records")
