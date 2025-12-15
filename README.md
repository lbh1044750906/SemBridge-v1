# *SemBridge*: Semantic Bridge for Event-Policy Linking

From Event to Policy: An Automated Semantic Bridging Approach for Security Event-Policy Linking

## 📝 Introduction

We have implemented **_SemBridge_**, an automated and fully auditable method for **Security Event-Policy Linking** and subsequent compliance judgment. Unlike traditional direct matching, *SemBridge* introduces **intermediate "bridge items"** (桥接项) that act as **pivotal anchors** between raw security events and abstract regulatory requirements. This design enables **multi-hop, explainable inference**, providing a complete and verifiable logical path (full audit trail) from the occurrence of a security event to the final regulatory conclusion. The method automatically generates comprehensive compliance reports, which not only identify potential policy violations but also furnish **detailed, trace-backed explanations** to facilitate compliance analysis and remediation.

<!-- 可选：截图、动图、Logo -->

## 🌟 Core Features 

- 🔗 **Auto-link events to laws**: Link security events (e.g., app logs or audit records) to relevant privacy regulation clauses (e.g., PIPL article 14).
- 🔍 **Automated compliance analysis:** Assessing whether security events meet regulatory requirements, such as compliance or violation.
- 📚 **Multi-source regulatory analysis**: Supports joint analysis of regulations across different levels, dimensions, and hierarchies (e.g., PIPL, CSL, DSL, etc.).
- 📑 **Generate link reports**: Export results in JSON format for link inference paths between events and legal clauses.
- ⚡ **Higher efficiency**: Minimizes manual auditing costs, enables rapid output of compliance insights, and supports human-in-the-loop review.
- 🧩 **Operational expandability**: Predefined bridge items and regulatory documents are modular, flexible, and easily updatable to adapt to domain-specific needs and evolving legal requirements.

## ⚙️ Method Workflow
The workflow transforms security event data into auditable compliance decisions through four stages:

- **Event Semantic Enhancement (ESE).** We use engineered prompts, iteratively refined based on key insights, to leverage an LLM to perform contextual expansion, generating comprehensive textual representation $T$ that preserve original event semantics while enhancing interpretability.

- **Bridge Foundation Construction (BFC).** This module establishes intermediate semantic constructs called *bridge items*. These items serve as pivots between event descriptions and policy clauses. Through an expert-guided process, we curate a set of bridge items $\mathcal{B}$ that capture compliance-relevant concepts at a suitable abstraction level for bidirectional mapping.

- **Semantic Bridge Linking (SBL)**. As the core analytical component, this module performs multi-stage semantic alignment via three operations: 1) *BridgeMatching* identifies relevant bridge items $\mathcal{B}_{\text{match}} \subseteq \mathcal{B}$ aligned with description $T$. 2) *BridgeMapping* constructs a Relationship Knowledge Graph ($\mathcal{RKG}$)`RKG` encoding semantic associations between bridge items and policy clauses. 3) *Link Inference* synthesizes these results to generate structured compliance reports linking events to specific policy provisions.

- **Explainable Decision-Making (EDM):** This final module ensures transparency through Chain-of-Thought (CoT) reasoning. It generates interpretable justifications for each inference step and extracts complete decision paths, allowing auditors to trace the logical progression from security events to compliance determinations.

![SemBridge Overview](resources/SemBridge_overview.jpg)

## 📂 Project Structure

```
SemBridge-v1/
├── BridgeMapping/                          # BridgeMapping and link inference
│   ├── __init__.py
│   ├── compliance_RKG_CoT.json
│   ├── construction_RKG.py                 # constructing RKG (Relationship Knowledge Graph)
│   ├── link_inference.py                   # link inference
│   ├── LLM_MapBridgeToRule.py              # LLM association analysis for bridge items and regulations
│   └── policies_all.json					# regulations data
├── BridgeMatching/                         # BridgeMatching
│   ├── __init__.py
│   ├── BridgeMatch_predict.py              # bridge items match prediction
│   ├── BridgeMatch_trainLLM.py             # training LLM for bridge matching
│   ├── config.json                         # Configuration file
│   └── train_bridge_data.json              # Training data
├── ESE_module/                             # ESE module
│   ├── __init__.py
│   ├── config.json                         # Configuration file
│   └── event_enhance.py                    # event semantic enhancement
├── bridge_items.json                       # Predefined Bridge items
├── end_to_end_sembridge.py                 # main .py for the SemBridge process
├── requirements.txt                 		
└── README.md       
```

## 📊 Dataset Description



## 🧑‍💻 Demo
![SemBridge Overview](resources/case_study2.jpg)


## 🚀 Quick Start

### Install

```bash
# pip 安装
pip install -r requirements.txt

# 方式 2：克隆源码
git clone https://github.com/yourname/your-repo.git
cd your-repo
pip install -r requirements.txt
```


## 📋 Evaluation Results



