# _SemBridge_-v1
From Event to Policy: An Automated Semantic Bridging Approach for Security Event-Policy Linking

## Introduction
We have implemented an automated method for linking security events with regulatory requirements and conducting compliance judgement analysis. _**SemBridge**_ introduces intermediate "bridge items" that serve as semantic pivots between events and regulatory requirements, enabling multi-hop inference with full audit trails. The method automatically generates compliance reports that not only identify violations but also provide detailed explanations tracing the logical path from security events to regulatory conclusions.



<!-- 可选：截图、动图、Logo -->

---

## 🌟 特性

- ✅ **功能 1**：简短描述
- ⚡ **功能 2**：高性能 / 易用 / 跨平台等
- 🔒 **功能 3**：安全 / 合规 / 隐私保护等
- 🧩 **功能 4**：支持插件 / 扩展 / API 等

---

## 🚀 快速开始

### Project Structure
```
SemBridgev1/
├── BridgeMapping/                          # BridgeMapping and link inference
│   ├── __init__.py
│   ├── compliance_RKG_CoT.json
│   ├── construction_RKG.py                 # constructing RKG (Rule Knowledge Graph)
│   ├── link_inference.py                   # link inference
│   ├── LLM_MapBridgeToRule.py              # LLM mapping bridge items to regulations
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
└── README.md    
```

### 安装

```bash
# 方式 1：pip 安装（如果是 Python 包）
pip install your-package-name

# 方式 2：克隆源码
git clone https://github.com/yourname/your-repo.git
cd your-repo
pip install -r requirements.txt
