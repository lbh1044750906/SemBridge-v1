# SemBridge ---> End-to-end implementation script
import os
import json
from ESE_module.event_enhance import init_enhancer, enhance_event
from BridgeMatching.BridgeMatch_predict import BehaviorPredictor
from BridgeMapping.link_inference import build_bridge_to_law_ids, process_behavior_jsonl

def count_behavior_records(file_path):
    """
    读取行为数据JSON文件，返回其中的行为记录总数。

    Args:
        file_path (str): JSON文件路径

    Returns:
        int: 行为数据的总数量

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: 文件格式不是合法JSON
        ValueError: JSON根节点不是数组
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到：{file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON数据必须是一个数组（列表）")

    return len(data)


if __name__ == "__main__":
    ### ESE模块加载
    
    # 初始化语义增强器
    init_enhancer()
    
    # 加载行为结构化数据json文件  批量处理
    # event_json = "event_audit.json"
    # try:
    #     total_count = count_behavior_records(event_json)
    #     print(f"共加载 {total_count} 条行为数据")
    # except Exception as e:
    #     print(f"错误：{e}")
    
    # 测试安全事件输入
    event =  {
          "app_name": "某能源服务公众号",
          "package_name": "com.energy.service",
          "audit_type": ["安全核查"],
          "check_items": [
            "穿透测试会员注册时对会员信息的收集行为"
          ],
          "findings": [
            {
              "type": "optional_personal_info_collection",
              "item": "是否强制收集非必要个人信息（如姓名、手机号、生日、车牌号）",
              "non_essential_info_required": "false",
              "optional_fields": ["姓名", "手机号", "生日", "车牌号"],
              "info_not_filled_impacts_functionality": "false"
            }
          ],
          "evidence_available": "true"
        }
    
    # 执行事件语义增强
    enhanced_text = enhance_event(event)
    print("enhanced_text:", enhanced_text)
    
    ### 事件增强描述T与桥接项匹配——BridgeMatching
    try:
        # 初始化预测器
        predictor = BehaviorPredictor()
        
        # 预测并保存结果
        result_file = predictor.predict_and_save(enhanced_text)
    
    except Exception as e:
        print(f"执行出错: {str(e)}")
    
    
    ### 桥接项与政策集合映射——BridgeMapping
    RELATION_DATA_PATH = "./BridgeMapping/compliance_RKG_CoT.json"  # 默认是construction_RKG.py输出知识图谱文件
    INPUT_JSONL_PATH = result_file       # 预测的桥接项结果文件
    OUTPUT_JSONL_PATH = "output_link_compliance_result.json"   # link结果输出文件
    
    print("=== 开始加载桥接项到法律条文的映射 ===")
    bridge_to_clauses = build_bridge_to_law_ids(RELATION_DATA_PATH)
    
    print("\n=== 开始处理行为数据 ===")
    process_behavior_jsonl(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH, bridge_to_clauses)


