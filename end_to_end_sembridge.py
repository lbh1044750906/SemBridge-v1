# SemBridge ---> End-to-end implementation script
import os
import json
from ESE_module.event_enhance import init_enhancer, enhance_event
from BridgeMatching.BridgeMatch_predict import BehaviorPredictor
from Link_inference.link_inference import build_bridge_to_law_ids, process_behavior_jsonl

def count_behavior_records(file_path):
    """
    Read the behavioural data JSON file and return the total number of behavioural records contained within.
    Args:
        file_path (str): JSON file path
    Returns:
        int: Total volume of behavioural data
    Raises:
        FileNotFoundError: The file does not exist.
        json.JSONDecodeError: The file format is not valid JSON.
        ValueError: The JSON root node is not an array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found：{file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The JSON data must be an array (list).")

    return len(data)


if __name__ == "__main__":
    # ESE module
    # Initialise the semantic enhancer
    init_enhancer()
    
    # Security event input
    event = {
        "behavior": "location_access",
        "log_lines": [
          "12-17 19:31:41.548  3608  3608 E InCall  : StatusBarNotifier - getNotificationBuilder",
          "12-17 19:31:41.551  1795  2332 I qti_sensors_hal: [ACC]processReportInd: X: 8.770551 Y: 4.652319 Z: 2.789478  x_offset:0.160831 y_offset:0.184366 z_offset:0.985027",
          "12-17 19:31:41.566  1795  3139 D StubTelephonyRegistry: uid:10028 pid:16628 PhoneStateListener.LISTEN_CELL_LOCATION:16 events:16",
          "12-17 19:31:41.566 16628 16628 I UnknownApp: Get precise location (lat:30.XXXX, lng:120.XXXX) without user consent",
          "12-17 19:31:41.566  1795  3139 D StubTelephonyRegistry: LISTEN_CELL_LOCATION uid:10028",
          "12-17 19:31:41.567 16628 16628 I UnknownApp: Send unencrypted location data to external.server.com"
        ]
      }
    
    # Batch processing of JSON file data for security events (enableable)
    # event_json = "event_audit.json"  # events input file
    # try:
    #     total_count = count_behavior_records(event_json)
    #     print(f"A total of {total_count} rows of data have been loaded.")
    # except Exception as e:
    #     print(f"Error：{e}")

    # Implementation of Security Event Semantic Enhancements
    enhanced_text = enhance_event(event)
    print("enhanced_text:", enhanced_text)
    
    # BridgeMatching ---> Matching between bridging items and enhanced description text
    Matching_result_file = ""  # Predicted bridging item results file
    try:
        # Initialise the match predictor
        predictor = BehaviorPredictor()
        
        # Predict and save the results
        Matching_result_file = predictor.predict_and_save(enhanced_text)
    
    except Exception as e:
        print(f"Execution error: {str(e)}")
    
    
    # BridgeMapping ---> Establishing the mapping between bridging items and policy set
    try:
        RELATION_DATA_PATH = "./BridgeMapping/compliance_RKG_CoT.json"  # The knowledge graph result file output by construction_RKG.py
        INPUT_JSONL_PATH = Matching_result_file       # Predicted bridging item results file
        # INPUT_JSONL_PATH = "predict_bridge_data.json"
        OUTPUT_JSONL_PATH = "output_link_compliance_result.json"   # Link result output file

        print("====== Loading the mapping between bridging items and policy clauses ======")
        bridge_to_clauses = build_bridge_to_law_ids(RELATION_DATA_PATH)

        print("\n====== Establishing the link between security event analysis and policy implementation ======")
        process_behavior_jsonl(INPUT_JSONL_PATH, OUTPUT_JSONL_PATH, bridge_to_clauses)

    except Exception as e:
        print(f"Execution error: {str(e)}")
