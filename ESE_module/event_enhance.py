# event_enhancer.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union, Dict, Any
from bs4 import BeautifulSoup
import os

def load_config(config_path: str = 'config.json') -> dict:
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist")
  
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load configuration
CONFIG = load_config()

# English
ENHANCE_PROMPT =  """[System Prompt]
You are a semantic enhancement tool for behavioral events. Extract factual information and explicitly reveal compliance-related elements. Do not make compliance judgments. Output in the following HTML format:

[Input]
<div class="event-enhancement">
  <div class="original">Original Event: {event_text}</div>

[Output Format]
  <div class="enhanced">
    <!-- Place enhanced description here -->
  </div>
</div>

[Constraints & Rules]
<div class="enhance-requirements">
  <h4>Enhancement Requirements:</h4>
  <ul>
    <li>1. Preserve all original facts. No additions, deletions, or modifications.</li>
    <li>2. Explicitly reveal compliance-related elements if present:
      <ul>
        <li>- Data: Type of data (location, name, ID), mark if sensitive</li>
        <li>- Action: Processing method (collect, transmit, store, share)</li>
        <li>- Technology: Protocol (HTTPS/HTTP), data form (plaintext, encrypted)</li>
        <li>- Entity: Third parties (SDK, server IP), user consent (click to agree, input)</li>
      </ul>
    </li>
    <li>3. Include: Agent (who), Context (when/where), Action (what)</li>
    <li>4. Use concise Chinese. Keep under 100 characters. State only objective facts.</li>
  </ul>
</div>
"""

# Chinese
ENHANCE_PROMPT =  """你是应用行为事件的语义增强工具，仅负责客观提炼行为事实并显性化合规相关信息，不做合规判定。严格按HTML格式输出：
<div class="event-enhancement">
  <div class="original">原始事件信息：{event_text}</div>

  <div class="enhance-requirements">
    <h4>增强要求：</h4>
    <ul>
      <li>1. 绝对保留原始事实，不增删、不修改行为细节</li>
      <li>2. 必须显性化呈现以下“合规相关客观要素”（若事件中存在）：
        <ul>
          <li>- 数据相关：处理的数据类型（如位置信息、姓名、身份证号等，标注是否为敏感类）</li>
          <li>- 行为相关：数据的具体处理方式（如收集、传输、存储、共享给第三方等）</li>
          <li>- 技术相关：传输协议（如HTTPS/HTTP）、数据呈现或保护形式（如明文、加密、脱敏）</li>
          <li>- 主体相关：涉及的第三方（如SDK名称、服务器IP、合作方）、用户授权行为（如点击同意、主动输入）</li>
        </ul>
      </li>
      <li>3. 结构需包含：主体（谁）→ 场景（在什么情况下）→ 行为（做了什么）→ 合规相关要素（上述客观信息）</li>
      <li>4. 语言简洁（≤120字），仅陈述客观事实，不做“是否合规”“是否有风险”的任何判断或结论</li>
      <li>5. 无合规相关要素时，仅完整描述行为即可</li>
      <li>6. 输出的增强描述为一段中文描述，不要有特殊符合和英文，使用中文回答</li>
    </ul>
  </div>

  <div class="enhanced">
    <!-- 此处放置增强后的内容 -->
  </div>
</div>
"""



class HTMLSemanticEnhancer:
    """Event Semantic Enhancer"""
    def __init__(self):
        self.model_name = CONFIG['ENHANCE_MODEL_NAME'].lower()
        self.model_path = self._get_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
      
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
      
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

    def _get_model_path(self) -> str:
        """Retrieve the corresponding path based on the model name"""
        model_paths = CONFIG['MODEL_PATHS']
        if self.model_name not in model_paths:
            raise ValueError(f"Unsupported model names：{self.model_name}")
        return model_paths[self.model_name]

    def generate_html(self, prompt: str, max_length: int = 4096) -> str:
        """Generate output in HTML format"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
            padding=True,

        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                # Sampling enabled by default
            )
      
        return self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

    def extract_content(self, html: str) -> str:
        """Extract plain text content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        enhanced_div = soup.find('div', class_='enhanced')
      
        if enhanced_div:
            return enhanced_div.get_text(separator=' ', strip=True)
      
        return html[-100:].strip() if len(html) > 100 else html

    def enhance(self, event: Union[Dict, str]) -> str:
        """Enhanced Event Description"""
        if isinstance(event, dict):
            event_text = json.dumps(event, ensure_ascii=False, indent=2)
        else:
            event_text = event.strip()
      
        prompt_template = ENHANCE_PROMPT
        html_prompt = prompt_template.format(event_text=event_text)
        html_output = self.generate_html(html_prompt)
        return self.extract_content(html_output)

# Global Enhancer Instance
_enhancer = None

def init_enhancer():
    """Initialise Event Enhancer"""
    global _enhancer
    if _enhancer is None:
        _enhancer = HTMLSemanticEnhancer()

def enhance_event(event: Union[Dict, str]) -> str:
    """Event Semantic Enhancement
    Args:
        event: Event data may be in dictionary format or event descriptions in string format.
    Return:
        Enhanced event description text
    """
    global _enhancer
    if _enhancer is None:
        init_enhancer()
    return _enhancer.enhance(event)

