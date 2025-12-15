# event_enhancer.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union, Dict, Any
from bs4 import BeautifulSoup
import os

def load_config(config_path: str = 'config.json') -> dict:
    """从JSON文件加载配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")
  
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载配置
CONFIG = load_config()

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
    """应用行为事件语义增强器"""
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
        """根据模型名称获取对应的路径"""
        model_paths = CONFIG['MODEL_PATHS']
        if self.model_name not in model_paths:
            raise ValueError(f"不支持的模型名称：{self.model_name}")
        return model_paths[self.model_name]

    def generate_html(self, prompt: str, max_length: int = 4096) -> str:
        """生成HTML格式的输出"""
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
                # 默认开启采样
            )
      
        return self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

    def extract_content(self, html: str) -> str:
        """从HTML中提取纯文本内容"""
        soup = BeautifulSoup(html, 'html.parser')
        enhanced_div = soup.find('div', class_='enhanced')
      
        if enhanced_div:
            return enhanced_div.get_text(separator=' ', strip=True)
      
        return html[-100:].strip() if len(html) > 100 else html

    def enhance(self, event: Union[Dict, str]) -> str:
        """增强事件描述"""
        if isinstance(event, dict):
            event_text = json.dumps(event, ensure_ascii=False, indent=2)
        else:
            event_text = event.strip()
      
        prompt_template = ENHANCE_PROMPT
        html_prompt = prompt_template.format(event_text=event_text)
        html_output = self.generate_html(html_prompt)
        return self.extract_content(html_output)

# 全局增强器实例
_enhancer = None

def init_enhancer():
    """初始化事件增强器"""
    global _enhancer
    if _enhancer is None:
        _enhancer = HTMLSemanticEnhancer()

def enhance_event(event: Union[Dict, str]) -> str:
    """增强事件描述
  
    参数:
        event: 可以是字典格式的事件数据或字符串格式的事件描述
      
    返回:
        增强后的事件描述文本
    """
    global _enhancer
    if _enhancer is None:
        init_enhancer()
    return _enhancer.enhance(event)

