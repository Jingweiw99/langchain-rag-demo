from modelscope import AutoTokenizer, AutoModel
from typing import List, Optional
import torch

# 自定义Qwen ChatModel
class ChatModel:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        """
        初始化对话模型
        :param model_name: ModelScope上的模型名称
        """
        self.model_name = model_name
        self.max_token = 4096
        self.temperature = 0.8
        self.top_p = 0.9
        self.tokenizer = None
        self.model = None
        self.history = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """从ModelScope加载模型"""
        print(f"正在从ModelScope加载模型: {self.model_name}")
        print(f"使用设备: {self.device}")
        
        # 从ModelScope加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cpu":
            self.model = self.model.float()
        
        self.model.eval()
        print("模型加载完成！")

    def chat(self, prompt: str, use_history: bool = True) -> str:
        """
        模型对话
        :param prompt: 输入提示
        :param use_history: 是否使用历史对话
        :return: 模型回复
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
        
        history = self.history if use_history else []
        
        # Qwen2.5使用chat方法
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=history,
            max_length=self.max_token,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        if use_history:
            self.history = history
        
        return response
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []