from modelscope import AutoTokenizer, AutoModel
from typing import List, Optional
import torch
import os

# 自定义Qwen ChatModel
class ChatModel:
    def __init__(self, model_path=None):
        """
        初始化对话模型
        :param model_path: 本地模型路径，如果为None则使用默认路径
        """
        # 如果没有指定路径，使用默认的本地路径
        if model_path is None:
            model_path = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen2___5-3B-Instruct"
        self.model_path = model_path
        self.max_token = 4096
        self.temperature = 0.8
        self.top_p = 0.9
        self.tokenizer = None
        self.model = None
        self.history = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """从本地路径加载模型"""
        print(f"正在从本地加载模型: {self.model_path}")
        print(f"使用设备: {self.device}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 从本地路径加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
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