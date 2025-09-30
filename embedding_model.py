from modelscope import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List
import os

class EmbeddingModel:
    def __init__(self, model_path=None):
        """
        初始化嵌入模型
        :param model_path: 本地模型路径，如果为None则使用默认路径
        """
        # 如果没有指定路径，使用默认的本地路径
        if model_path is None:
            model_path = "/mnt/workspace/.cache/modelscope/models/Qwen/Qwen3-Embedding-4B"
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """从本地路径加载嵌入模型"""
        print(f"正在从本地加载嵌入模型: {self.model_path}")
        print(f"使用设备: {self.device}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
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
        print("嵌入模型加载完成！")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为向量
        :param texts: 文本列表
        :return: 向量数组
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用load_model()方法")
        
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为句子嵌入
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        将单个文本转换为向量
        :param text: 单个文本
        :return: 向量
        """
        return self.encode([text])[0]
