from modelscope import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B"):
        """
        初始化嵌入模型
        :param model_name: ModelScope上的模型名称
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """从ModelScope加载嵌入模型"""
        print(f"正在从ModelScope加载嵌入模型: {self.model_name}")
        print(f"使用设备: {self.device}")
        
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
