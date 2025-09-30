from modelscope import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import torch
import os
import re

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
        
        # 设置 pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
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
        
        # 构建消息格式
        messages = []
        
        # 添加历史对话
        if use_history and self.history:
            messages.extend(self.history)
        
        # 添加当前问题
        messages.append({"role": "user", "content": prompt})
        
        # 使用 tokenizer 的 apply_chat_template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成回复
        print("正在生成回复...", end="", flush=True)
        
        # 构建生成参数（只包含模型支持的参数）
        gen_kwargs = {
            "max_new_tokens": 128,  # CPU模式下减少生成长度，提高速度
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_beams": 1,  # 不使用beam search，更快
        }
        
        # 根据设备类型添加采样参数
        if self.device == "cuda":
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
            })
        else:
            gen_kwargs["do_sample"] = False  # CPU模式使用贪心解码，更快
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )
        print(" 完成！")
        
        # 解码输出（只获取新生成的部分）
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 清理响应 - 移除<think>标签及其内容
        response = response.strip()
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()
        
        # 更新历史
        if use_history:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []