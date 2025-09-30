#!/usr/bin/env python3
"""
简单测试脚本 - 快速验证模型能否正常工作
生成更短的回复，速度更快
"""

from model import ChatModel
import config

def test_model():
    """测试模型基本功能"""
    print("=" * 50)
    print("快速模型测试")
    print("=" * 50)
    
    # 加载模型
    print("\n加载模型...")
    llm = ChatModel(model_path=config.LLM_MODEL_PATH)
    llm.load_model()
    
    # 测试简单问题（应该很快得到答案）
    print("\n" + "=" * 50)
    print("测试简单问题")
    print("=" * 50)
    
    question = "你好"
    print(f"\n问题: {question}")
    
    # 使用更快的生成参数
    import torch
    with torch.no_grad():
        # 构建消息
        messages = [{"role": "user", "content": question}]
        
        # 应用模板
        text = llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码
        model_inputs = llm.tokenizer([text], return_tensors="pt").to(llm.device)
        
        # 快速生成（贪心解码，只生成50个token）
        print("正在生成（最多50个token）...", end="", flush=True)
        generated_ids = llm.model.generate(
            **model_inputs,
            max_new_tokens=50,  # 只生成50个token，大约30秒-2分钟
            do_sample=False,    # 贪心解码，更快
            pad_token_id=llm.tokenizer.pad_token_id,
            eos_token_id=llm.tokenizer.eos_token_id
        )
        print(" 完成！")
        
        # 解码
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = llm.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n回答: {response}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
