#!/usr/bin/env python3
"""
模型下载脚本
使用 ModelScope 下载所需的 Qwen 模型
"""

from modelscope.hub.snapshot_download import snapshot_download
import os

def download_models():
    """下载所有需要的模型"""
    print("=" * 60)
    print("开始下载 Qwen 模型")
    print("=" * 60)
    
    # 下载 Qwen3-Embedding-4B 嵌入模型
    print("\n1. 下载 Qwen3-Embedding-4B 嵌入模型...")
    print("   模型大小约 8GB，请耐心等待...")
    try:
        embedding_model_dir = snapshot_download('Qwen/Qwen3-Embedding-4B')
        print(f"✓ 嵌入模型下载完成！")
        print(f"  路径：{embedding_model_dir}")
    except Exception as e:
        print(f"✗ 嵌入模型下载失败：{str(e)}")
        return False
    
    # 下载 Qwen2.5-3B-Instruct 对话模型
    print("\n2. 下载 Qwen2.5-3B-Instruct 对话模型...")
    print("   模型大小约 6GB，请耐心等待...")
    try:
        llm_model_dir = snapshot_download('Qwen/Qwen2.5-3B-Instruct')
        print(f"✓ 对话模型下载完成！")
        print(f"  路径：{llm_model_dir}")
    except Exception as e:
        print(f"✗ 对话模型下载失败：{str(e)}")
        return False
    
    print("\n" + "=" * 60)
    print("所有模型下载完成！")
    print("=" * 60)
    
    # 生成配置信息
    print("\n请将以下路径复制到 config.py 中：")
    print("-" * 60)
    print(f"EMBEDDING_MODEL_PATH = \"{embedding_model_dir}\"")
    print(f"LLM_MODEL_PATH = \"{llm_model_dir}\"")
    print("-" * 60)
    
    # 提示下一步操作
    print("\n下一步：")
    print("1. 编辑 config.py，设置上述模型路径")
    print("2. 运行 python get_vector.py 创建向量数据库")
    print("3. 运行 python main.py 启动问答系统")
    
    return True

if __name__ == "__main__":
    try:
        success = download_models()
        if success:
            print("\n✓ 全部完成！")
        else:
            print("\n✗ 下载过程中出现错误，请检查网络连接后重试")
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
    except Exception as e:
        print(f"\n✗ 发生错误：{str(e)}")
