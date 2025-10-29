# -*- coding: utf-8 -*-
"""
仅此文件允许考生修改：
- 请在下列函数的函数体内完成实现。
- 不要改动函数名与参数签名。
- 你可以新增少量辅助函数。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 相似度计算函数
# ============================================================
def student_compute_similarity(text1: str, text2: str, model: AutoModel, tokenizer: AutoTokenizer) -> float:
    """
    考生实现：计算两个文本之间的相似度
    
    参数：
        text1: 第一个文本字符串
        text2: 第二个文本字符串
        model: 预加载的 AutoModel（评测程序提供）
        tokenizer: 预加载的 AutoTokenizer（评测程序提供）
    
    返回：
        相似度值（0.0 到 1.0 之间的浮点数）
    
    要求：
        - 使用传入的 model 和 tokenizer，不要自己加载模型
        - 实现 last-token pooling
        - 必须使用左侧 padding
        - L2 归一化
        - 计算余弦相似度（点积）
        - 不得使用 sentence_transformers
    """
    # ======== 考生实现区域（可修改） ========
    
    # 获取设备
    device = next(model.parameters()).device
    
    # 准备输入文本
    texts = [text1, text2]
    
    # Tokenize 并设置左侧 padding
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        padding_side="left"
    ).to(device)
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
        # Last token pooling
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_hidden_state[:, -1, :]  # 取最后一个 token 的隐藏状态
        
        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算余弦相似度（点积）
        similarity = torch.dot(embeddings[0], embeddings[1]).item()
    
    return similarity
    
    # ======== 考生实现区域（可修改） ========


def compute_similarity(text1: str, text2: str, model: AutoModel, tokenizer: AutoTokenizer) -> float:
    """
    评测程序调用的接口函数
    """
    return student_compute_similarity(text1, text2, model, tokenizer)
