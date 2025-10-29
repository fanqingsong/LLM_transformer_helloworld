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
        text1: 第一个文本字符串（如 "12 + 35"）
        text2: 第二个文本字符串（如 "35 + 12"）
        model: 预加载的 Qwen3-Embedding 模型（评测程序提供）
        tokenizer: 预加载的 tokenizer（评测程序提供）
    
    返回：
        相似度值（0.0 到 1.0 之间的浮点数）
        1.0 表示完全相同，0.0 表示完全不同
    
    要求：
        - 使用传入的 model 和 tokenizer，不要自己加载模型
        - 实现 last-token pooling（取最后一个 token 的隐藏状态作为句子表示）
        - 必须使用左侧 padding（padding_side="left"）
        - L2 归一化（将向量标准化为单位向量）
        - 计算余弦相似度（归一化向量的点积）
        - 不得使用 sentence_transformers
    
    相似度计算原理：
        1. 将文本编码为向量（embedding）
        2. Last-token pooling：取最后一个 token 的隐藏状态作为句子表示
        3. L2 归一化：将向量标准化为单位向量
        4. 余弦相似度：计算两个归一化向量的点积（cos(θ) 的值域为 [-1, 1]）
        
        为什么使用 last-token pooling？
        - Embedding 模型在训练时，最后一个 token 包含了整个句子的语义信息
        - 相比平均 pooling 或 CLS token，last-token 能更好地捕捉句子级语义
    """
    # ======== 考生实现区域（可修改） ========
    
    # 步骤 1：获取模型所在的设备（CPU/GPU/NPU）
    # next(model.parameters()) 获取模型的第一个参数，.device 获取其设备
    # 这样可以确保输入张量与模型在同一设备上
    device = next(model.parameters()).device
    
    # 步骤 2：准备输入文本列表
    # 将两个文本放入列表，以便批量处理
    texts = [text1, text2]
    
    # 步骤 3：使用 tokenizer 编码文本
    # padding=True: 将短序列填充到最长序列的长度
    # truncation=True: 超长序列截断到 max_length
    # max_length=512: 最大序列长度（典型的 BERT 等模型长度）
    # return_tensors="pt": 返回 PyTorch 张量
    # padding_side="left": 在左侧填充（重要！）
    #   为什么使用左侧 padding？
    #   - 因为使用 last-token pooling，我们需要保证最后一个 token 是真实的文本 token
    #   - 如果在右侧 padding，最后一个 token 会是填充 token，失去语义信息
    inputs = tokenizer(
        texts,                      # 文本列表：["12 + 35", "35 + 12"]
        padding=True,              # 自动填充到相同长度
        truncation=True,            # 超长序列截断
        max_length=512,            # 最大长度
        return_tensors="pt",        # 返回 PyTorch 张量
        padding_side="left"         # 关键：左侧填充
    ).to(device)                    # 移动到模型设备
    
    # 步骤 4：获取模型输出的嵌入向量
    # 使用 torch.no_grad() 禁用梯度计算，节省显存并提高速度
    with torch.no_grad():
        # 模型前向传播：输入 token IDs，输出隐藏状态
        # outputs.last_hidden_state 形状: [batch_size=2, seq_len, hidden_dim]
        outputs = model(**inputs)
        
        # Last-token pooling：取最后一个 token 的隐藏状态作为句子表示
        # last_hidden_state 形状: [2, seq_len, hidden_dim]
        # 例如：[[[token1], [token2], [token3]], [[token4], [token5], [token6]]]
        # [:, -1, :] 表示取每一行的最后一个 token
        # embeddings 形状: [2, hidden_dim]
        # 例如：[[token3], [token6]]
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_hidden_state[:, -1, :]  # 索引 -1 表示最后一个元素
        
        # 步骤 5：L2 归一化
        # 将向量标准化为单位向量（长度为 1）
        # 公式：v_normalized = v / ||v||
        # F.normalize(..., p=2, dim=1):
        #   - p=2: 使用 L2 范数（欧氏距离）
        #   - dim=1: 在 hidden_dim 维度上归一化
        # 归一化后的向量满足：||v|| = 1
        # embeddings 形状保持不变: [2, hidden_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 步骤 6：计算余弦相似度
        # 对于归一化的向量 a, b，余弦相似度 = a · b（点积）
        # 如果 a 和 b 都是单位向量，则 cos(θ) = a · b
        # - 如果 a 和 b 完全相同：a · b = 1
        # - 如果 a 和 b 垂直：a · b = 0
        # - 如果 a 和 b 相反：a · b = -1
        # torch.dot(a, b) 计算向量点积
        # .item() 将标量张量转换为 Python float
        # embeddings[0] 是 text1 的嵌入，embeddings[1] 是 text2 的嵌入
        similarity = torch.dot(embeddings[0], embeddings[1]).item()
        # similarity 的范围：[0, 1]，因为所有向量都是正方向（经过归一化后的嵌入向量）
    
    return similarity
    
    # ======== 考生实现区域（可修改） ========


def compute_similarity(text1: str, text2: str, model: AutoModel, tokenizer: AutoTokenizer) -> float:
    """
    评测程序调用的接口函数
    """
    return student_compute_similarity(text1, text2, model, tokenizer)
