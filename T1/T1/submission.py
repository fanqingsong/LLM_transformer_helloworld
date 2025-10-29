# -*- coding: utf-8 -*-
"""
仅此文件允许修改：
- 请在下列函数的函数体内完成实现。
- 不要改动函数名与参数签名。
- 你可以新增少量辅助函数。
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 第一部分：Prompt 定义
# ============================================================
def build_system_prompt() -> str:
    """
    实现：定义 system prompt
    - 返回一个 system prompt，要求模型以"[Answer]: xxxx"的格式给出最终数值。
    
    System Prompt 的作用：
    1. 定义模型的身份和角色（数学助手）
    2. 指导模型的推理过程（逐步分析、解题思路）
    3. 规范模型的输出格式（[Answer]: 数值）
    
    为什么要用 [Answer]: 格式？
    - 便于评测程序从模型输出中提取正确答案
    - 统一答案格式，避免模型输出过多解释性文字
    - 符合测试数据的标准格式要求
    """
    # ======== 实现区域（可修改） ========
    
    # System Prompt 定义：指导模型如何回答数学问题
    # 关键点：必须包含 [Answer]: 格式的要求，这是评测程序提取答案的标准
    system_prompt = """你是一个数学助手，擅长解决各种数学问题。请仔细分析问题，逐步推理，并最终以 [Answer]: 数值 的格式给出答案。

要求：
1. 仔细阅读题目，理解题意
2. 逐步分析解题思路
3. 进行必要的计算
4. 最终答案必须以 [Answer]: 数值 的格式给出

请开始解题。"""
    
    return system_prompt
    
    # ======== 实现区域（可修改） ========


# ============================================================
# 第二部分：模板拼装
# ============================================================
def apply_chat_template_single(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    problem: str,
) -> str:
    """
    实现：将单个问题转换为模型输入文本
    - 使用 tokenizer.apply_chat_template 构造对话
    - 返回拼装好的文本字符串
    
    参数说明：
        tokenizer: Qwen3 模型的 tokenizer，包含特殊 token 和模板定义
        system_prompt: 系统提示词，定义模型角色和输出格式
        problem: 用户问题（如 "12 + 35"）
    
    返回值：
        完整格式化后的对话文本，包含 system、user、assistant 标记
    
    Chat Template 的作用：
        transformers 库的 tokenizer.apply_chat_template 会根据模型类型自动构造
        符合该模型预期的对话格式。对于 Qwen 模型，它会把 messages 列表转换为：
        <|system|>你是一个数学助手...<|user|>12 + 35<|assistant|>
        这样的格式。
    """
    # ======== 实现区域（可修改） ========
    
    # 构造消息列表：包含系统提示和用户问题
    # 消息格式：每个消息是一个字典，包含 role（角色）和 content（内容）
    # - "system": 定义模型行为
    # - "user": 用户输入的问题
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    
    # 应用聊天模板，将消息列表转换为模型可理解的文本格式
    rendered = tokenizer.apply_chat_template(
        messages,                      # 消息列表
        tokenize=False,                # 不进行 tokenize，返回原始字符串而不是 token IDs
                                      # 这样可以在调试时看到完整的文本内容
        add_generation_prompt=True,    # 添加 <|assistant|> 标记，引导模型生成回复
                                      # 如果没有这个标记，模型不知道要在哪里开始生成
        enable_thinking=True,          # 启用思考标签支持
                                      # 允许模型输出 <think>...</think>
                                      # 模型会在 <> 标签内进行推理，评测程序会自动移除这部分内容
    )
    return rendered
    
    # ======== 实现区域（可修改） ========


# ============================================================
# 第三部分：核心推理实现
# ============================================================
def generate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_text: str,
    max_new_tokens: int,
    do_sample: bool,
) -> torch.Tensor:
    """
    实现：单条推理
    - 将文本 tokenize 后送入模型生成
    - 返回包含输入和输出的完整 token 序列
    
    参数说明：
        model: Qwen3 预训练模型，用于文本生成
        tokenizer: 用于将文本转换为 token IDs
        rendered_text: 已格式化的完整对话文本（包含 system、user、assistant 标记）
        max_new_tokens: 生成的最大新 token 数量（不包括输入长度）
        do_sample: 是否使用采样（True=随机采样，False=贪心解码）
    
    返回值：
        torch.Tensor，形状为 [1, seq_len]，包含输入+新生成的完整序列
    
    注意事项：
        1. 返回的是包含输入的完整序列（evaluate.py 会提取新生成的部分）
        2. 使用 .to(model.device) 确保输入在正确的设备上（CPU/GPU/NPU）
        3. 使用 model.generate() 的默认参数，模型会自动处理特殊 token
    """
    # ======== 实现区域（可修改） ========
    
    # 步骤 1：将文本编码为 token IDs
    # - tokenizer.encode() 将字符串转换为整数列表
    # - return_tensors="pt" 返回 PyTorch 张量
    # - .to(model.device) 将张量移动到模型所在的设备（CPU/GPU/NPU）
    inputs = tokenizer.encode(rendered_text, return_tensors="pt").to(model.device)
    
    # 步骤 2：使用模型生成文本
    # - model.generate() 会自动生成新的 token
    # - 返回的 outputs 包含完整的序列：输入 + 新生成的 token
    # - 评测程序会从 outputs 中提取新生成的部分
    outputs = model.generate(
        inputs,                        # 输入 token IDs
        max_new_tokens=max_new_tokens, # 最多生成多少个新 token
        do_sample=do_sample,          # 采样模式：True=随机采样（更灵活），False=贪心（更确定）
    )
    return outputs
    # ======== 实现区域（可修改） ========


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_texts: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[torch.Tensor]:
    """
    实现：批量推理
    - 一次处理多个问题，提高效率
    - 返回所有批次的输出列表
    
    参数说明：
        model: Qwen3 预训练模型
        tokenizer: 用于编码文本
        rendered_texts: 多个已格式化的对话文本列表
        max_new_tokens: 生成的最大新 token 数量
        do_sample: 是否使用采样
    
    返回值：
        List[torch.Tensor]，每个元素是一个批次的输出
        批次形状为 [batch_size, seq_len]
    
    批量处理的优势：
        1. 并行处理：GPU 可以同时处理多条数据
        2. 减少数据传输：一次 GPU 调用处理多条
        3. 提高吞吐量：相比逐条处理能获得 3+ 倍加速
    
    实现要点：
        1. 使用循环分批处理，避免一次性处理过多数据导致内存溢出
        2. 使用 padding=True 确保同一批次内序列长度一致
        3. 使用 attention_mask 告诉模型哪些 token 是 padding，哪些是真实内容
        4. 使用 torch.no_grad() 节省显存（推理时不需要梯度）
    """
    # ======== 实现区域（可修改） ========
    
    all_outputs = []  # 存储所有批次的输出
    batch_size: int = 4  # 批次大小：一次处理 4 条数据
                        # 可以调整：太小效率低，太大可能内存不足

    # 循环处理：每次处理一个批次
    # range(0, len(rendered_texts), batch_size) 生成：0, 4, 8, 12, ...
    for i in range(0, len(rendered_texts), batch_size):
        # 提取当前批次：rendered_texts[i:i+batch_size]
        # 例如：如果 i=0, batch_size=4，则提取第 0, 1, 2, 3 条
        batch_texts = rendered_texts[i:i + batch_size]
        
        # 步骤 1：对批次进行编码和填充
        # tokenizer() 可以将多条文本同时编码，并自动填充到相同长度
        # - batch_texts: 多条文本（列表）
        # - padding=True: 短序列会填充到和最长序列一样长
        # - truncation=True: 超长序列会被截断
        # - 返回的 inputs 包含 input_ids 和 attention_mask
        inputs = tokenizer(
            batch_texts,               # 多条文本（如 ["问题1", "问题2", "问题3", "问题4"]）
            return_tensors="pt",        # 返回 PyTorch 张量
            padding=True,              # 自动填充到相同长度
            truncation=True            # 超长序列截断（防止溢出）
        ).to(model.device)             # 移动到模型设备
        
        # 步骤 2：批量生成文本
        # 使用 torch.no_grad() 禁用梯度计算，节省显存并提高速度
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,        # 编码后的 token IDs，形状 [batch_size, seq_len]
                attention_mask=inputs.attention_mask,  # 注意力掩码，标记哪些是真实内容（1），哪些是填充（0）
                max_new_tokens=max_new_tokens,  # 最多生成多少个新 token
                do_sample=do_sample,     # 采样模式
                pad_token_id=tokenizer.eos_token_id  # 指定填充 token 的 ID
                # 注意：这里没有使用 pad_token_id=tokenizer.pad_token_id
                # 因为 Qwen 模型可能没有 pad_token，使用 eos_token_id 作为替代
            )
        
        # 将当前批次的输出添加到总结果中
        all_outputs.append(outputs)

    # 返回所有批次的输出列表
    # 例如：如果 10 条数据分成 3 批，则返回 [batch1_output, batch2_output, batch3_output]
    # evaluate.py 会将这些批次的结果合并到一起
    return all_outputs
    
    # ======== 实现区域（可修改） ========

