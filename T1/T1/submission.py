# -*- coding: utf-8 -*-
"""
仅此文件允许考生修改：
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
    考生实现：定义 system prompt
    - 返回一个 system prompt，要求模型以"[Answer]: xxxx"的格式给出最终数值。
    """
    # ======== 考生实现区域（可修改） ========
    
    system_prompt = """你是一个数学助手，擅长解决各种数学问题。请仔细分析问题，逐步推理，并最终以 [Answer]: 数值 的格式给出答案。

要求：
1. 仔细阅读题目，理解题意
2. 逐步分析解题思路
3. 进行必要的计算
4. 最终答案必须以 [Answer]: 数值 的格式给出

请开始解题。"""
    
    return system_prompt
    
    # ======== 考生实现区域（可修改） ========


# ============================================================
# 第二部分：模板拼装
# ============================================================
def apply_chat_template_single(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    problem: str,
) -> str:
    """
    考生实现：将单个问题转换为模型输入文本
    - 使用 tokenizer.apply_chat_template 构造对话
    - 返回拼装好的文本字符串
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里实现对话模板的构造
    # pass
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,                # 不进行 tokenize，返回字符串形式
        add_generation_prompt=True,    # 添加 assistant 回复标记，引导模型生成答案
        enable_thinking=True,          # 支持 <think></think> 思考标签用法
    )
    return rendered
    
    # ======== 考生实现区域（可修改） ========


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
    考生实现：单条推理
    - 将文本 tokenize 后送入模型生成
    - 返回包含输入和输出的完整 token 序列
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里实现单条推理
    # pass

    inputs = tokenizer.encode(rendered_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    return outputs
    # ======== 考生实现区域（可修改） ========


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_texts: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[torch.Tensor]:
    """
    考生实现：批量推理
    - 一次处理多个问题，提高效率
    - 返回所有批次的输出列表
    """
    # ======== 考生实现区域（可修改） ========
    
    all_outputs = []
    batch_size: int = 4  # 增加批次大小以提高效率

    # 分批处理
    for i in range(0, len(rendered_texts), batch_size):
        batch_texts = rendered_texts[i:i + batch_size]
        
        # 对批次进行 tokenize 和 padding
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        all_outputs.append(outputs)

    return all_outputs
    
    # ======== 考生实现区域（可修改） ========

