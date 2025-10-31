"""
评测程序 - 支持两种运行模式：

1. 演示模式（demo）：
   - 详细展示每个题目的推理过程和结果
   - 显示错误样例的详细信息
   - 适合调试和查看模型表现
   使用方法：
     python evaluate.py
     python evaluate.py --mode demo

2. 评分模式（grading）：
   - 简洁的进度输出
   - 不展示错误详情
   - 最终输出详细评分
   使用方法：
     python evaluate.py --mode grading

评分规则：
  - 阶段一-正确性（逐条推理）：8题，每题10分，答对≥5题给满分50分
  - 阶段一-时间性能：单条平均耗时≤5s满分，得分=min(10, 50/t)，满分10分
  - 阶段二-正确性（批量推理）：32题，每题1分，答对≥30题给满分30分
  - 阶段二-时间性能：加速比测试，3倍加速满分10分，1倍0分，线性给分
    * Grading 模式会先用单条推理测试相同的 32 条数据作为基准
    * 加速比 = 单条推理总耗时 / 批量推理总耗时
  - 总分：100分
"""
import os
import sys
import time
import re
import json
import argparse
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from submission import build_system_prompt, apply_chat_template_single, generate_single, generate_batch


# =========================
# 配置
# =========================
MODEL_NAME = os.environ.get("MODEL_NAME", "../Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = 512

DO_SAMPLE = True


# ============================================================
# 统一后处理（考题自带）
# ============================================================
_think_pat = re.compile(r"<think>.*?</think>", re.DOTALL)

def remove_thinking(text: str) -> str:
    """移除文本中的 <think></think> 标签及其内容"""
    return _think_pat.sub("", text).strip()

def remove_answer_prefix(text: str) -> str:
    """移除文本中的 [Answer]: 前缀"""
    text = text.strip()
    # 匹配 [Answer]: 或 [Answer] : 等变体
    pattern = r"^\s*\[?\s*Answer\s*\]?\s*:?\s*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================
# 评测工具
# ============================================================
def _normalize_num(s: str) -> str:
    def to_half(c):
        code = ord(c)
        if 0xFF10 <= code <= 0xFF19:
            return chr(code - 0xFF10 + ord('0'))
        if code == 0xFF0E:
            return '.'
        if code == 0xFF0C:
            return ','
        return c
    s = "".join(to_half(c) for c in s).strip()
    s = s.replace(",", "")
    s = s.rstrip("。．.，,；;、 ")
    return s.strip()


def check_format_and_value(output: str, correct_value: str):
    output = output.strip()
    
    # 检查是否为空
    if not output:
        return False, None, "❌ 输出为空"
    
    # 直接将输出当作答案处理（新格式）
    pred_raw = output
    pred = _normalize_num(pred_raw)
    gold = _normalize_num(correct_value)

    def to_num(x):
        try:
            if "." in x:
                return float(x)
            return int(x)
        except:
            try:
                return float(x)
            except:
                return None

    p, g = to_num(pred), to_num(gold)
    if p is not None and g is not None:
        equal = (p == g) if (isinstance(p, int) and isinstance(g, int)) else (abs(float(p) - float(g)) < 1e-9)
    else:
        equal = (pred == gold)

    if equal:
        return True, pred, "✅ 答案正确"
    else:
        return False, pred, f"❌ 答案错误，应为 {gold}（模型输出: {pred_raw}）"


def score_report(stage_name: str, n_total: int, n_correct: int, elapsed: float, mode: str = "demo"):
    acc = n_correct / n_total if n_total else 0.0
    if mode == "demo":
        print(f"—— {stage_name} 评分 ——")
        print(f"正确题数/总题数：{n_correct}/{n_total}")
        print(f"准确率：{acc*100:.2f}%")
        print(f"总耗时：{elapsed:.3f} s\n")
    return acc

# ============================================================
# 推理封装（调用 submission 实现 + decode）
# ============================================================
def solve_single(
    model,
    tokenizer,
    problem: str,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    """
    单条推理完整流程：
    1. 拼装模板 (调用 submission 实现)
    2. 推理生成 (调用 submission 实现)
    3. 解码输出 (评测程序)
    """
    # 1. 拼装模板 (调用 submission 实现)
    system_prompt = build_system_prompt()
    rendered_text = apply_chat_template_single(tokenizer, system_prompt, problem)
    
    # 2. 推理生成 (调用 submission 实现)
    outputs = generate_single(model, tokenizer, rendered_text, max_new_tokens, do_sample)
    
    # 3. 解码新生成的 token (评测程序)
    # 假设 rendered_text = "<|system|>你是助教<|user|>1+1是多少？<|assistant|>"
    # 模型会接着 rendered_text 继续生成 token 序列，如 total_output = [token_ids for "<|system|>你是助教<|user|>1+1是多少？<|assistant|>2"]
    # 但只需要从 "<|assistant|>" 之后模型新生成的内容，这部分才是真正的答案

    # 先将输入的 rendered_text 转为 token_ids，得到输入长度 inp_len
    inputs = tokenizer(rendered_text, return_tensors="pt", padding=True).to(model.device)
    if "attention_mask" in inputs:
        inp_len = int(inputs["attention_mask"].sum(dim=1).item())
        # attention_mask 标记了内容部分的 token 数（不含 padding）
    else:
        inp_len = inputs["input_ids"].shape[1]
        # 若无 attention_mask，则直接用 token 序列长度

    # 输出 outputs[0] 形如 [输入的 token_ids, 新生成的 token_ids]
    # 只取新生成部分 gen_only = outputs[0][inp_len:]
    # 例如：outputs[0]=[100,101,102,2000]，inp_len=3，则 gen_only=[2000]
    gen_only = outputs[0][inp_len:]

    # 将新生成的 token 解码为字符串，即为最终答案。例如：[2000]->"2"
    text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    
    return text


def solve_batch(
    model,
    tokenizer,
    problems: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[str]:
    """
    批量推理完整流程：
    1. 拼装模板 (调用 submission 实现)
    2. 推理生成 (调用 submission 实现)
    3. 批量解码 (评测程序)
    """
    # 1. 批量拼装模板 (调用 submission 实现)
    system_prompt = build_system_prompt()
    rendered_texts = [
        apply_chat_template_single(tokenizer, system_prompt, p) 
        for p in problems
    ]
    
    # 边界情况：空列表
    if len(rendered_texts) == 0:
        return []
    
    # 2. 批量推理生成 (调用 submission 实现)
    # 例子说明 all_outputs 的数据格式：
    # 假设 problems = 8 道题，batch_size = 4，则 generate_batch 返回 all_outputs 结构如下：
    # all_outputs = [outputs_batch1, outputs_batch2]
    # 其中 outputs_batch1.shape = [4, seq_len1]，outputs_batch2.shape = [4, seq_len2]
    # 每个 outputs_batch 代表一批问题的生成输出，形状是 [本批题数, 每条序列长度]
    # 如果总题数不是 batch_size 的整数倍，比如 10 道题，会返回 3 批，最后一批只有 2 条 shape=[2, seq_len3]
    # 后续会将多个 outputs 拼到一起，处理不同序列长度时用 padding 补齐
    all_outputs = generate_batch(model, tokenizer, rendered_texts, max_new_tokens, do_sample)
    
    # 边界情况：生成结果为空
    if len(all_outputs) == 0:
        return [""] * len(problems)
    
    # 3. 拼接所有批次的结果（评测程序处理）
    if len(all_outputs) == 1:
        batch_outputs = all_outputs[0]
    else:
        # 找到最大序列长度
        max_len = max(output.shape[1] for output in all_outputs)
        
        # 对每个批次的输出进行填充
        padded_outputs = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        for output in all_outputs:
            if output.shape[1] < max_len:
                # 需要在右侧填充
                padding_size = max_len - output.shape[1]
                padding = torch.full(
                    (output.shape[0], padding_size),
                    pad_token_id,
                    dtype=output.dtype,
                    device=output.device
                )
                # 例子说明此代码逻辑：
                # 假设 output 是 [4, 28] （即当前批次4条数据，每条为28个token）
                # 假设 max_len = 30，需要填充2个token到每一行右侧
                # padding 张量就是 [4, 2]，值为 pad_token_id
                # torch.cat([output, padding], dim=1) 会把每一行的 output 右侧接上2个 padding
                # 变成 [4, 30]，这样各批次能对齐拼接，便于后续统一处理
                padded_output = torch.cat([output, padding], dim=1)
                padded_outputs.append(padded_output)
            else:
                padded_outputs.append(output)
        
        # 这里做的是把多个批次的模型输出（每个批次是 [batch_size, seq_len] 张量）拼在一起，组成完整的批量输出。
        # 场景举例：假如一共8个问题，每次4个问题作为一批推理，则all_outputs有2个元素，每个元素形状类似[4, 30]。
        # 这里用torch.cat在第0维拼起来，效果就是[8, 30]，统一后面解码处理。
        batch_outputs = torch.cat(padded_outputs, dim=0)
    
    # 边界情况：生成结果为空
    if batch_outputs.numel() == 0:
        return [""] * len(problems)
    
    # 4. 批量解码（评测程序）
    # 直接解码完整输出，然后提取 assistant 回复部分
    out_texts: List[str] = []
    for out in batch_outputs:
        full_text = tokenizer.decode(out, skip_special_tokens=True).strip()
        
        # 找到最后一个 "assistant" 标记，提取其后的内容
        # 这样可以跳过 system 和 user 部分
        if "assistant" in full_text:
            # 找到最后一个 assistant 的位置
            last_assistant_pos = full_text.rfind("assistant")
            # 提取 assistant 之后的内容
            text = full_text[last_assistant_pos + len("assistant"):].strip()
        else:
            # 如果没有 assistant 标记，直接使用全文
            text = full_text
        
        out_texts.append(text)
    
    return out_texts


def run_pipeline(mode: str = "demo", model_name: str = None):
    """
    """
    运行评测流程

    参数说明：
        mode: 运行模式，可选 "demo" 或 "grading"
            - "demo"：演示模式。会在每个步骤详细输出推理过程、问题、模型生成的内容以及比分，便于观察模型表现和调试。
            - "grading"：评分模式。只输出最终的各项分数结果，无详细推理和中间输出，更适合自动评测或排行榜上传成绩。

        model_name: 加载的模型路径，不指定则使用默认模型名称。
    """
    """
    if mode not in ["demo", "grading"]:
        print(f"⚠️  警告：未知的运行模式 '{mode}'，使用默认模式 'demo'")
        mode = "demo"
    
    # 确定模型路径
    if model_name is None:
        model_name = MODEL_NAME
    
    if mode == "demo":
        print(f"========== 演示模式 ==========")
        print(f"Loading model: {model_name}\n")
    else:
        print(f"========== 评分模式 ==========")
        print(f"Loading model: {model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,  # 允许加载模型仓库自定义的tokenizer/model类（如Qwen等官方支持的模型）
        padding_side="left",
        truncation_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 预检 system_prompt（来源于 submission 实现）
    sp = build_system_prompt().strip()
    assert len(sp) > 0, "system_prompt 为空，请在 submission.py 中实现 build_system_prompt()"

    # ===== 阶段一：逐条推理 =====
    # 从 data/test_data.jsonl 读取测试数据
    stage1_items: List[Tuple[str, str]] = []
    test_data_path = os.path.join(os.path.dirname(__file__), "data", "test_data_1.jsonl")
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                stage1_items.append((item["problem"], item["answer"]))

    if mode == "demo":
        print("==== 阶段一：逐条推理（chat template） ====")
    else:
        print("==== 阶段一：逐条推理 ====")
    
    stage1_correct = 0
    stage1_start = time.time()
    # 使用 torch.inference_mode():
    # 1. inference_mode() 用于模型推理阶段，确保不会追踪/存储计算图，从而节省显存并提升推理速度。
    # 2. 它比 torch.no_grad() 更进一步：
    #    - torch.no_grad() 只是不保存梯度，适合评估/推理；
    #    - torch.inference_mode() 除了不保存梯度，还会禁用 Autograd 引擎，提高内存效率和推理吞吐量，且对输入的 requires_grad 状态有额外优化，几乎禁止任何 Autograd 操作。
    #    - 在多数情况下，推理任务推荐用 inference_mode()，尤其是只做前向、绝不需要反向传播的时候。
    with torch.inference_mode():
        # enumerate(stage1_items, 1) 会生成一个带计数的迭代器：
        # - idx 表示序号（从 1 开始）
        # - (problem, gold) 表示每条题目和标准答案
        for idx, (problem, gold) in enumerate(stage1_items, 1):
            # 记录单条推理开始时间
            t0 = time.time()
            # 调用 solve_single 进行单条推理，获取模型生成的原始文本
            raw_text = solve_single(model, tokenizer, problem, MAX_NEW_TOKENS, DO_SAMPLE)
            # 移除文本中的 <think> 标签及其内容，得到非思考部分文本
            no_think_text = remove_thinking(raw_text)
            # 移除答案前缀（如 "答案：" 等），得到最终提取的答案
            cleaned_text = remove_answer_prefix(no_think_text)
            t1 = time.time()
            # 检查答案格式和数值是否正确
            ok, _, msg = check_format_and_value(cleaned_text, gold)
            
            if mode == "demo":
                # 演示模式：详细输出每条推理过程
                print(f"【题目】{problem}")
                print(f"【模型原始输出】{raw_text}")
                print(f"【非思考部分文本】{no_think_text}")
                print(f"【提取答案】{cleaned_text}")
                print(msg)
                print(f"【耗时】{t1 - t0:.6f} 秒\n")
            else:
                # 评分模式：简洁输出进度
                # 用✓/✗标记正确/错误
                status = "✓" if ok else "✗"
                # 输出进度：[题目序号/总题数] 状态
                print(f"  [{idx}/{len(stage1_items)}] {status}")
            
            if ok:
                stage1_correct += 1
    # 记录阶段一总耗时
    stage1_elapsed = time.time() - stage1_start
    # 调用 score_report 输出阶段一的分数报告
    score_report("阶段一（逐条）", len(stage1_items), stage1_correct, stage1_elapsed, mode)

    # 计算第一阶段单条平均耗时
    stage1_avg_time = stage1_elapsed / len(stage1_items) if stage1_items else float("inf")

    # ===== 阶段二：批量推理 =====
    print("==== 阶段二：批量推理 ====")
    # 从 data/test_data_stage2.jsonl 读取测试数据
    stage2_items: List[Tuple[str, str]] = []
    test_data_2_path = os.path.join(os.path.dirname(__file__), "data", "test_data_2.jsonl")
    with open(test_data_2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                stage2_items.append((item["problem"], item["answer"]))

    problems_batch = [p for p, _ in stage2_items]

    # Grading 模式：先用单条推理测试相同的数据，以获得准确的加速比基准
    stage2_single_elapsed = None
    if mode == "grading":
        print("  正在用单条推理测试 32 条数据（用于计算加速比）...")
        with torch.inference_mode():
            stage2_single_start = time.time()
            for problem, _ in stage2_items:
                raw_text = solve_single(model, tokenizer, problem, MAX_NEW_TOKENS, DO_SAMPLE)
            stage2_single_elapsed = time.time() - stage2_single_start
        print(f"  单条推理完成，耗时：{stage2_single_elapsed:.3f} s")
        print("  正在进行批量推理...")

    with torch.inference_mode():
        stage2_start = time.time()
        raw_list = solve_batch(model, tokenizer, problems_batch, MAX_NEW_TOKENS, DO_SAMPLE)
        no_think_list = [remove_thinking(t) for t in raw_list]
        cleaned_list = [remove_answer_prefix(t) for t in no_think_list]
        stage2_elapsed = time.time() - stage2_start
    
    if mode == "grading":
        print(f"  批量推理完成，耗时：{stage2_elapsed:.3f} s")

    # 统计并展示结果
    stage2_correct = 0
    wrong_samples = []  # 存储错误样例
    
    # 先统计所有结果
    for idx, ((prob, gold), raw_text, no_think_text, cleaned_text) in enumerate(
        zip(stage2_items, raw_list, no_think_list, cleaned_list), 1
    ):
        ok, _, msg = check_format_and_value(cleaned_text, gold)
        
        if ok:
            stage2_correct += 1
        else:
            # 记录错误样例
            wrong_samples.append((idx, prob, gold, raw_text, no_think_text, cleaned_text, msg))
    
    # 演示模式：仅展示错误样例
    if mode == "demo":
        if wrong_samples:
            print("—— 阶段二错误样例（仅展示出错条目）——")
            for idx, prob, gold, raw_text, no_think_text, cleaned_text, msg in wrong_samples:
                print(f"【题目 {idx}/{len(stage2_items)}】{prob}")
                print(f"【模型原始输出】{raw_text}")
                print(f"【非思考部分文本】{no_think_text}")
                print(f"【提取答案】{cleaned_text}")
                print(msg)
                print()
        else:
            print("—— 阶段二全部正确，无错误样例。——\n")

    score_report("阶段二（批量32题）", len(stage2_items), stage2_correct, stage2_elapsed, mode)

    # 计算阶段二单条平均耗时
    stage2_avg_time = stage2_elapsed / len(stage2_items) if stage2_items else float("inf")
    
    if mode == "demo":
        print("==== 性能对比 ====")
        print(f"阶段一单条平均耗时：{stage1_avg_time*1000:.1f} ms/样本 ({stage1_avg_time:.3f} s)")
        print(f"阶段二单条平均耗时：{stage2_avg_time*1000:.1f} ms/样本 ({stage2_avg_time:.3f} s)")
        
        # 显示如果按评分模式的得分
        print(f"\n==== 如果按评分模式计分 ====")
        
        # 阶段一正确性
        if stage1_correct >= 5:
            stage1_accuracy_score = 50
        else:
            stage1_accuracy_score = stage1_correct * 10
        
        # 阶段一时间性能
        stage1_time_score = min(10, 50 / stage1_avg_time) if stage1_avg_time > 0 else 0
        
        # 阶段二正确性
        if stage2_correct >= 30:
            stage2_accuracy_score = 30
        else:
            stage2_accuracy_score = stage2_correct * 1
        
        # 阶段二时间性能：加速比测试（Demo 模式使用参考值）
        # Demo 模式：使用阶段一的平均时间作为参考（不够精确，仅供参考）
        if stage1_avg_time > 0 and stage2_avg_time > 0:
            speedup_ref = stage1_avg_time / stage2_avg_time
            # 3倍加速满分10分，1倍0分，线性给分
            if speedup_ref >= 3:
                stage2_time_score = 10
            elif speedup_ref <= 1:
                stage2_time_score = 0
            else:
                stage2_time_score = (speedup_ref - 1) * 5  # (speedup - 1) / (3 - 1) * 10
        else:
            speedup_ref = 0
            stage2_time_score = 0
        
        # 总分
        total_score = (stage1_accuracy_score + stage1_time_score + 
                      stage2_accuracy_score + stage2_time_score)
        
        print(f"阶段一正确性：{stage1_accuracy_score:.1f}/50")
        print(f"阶段一时间性能：{stage1_time_score:.1f}/10")
        print(f"阶段二正确性：{stage2_accuracy_score:.1f}/30")
        print(f"阶段二时间性能：{stage2_time_score:.1f}/10 (参考加速比: {speedup_ref:.2f}x)")
        print(f"注：Demo 模式的加速比仅供参考，Grading 模式会用单条推理测试相同数据以获得精确加速比")
        print(f"总分：{total_score:.1f}/100\n")
    else:
        # 评分模式：计算最终分数
        print("\n" + "="*50)
        print("========== 最终评分 ==========")
        print("="*50)
        
        # 阶段一正确性评分：每题10分，达到5题及以上给满分50分
        if stage1_correct >= 5:
            stage1_accuracy_score = 50
        else:
            stage1_accuracy_score = stage1_correct * 10
        
        # 阶段一时间性能评分：单条平均耗时≤5s满分，得分=min(10, 50/t)
        stage1_time_score = min(10, 50 / stage1_avg_time) if stage1_avg_time > 0 else 0
        
        # 阶段二正确性评分：每题1分，达到30题及以上给满分30分
        if stage2_correct >= 30:
            stage2_accuracy_score = 30
        else:
            stage2_accuracy_score = stage2_correct * 1
        
        # 阶段二时间性能评分：加速比测试，3倍加速满分10分，1倍0分，线性给分
        # Grading 模式：使用相同 32 条数据的单条推理时间作为基准
        if stage2_single_elapsed is not None:
            stage2_single_avg_time = stage2_single_elapsed / len(stage2_items)
            if stage2_single_elapsed > 0 and stage2_elapsed > 0:
                speedup = stage2_single_elapsed / stage2_elapsed
                # 3倍加速满分10分，1倍0分，线性给分
                if speedup >= 3:
                    stage2_time_score = 10
                elif speedup <= 1:
                    stage2_time_score = 0
                else:
                    stage2_time_score = (speedup - 1) * 5  # (speedup - 1) / (3 - 1) * 10
            else:
                speedup = 0
                stage2_time_score = 0
        else:
            stage2_single_avg_time = None
            speedup = 0
            stage2_time_score = 0
        
        # 总分
        total_score = (stage1_accuracy_score + stage1_time_score + 
                      stage2_accuracy_score + stage2_time_score)
        
        print(f"\n【阶段一：逐条推理 - 正确性】")
        print(f"  正确题数：{stage1_correct}/{len(stage1_items)}")
        print(f"  得分：{stage1_accuracy_score:.1f}/50  (每题10分，≥5题满分)")
        
        print(f"\n【阶段一：逐条推理 - 时间性能】")
        print(f"  单条平均耗时：{stage1_avg_time:.3f} s/样本")
        print(f"  得分：{stage1_time_score:.1f}/10  (公式: min(10, 50/{stage1_avg_time:.3f}))")
        
        print(f"\n【阶段二：批量推理 - 正确性】")
        print(f"  正确题数：{stage2_correct}/{len(stage2_items)}")
        print(f"  得分：{stage2_accuracy_score:.1f}/30  (每题1分，≥30题满分)")
        
        print(f"\n【阶段二：批量推理 - 时间性能】")
        print(f"  32条数据单条推理总耗时：{stage2_single_elapsed:.3f} s  (平均: {stage2_single_avg_time:.3f} s/样本)")
        print(f"  32条数据批量推理总耗时：{stage2_elapsed:.3f} s  (平均: {stage2_avg_time:.3f} s/样本)")
        print(f"  加速比：{speedup:.2f}x")
        print(f"  得分：{stage2_time_score:.1f}/10  (3倍加速满分，1倍0分，线性给分)")
        
        print(f"\n{'='*50}")
        print(f"【总分汇总】")
        print(f"  阶段一正确性：{stage1_accuracy_score:.1f}/50")
        print(f"  阶段一时间性能：{stage1_time_score:.1f}/10")
        print(f"  阶段二正确性：{stage2_accuracy_score:.1f}/30")
        print(f"  阶段二时间性能：{stage2_time_score:.1f}/10")
        print(f"  " + "-"*46)
        print(f"  总分：{total_score:.1f}/100")
        print(f"{'='*50}\n")
    

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="T1 评测程序 - 支持演示模式和评分模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "grading"],
        help="运行模式：demo（演示模式，详细输出）或 grading（评分模式，仅输出分数）"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(mode=args.mode if args.mode else "demo")