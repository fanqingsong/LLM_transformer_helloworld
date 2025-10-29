#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据生成脚本 - 生成1024条数学题目数据
"""

import json
import random
import re
from typing import List, Tuple, Set

def count_numbers_in_text(text: str) -> int:
    """
    统计文本中的数字个数（连续的数字算作一个）
    
    参数：
        text: 输入文本（如 "12 + 35"）
    
    返回：
        数字的数量
    
    示例：
        "12 + 35" → 2（两个数字）
        "12.5 + 3" → 2
        "123" → 1
    """
    # 使用正则表达式匹配所有数字（包括小数）
    # \d+: 一个或多个连续的数字
    # \.?: 可选的小数点
    # \d*: 小数点后的数字（可选）
    # 例如："12.5" → ["12.5"], "123 + 45" → ["123", "45"]
    numbers = re.findall(r'\d+\.?\d*', text)
    return len(numbers)

def is_valid_answer(answer: str) -> bool:
    """
    检查答案是否符合要求（3位数：100-999）
    
    参数：
        answer: 答案字符串（如 "123"）
    
    返回：
        True 如果答案在 100-999 范围内，否则 False
    
    用于验证：评测要求所有答案必须是3位数
    """
    try:
        num = int(answer)  # 将字符串转换为整数
        return 100 <= num <= 999  # 检查是否在有效范围内
    except:
        # 如果无法转换为整数（如 "abc"），返回 False
        return False

def generate_math_problems(n_problems: int = 1024) -> List[dict]:
    """
    生成数学题目数据
    
    策略：
    1. 使用不同的运算符和表达方式
    2. 确保答案在100-999范围内
    3. 每个题目最多3个数字
    4. 使用多种表达方式增加多样性
    
    参数：
        n_problems: 需要生成的题目数量（默认 1024）
    
    返回：
        List[dict]，每个字典包含 "problem" 和 "answer" 键
    
    数据要求：
        - 题目必须唯一（使用 set 去重）
        - 每个题目最多3个数字（使用 count_numbers_in_text 验证）
        - 答案必须是3位数 100-999
        - 题目多样性要足够高（平均相似度 ≤ 0.5）
    """
    
    # 运算符变体库：增加题目多样性
    # 同一个运算符有不同的表达方式，让模型生成的题目更加丰富
    add_ops = ['+', '＋', '加', '加上', '加上', '加上']      # 加法运算符变体
    sub_ops = ['-', '－', '减', '减去', '减去', '减去']      # 减法运算符变体
    mul_ops = ['×', '乘', '乘以', '乘以', '乘以', '乘以']   # 乘法运算符变体
    div_ops = ['/', '÷', '除', '除以', '除以', '除以']      # 除法运算符变体
    
    # 括号变体：用于复合表达式
    brackets = ['(', ')', '（', '）', '【', '】']
    
    # 前缀和后缀：增加表达方式的多样性
    # 例如："计算：12 + 35" 或 "12 + 35？"
    prefixes = ['计算：', '求解：', '算一算：', '计算', '求解', '算一算', '']
    suffixes = ['', '？', '。', '！']
    
    problems = []              # 存储生成的题目
    seen_problems = set()     # 用于去重，确保所有题目唯一
    
    # 固定随机种子：确保每次运行生成的题目相同
    # 这在调试和复现实验时非常重要
    random.seed(42)
    
    # 循环生成题目，直到达到目标数量
    while len(problems) < n_problems:
        # 随机选择题目类型
        # 8种题目类型：加法、减法、乘法、除法、混合运算1、混合运算2、幂运算、复杂表达式
        # 这样可以增加题目的多样性
        problem_type = random.choice([
            'addition', 'subtraction', 'multiplication', 'division',
            'mixed_1', 'mixed_2', 'power', 'complex'
        ])
        
        problem_text = ""   # 初始化题目文本
        answer = 0          # 初始化答案
        
        # 根据题目类型生成对应的表达式
        # 策略：先随机生成数字，计算答案，然后验证答案是否符合要求（100-999）
        
        if problem_type == 'addition':
            # 加法：确保答案在100-999
            # 策略：让两个数都在50-500范围内，这样它们的和更可能在100-999之间
            a = random.randint(50, 500)
            b = random.randint(50, 500)
            answer = a + b
            # 验证答案范围
            if 100 <= answer <= 999:
                op = random.choice(add_ops)          # 随机选择加法运算符变体
                prefix = random.choice(prefixes)      # 随机选择前缀
                suffix = random.choice(suffixes)     # 随机选择后缀
                problem_text = f"{prefix}{a} {op} {b}{suffix}"
        
        elif problem_type == 'subtraction':
            # 减法：确保答案在100-999
            a = random.randint(200, 999)
            b = random.randint(50, min(a-100, 500))
            answer = a - b
            if 100 <= answer <= 999:
                op = random.choice(sub_ops)
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                problem_text = f"{prefix}{a} {op} {b}{suffix}"
        
        elif problem_type == 'multiplication':
            # 乘法：确保答案在100-999
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            answer = a * b
            if 100 <= answer <= 999:
                op = random.choice(mul_ops)
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                problem_text = f"{prefix}{a} {op} {b}{suffix}"
        
        elif problem_type == 'division':
            # 除法：确保答案在100-999
            b = random.randint(2, 9)
            answer = random.randint(100, 999)
            a = answer * b
            if a <= 9999:  # 避免数字过大
                op = random.choice(div_ops)
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                problem_text = f"{prefix}{a} {op} {b}{suffix}"
        
        elif problem_type == 'mixed_1':
            # 混合运算1：(a + b) × c
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            c = random.randint(2, 9)
            answer = (a + b) * c
            if 100 <= answer <= 999:
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                add_op = random.choice(add_ops)
                mul_op = random.choice(mul_ops)
                problem_text = f"{prefix}({a} {add_op} {b}) {mul_op} {c}{suffix}"
        
        elif problem_type == 'mixed_2':
            # 混合运算2：a × b + c
            a = random.randint(10, 30)
            b = random.randint(3, 15)
            c = random.randint(50, 200)
            answer = a * b + c
            if 100 <= answer <= 999:
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                mul_op = random.choice(mul_ops)
                add_op = random.choice(add_ops)
                problem_text = f"{prefix}{a} {mul_op} {b} {add_op} {c}{suffix}"
        
        elif problem_type == 'power':
            # 幂运算：a^b
            a = random.randint(5, 15)
            b = random.randint(2, 3)
            answer = a ** b
            if 100 <= answer <= 999:
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                if random.random() < 0.5:
                    problem_text = f"{prefix}{a}的{b}次方{suffix}"
                else:
                    problem_text = f"{prefix}{a}^{b}{suffix}"
        
        elif problem_type == 'complex':
            # 复杂表达式：(a - b) × c + d
            a = random.randint(50, 200)
            b = random.randint(10, min(a-20, 100))
            c = random.randint(2, 8)
            d = random.randint(20, 100)
            answer = (a - b) * c + d
            if 100 <= answer <= 999:
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                sub_op = random.choice(sub_ops)
                mul_op = random.choice(mul_ops)
                add_op = random.choice(add_ops)
                problem_text = f"{prefix}({a} {sub_op} {b}) {mul_op} {c} {add_op} {d}{suffix}"
        
        # 检查题目是否有效
        # 必须满足：
        # 1. problem_text 不为空
        # 2. answer 在有效范围内（100-999）
        # 3. 题目中数字个数不超过3个
        # 4. 题目唯一（没有重复）
        if (problem_text and 
            answer and 
            100 <= answer <= 999 and 
            count_numbers_in_text(problem_text) <= 3 and  # 验证数字个数
            problem_text not in seen_problems):            # 验证唯一性
            
            # 添加到结果列表
            problems.append({
                "problem": problem_text.strip(),      # 题目文本（去除首尾空白）
                "answer": str(answer)                 # 答案（转换为字符串）
            })
            seen_problems.add(problem_text)           # 记录已生成的题目，避免重复
            
            # 进度提示：每生成100条打印一次
            if len(problems) % 100 == 0:
                print(f"已生成 {len(problems)} 条题目...")
    
    return problems  # 返回所有生成的题目

def save_dataset(problems: List[dict], output_path: str):
    """
    保存数据集到JSONL文件
    
    参数：
        problems: 题目列表，每个元素是 {"problem": ..., "answer": ...}
        output_path: 输出文件路径
    
    JSON Lines 格式（JSONL）：
        每行是一个 JSON 对象，便于逐行读取
        例如：
            {"problem": "12 + 35", "answer": "47"}
            {"problem": "91 + 24", "answer": "115"}
    
    用途：
        评测程序会读取这个文件，计算数据多样性和相似度
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for problem in problems:
            # 将字典转换为 JSON 字符串
            # ensure_ascii=False: 允许中文等非 ASCII 字符，不进行转义
            # 例如："计算：12 + 35" 而不是 "\\u8ba1\\u7b97"
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    print(f"数据集已保存到: {output_path}")

def validate_dataset(problems: List[dict]) -> dict:
    """
    验证数据集质量
    
    参数：
        problems: 题目列表
    
    返回：
        包含统计信息的字典
    
    验证项：
        1. 总数量：应该正好是 1024
        2. 唯一性：所有题目都不重复
        3. 有效答案：所有答案都在 100-999 范围内
        4. 数字个数：所有题目最多3个数字
        5. 非空题目：所有题目都不为空
    
    用于生成数据集后的质量检查
    """
    stats = {
        'total': len(problems),                                      # 总题目数
        'unique': len(set(p['problem'] for p in problems)),          # 唯一题目数（去重后）
        'valid_answers': sum(1 for p in problems if is_valid_answer(p['answer'])),  # 有效答案数
        'valid_numbers': sum(1 for p in problems if count_numbers_in_text(p['problem']) <= 3),  # 数字个数≤3的题目数
        'non_empty': sum(1 for p in problems if p['problem'].strip()),  # 非空题目数
    }
    
    print("数据集验证结果:")
    print(f"  总数量: {stats['total']}")
    print(f"  唯一题目: {stats['unique']}")
    print(f"  有效答案(100-999): {stats['valid_answers']}")
    print(f"  数字个数≤3: {stats['valid_numbers']}")
    print(f"  非空题目: {stats['non_empty']}")
    
    return stats

if __name__ == "__main__":
    print("开始生成1024条数学题目数据...")
    
    # 生成数据
    problems = generate_math_problems(1024)
    
    # 验证数据
    stats = validate_dataset(problems)
    
    # 保存数据
    output_path = "/home/song/workspace/chuanyu/T2/T2/data/dataset.jsonl"
    save_dataset(problems, output_path)
    
    print(f"\n✅ 数据生成完成！")
    print(f"生成了 {len(problems)} 条题目")
    print(f"所有题目都满足要求：唯一性、数字个数≤3、答案3位数")
