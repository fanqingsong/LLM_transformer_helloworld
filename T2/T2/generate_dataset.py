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
    """统计文本中的数字个数（连续的数字算作一个）"""
    numbers = re.findall(r'\d+\.?\d*', text)
    return len(numbers)

def is_valid_answer(answer: str) -> bool:
    """检查答案是否符合要求（3位数：100-999）"""
    try:
        num = int(answer)
        return 100 <= num <= 999
    except:
        return False

def generate_math_problems(n_problems: int = 1024) -> List[dict]:
    """
    生成数学题目数据
    
    策略：
    1. 使用不同的运算符和表达方式
    2. 确保答案在100-999范围内
    3. 每个题目最多3个数字
    4. 使用多种表达方式增加多样性
    """
    
    # 运算符变体
    add_ops = ['+', '＋', '加', '加上', '加上', '加上']
    sub_ops = ['-', '－', '减', '减去', '减去', '减去']
    mul_ops = ['×', '乘', '乘以', '乘以', '乘以', '乘以']
    div_ops = ['/', '÷', '除', '除以', '除以', '除以']
    
    # 括号变体
    brackets = ['(', ')', '（', '）', '【', '】']
    
    # 前缀和后缀
    prefixes = ['计算：', '求解：', '算一算：', '计算', '求解', '算一算', '']
    suffixes = ['', '？', '。', '！']
    
    problems = []
    seen_problems = set()
    
    random.seed(42)  # 固定种子确保可重现
    
    while len(problems) < n_problems:
        # 随机选择题目类型
        problem_type = random.choice([
            'addition', 'subtraction', 'multiplication', 'division',
            'mixed_1', 'mixed_2', 'power', 'complex'
        ])
        
        problem_text = ""
        answer = 0
        
        if problem_type == 'addition':
            # 加法：确保答案在100-999
            a = random.randint(50, 500)
            b = random.randint(50, 500)
            answer = a + b
            if 100 <= answer <= 999:
                op = random.choice(add_ops)
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
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
        if (problem_text and 
            answer and 
            100 <= answer <= 999 and 
            count_numbers_in_text(problem_text) <= 3 and
            problem_text not in seen_problems):
            
            # 添加到结果
            problems.append({
                "problem": problem_text.strip(),
                "answer": str(answer)
            })
            seen_problems.add(problem_text)
            
            if len(problems) % 100 == 0:
                print(f"已生成 {len(problems)} 条题目...")
    
    return problems

def save_dataset(problems: List[dict], output_path: str):
    """保存数据集到JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + '\n')
    print(f"数据集已保存到: {output_path}")

def validate_dataset(problems: List[dict]) -> dict:
    """验证数据集质量"""
    stats = {
        'total': len(problems),
        'unique': len(set(p['problem'] for p in problems)),
        'valid_answers': sum(1 for p in problems if is_valid_answer(p['answer'])),
        'valid_numbers': sum(1 for p in problems if count_numbers_in_text(p['problem']) <= 3),
        'non_empty': sum(1 for p in problems if p['problem'].strip()),
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
