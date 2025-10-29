# LLM Transformer Helloworld

一个基于 PyTorch 和 Transformers 的大语言模型实战项目，包含两个核心任务：数学问题自动回答和数据相似度计算。

## 📚 项目概述

本项目旨在通过实践掌握大语言模型的基础应用，包括：
- 文本生成与推理
- 相似度计算与 embedding
- 批量处理与性能优化

## 🗂️ 项目结构

```
LLM_transformer_helloworld/
├── T1/          # 任务一：小小数学助手
│   ├── data/    # 测试数据
│   ├── submission.py   # 实现文件
│   └── evaluate.py     # 评测脚本
│
├── T2/          # 任务二：多样性数据生成与相似度计算
│   ├── data/    # 数据集（含 test_data_1.jsonl 和 dataset.jsonl）
│   ├── submission.py   # 实现文件
│   └── evaluate.py     # 评测脚本
│
├── documents-export-2025-10-17/  # 大模型文件（已忽略）
├── T1/Qwen/     # Qwen3-0.6B 模型文件（已忽略）
├── T2/Qwen/     # Qwen3-Embedding-0.6B 模型文件（已忽略）
└── README.md    # 本文档
```

> ⚠️ **注意**：大模型文件（`.safetensors`）体积较大，已通过 `.gitignore` 排除，不会提交到 git 仓库中。

## 🎯 任务说明

### T1 - 小小数学助手

**目标**：实现一个能够理解各种数学符号、自动回答数学问题的助手。

**核心功能**：
1. 识别各种数学符号（`+`、`⊕`、`＋`、`加` 等）
2. 自动计算并返回标准格式的答案
3. 支持逐条推理和批量推理两种模式

**技术要点**：
- 使用 Qwen3-0.6B 作为基础模型
- 设计合适的 system prompt
- 实现高效的批量推理
- 性能优化（3倍以上加速比）

**评分标准**（100 分）：
- 阶段一：逐条推理（60 分）- 正确性 50 分 + 时间性能 10 分
- 阶段二：批量推理（40 分）- 正确性 30 分 + 加速比测试 10 分

> 详细说明请参考 [T1/README.md](T1/T1/README.md)

### T2 - 多样性数据生成与相似度计算

**目标**：实现文本相似度计算，并生成高多样性、高质量的数学题目数据集。

**核心功能**：
1. 使用 embedding 模型计算文本余弦相似度
2. 生成 1024 条多样化的数学题目
3. 确保数据集满足质量和多样性要求

**技术要点**：
- 使用 Qwen3-Embedding-0.6B 作为 embedding 模型
- Last-token pooling
- L2 normalization
- 数据生成与质量控制

**评分标准**（100 分）：
- 第一部分：相似度计算准确性（50 分）
- 第二部分：数据多样性（50 分）
  - 数据质量扣分
  - 基于平均相似度的评分

> 详细说明请参考 [T2/README.md](T2/T2/README.md)

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
# PyTorch
# Transformers
pip install torch transformers
```

### 运行任务

#### 运行 T1

```bash
cd T1/T1

# 演示模式（详细输出）
python evaluate.py --mode demo

# 评分模式（简洁输出）
python evaluate.py --mode grading
```

#### 运行 T2

```bash
cd T2/T2

# 演示模式（详细输出）
python evaluate.py --mode demo

# 评分模式（简洁输出）
python evaluate.py --mode grading
```

## 📝 实现指南

### T1 实现要点

在 `submission.py` 中实现以下函数：

```python
def build_system_prompt() -> str:
    """编写 system prompt，引导模型识别数学符号并输出标准格式答案"""
    pass

def apply_chat_template_single(system_prompt: str, user_query: str, tokenizer) -> str:
    """将 system prompt 和用户问题组合成输入格式"""
    pass

def generate_single(model, tokenizer, input_text: str) -> torch.Tensor:
    """单条推理"""
    pass

def generate_batch(model, tokenizer, input_texts: List[str]) -> torch.Tensor:
    """批量推理（需要高效实现）"""
    pass
```

### T2 实现要点

在 `submission.py` 中实现以下函数：

```python
def compute_similarity(text1: str, text2: str, model, tokenizer) -> float:
    """计算两个文本的余弦相似度
    - 使用 last-token pooling
    - 进行 L2 normalization
    - 返回范围：[0, 1]
    """
    pass

# 生成 dataset.jsonl 文件（1024条数据）
def generate_dataset():
    """生成多样化的数学题目数据集"""
    pass
```

## 📊 评测说明

- **demo 模式**：详细展示每个测试用例的结果，便于调试和学习
- **grading 模式**：简洁输出，用于正式评测
- 每道题目都有标准答案，支持自动评分

## 🔧 注意事项

1. **大模型文件**：本项目使用的模型文件体积较大（GB 级别），已通过 `.gitignore` 排除
2. **不可修改文件**：`evaluate.py` 是评测脚本，不要修改其内容
3. **仅修改 `submission.py`**：所有实现都在 `submission.py` 中完成
4. **性能要求**：T1 需要实现批量推理的加速效果；T2 需要生成足够多样化的数据

## 📚 参考资料

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Qwen 模型介绍](https://github.com/QwenLM/Qwen)

各位好，huggingface作为境外网站受流量限制有时网速较慢，大家可以使用国内镜像网站+迅雷(如果git使用不熟练)，教程如下：
https://juejin.cn/post/7541297378126921769

## 📄 许可证

本项目仅用于学习目的。

