#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BERT句子向量生成器

该脚本使用预训练的BERT模型将输入文本句子转换为固定维度的语义向量表示。
通过提取[MASK]标记位置的隐藏状态作为句子表示，能够捕捉上下文语义信息。

示例用法:
    python generate_vectors.py \
        --checkpoint ./bert_model \
        --corpus_file ./sentences.txt \
        --sentence_vectors_np_file ./output/vectors.npy
"""

import argparse
import json
import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn as nn
import string
from tqdm import tqdm

# 定义常见标点符号集合
PUNCTUATION = {'.', '!', '?', ';'}

def calculate_vectors(tokenizer, model, texts):
    """
    使用BERT模型计算输入文本的向量表示
    
    参数:
        tokenizer (BertTokenizer): 预训练的BERT分词器
        model (BertModel): 预训练的BERT模型
        texts (List[str]): 待处理的文本列表
        
    返回:
        numpy.ndarray: 文本向量矩阵，形状为[文本数量, 隐藏层维度]
    """
    # 对文本进行分词和编码，自动添加[CLS]/[SEP]等特殊标记
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # 将输入数据移动到GPU
    for _ in inputs:
        inputs[_] = inputs[_].cuda()

    temp = inputs["input_ids"]
    
    # 获取BERT模型的隐藏状态（不计算梯度）
    with torch.no_grad():
        # 获取最后一层的隐藏状态，形状为[batch_size, seq_len, hidden_dim]
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state.cpu()
        
    # 提取[MASK]标记位置对应的向量作为句子表示
    temp = temp.to(embeddings.device)
    embeddings = embeddings[temp == tokenizer.mask_token_id]

    # 转换为numpy数组
    embeddings = embeddings.numpy()

    return embeddings

def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 包含以下属性的对象:
            - checkpoint: BERT模型目录路径
            - corpus_file: 输入文本文件路径
            - sentence_vectors_np_file: 输出向量文件路径
    """
    parser = argparse.ArgumentParser(description="生成句子向量的BERT模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="BERT模型检查点目录")
    parser.add_argument("--corpus_file", type=str, required=True, help="输入文本文件路径，每行一个句子")
    parser.add_argument("--sentence_vectors_np_file", type=str, required=True, help="输出向量文件路径(.npy格式)")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 初始化BERT分词器
    tokenizer = BertTokenizer(vocab_file=os.path.join(args.checkpoint, "vocab.txt"))

    # 确保分词器包含[MASK]标记
    temp = {"mask_token": tokenizer.mask_token}
    tokenizer.add_special_tokens(temp)

    # 加载预训练BERT模型并移动到GPU
    model = BertModel.from_pretrained(args.checkpoint).cuda()
    
    # 设置设备（优先使用GPU）
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    # 多GPU并行处理
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()  # 设置为评估模式

    model = model.to(device)
    
    # 设置批处理大小
    batch_size = 128

    # 读取输入文本文件
    with open(args.corpus_file, "r", encoding='utf-8') as f:
        sentences = f.readlines()
    
    # 分批处理文本
    outputs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="计算向量中..."):
        batch_sentences = sentences[i:i+batch_size]
        batch = []
        for line in batch_sentences:
            text = line.strip()
            # 确保句子以标点符号结尾
            text = text + " ." if text.strip()[-1] not in PUNCTUATION else text
            # 构造包含[MASK]的模板文本
            text = '''This sentence : " ''' + text + ''' " means [MASK] .'''
            batch.append(text)
        # 计算当前批次的向量
        vectors = calculate_vectors(tokenizer=tokenizer, model=model, texts=batch)
        outputs.append(vectors)
    
    # 合并所有批次的向量
    outputs = np.concatenate(outputs, axis=0)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.sentence_vectors_np_file), exist_ok=True)
    
    # 保存向量到numpy二进制文件
    with open(args.sentence_vectors_np_file, "wb") as f:
        np.save(f, outputs)
