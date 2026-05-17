#!/usr/bin/env python
# coding: utf-8

# 医疗模型LoRA微调脚本 - 全量数据版
# 使用每科10000条数据，共60000条进行LoRA微调

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
import pandas as pd
import psutil
from tqdm import tqdm
import time
from peft import LoraConfig, PeftModel, get_peft_model

max_seq_length = 512
samples_per_dept = 10000
total_epochs = 2

model_name = "distilgpt2"
print(f"加载模型: {model_name}")

print("加载分词器中...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("分词器加载完成！")

print("加载模型中...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
print("模型加载完成！")

print("配置LoRA中...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("LoRA配置完成！")

medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

EOS_TOKEN = tokenizer.eos_token

def read_csv_with_encoding(file_path):
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")

def load_medical_data(data_dir, samples_per_dept=10000):
    """加载医疗对话数据，每个科室筛选指定条数"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'OAGD_妇产科': '妇产科',
        'Oncology_肿瘤科': '肿瘤科',
        'Andriatria_男科': '男科'
    }
    
    total_per_dept = {}
    
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"目录不存在: {dept_path}")
            continue
            
        print(f"\n{'='*50}")
        print(f"处理 {dept_name} 数据...")
        
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        dept_sample_count = 0
        
        for csv_file in csv_files:
            if dept_sample_count >= samples_per_dept:
                break
                
            file_path = os.path.join(dept_path, csv_file)
            
            try:
                df = read_csv_with_encoding(file_path)
                total_rows = len(df)
                print(f"文件: {csv_file}，总行数: {total_rows}")
                print(f"列名: {df.columns.tolist()}")
                
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{dept_name}"):
                    if dept_sample_count >= samples_per_dept:
                        break
                        
                    try:
                        # 数据的列名为: department, title, ask, answer
                        question = None
                        answer = None
                        
                        if 'ask' in row:
                            question = str(row['ask']).strip()
                        elif 'question' in row:
                            question = str(row['question']).strip()
                        elif '问题' in row:
                            question = str(row['问题']).strip()
                            
                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif '回答' in row:
                            answer = str(row['回答']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()
                        
                        if not question or not answer:
                            continue
                            
                        if len(question) > 200 or len(answer) > 200:
                            continue
                            
                        data.append({
                            "instruction": "请回答以下医疗相关问题",
                            "input": question,
                            "output": answer
                        })
                        
                        dept_sample_count += 1
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue
        
        total_per_dept[dept_name] = dept_sample_count
        print(f"{dept_name} 完成，共筛选 {dept_sample_count} 条数据")
    
    if not data:
        raise ValueError("没有成功处理任何数据！")
    
    dept_summary = ", ".join([f"{k}: {v}条" for k, v in total_per_dept.items()])
    print(f"\n{'='*50}")
    print(f"数据加载完成！总计 {len(data)} 条数据")
    print(f"各科室分布: {dept_summary}")
    return Dataset.from_list(data)

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = medical_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("\n加载医疗数据集中...")
start_time = time.time()
dataset = load_medical_data("【数据集】中文医疗数据", samples_per_dept=samples_per_dept)
data_loading_time = time.time() - start_time
print(f"数据加载用时: {round(data_loading_time/60, 2)}分钟")

print("格式化数据中...")
dataset = dataset.map(formatting_prompts_func, batched=True)
print("分词处理中...")
dataset = dataset.map(tokenize_function, batched=True)

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

total_steps = (len(train_dataset) // (1 * 2)) * total_epochs
print(f"预计总迭代步数: {total_steps} 步")

training_args = TrainingArguments(
    output_dir="./medical-lora-checkpoint-full",
    num_train_epochs=total_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="logs_medical_lora_full",
    logging_steps=50,
    learning_rate=2e-4,
    save_strategy="epoch",
    eval_strategy="epoch",
    remove_unused_columns=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

memory = psutil.virtual_memory()
print(f"\nCPU mode. Total memory: {round(memory.total / 1024**3, 2)} GB")
print(f"Available memory: {round(memory.available / 1024**3, 2)} GB")
estimated_hours = round((total_steps * 2.7) / 3600, 1)
print(f"预计训练时间: 约 {estimated_hours} 小时")

print("\n开始训练...")
train_start_time = time.time()
trainer_stats = trainer.train()
train_end_time = time.time()
train_time = train_end_time - train_start_time

print(f"\n训练用时: {round(train_time/60, 2)} 分钟 ({round(train_time/3600, 2)} 小时)")
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")

memory = psutil.virtual_memory()
print(f"Final available memory: {round(memory.available / 1024**3, 2)} GB")

print("\n评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

print("\n保存LoRA权重...")
model.save_pretrained("./medical-lora-checkpoint-full")
tokenizer.save_pretrained("./medical-lora-checkpoint-full")
print("LoRA权重保存完成！")

def generate_medical_response(question):
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    ).to("cpu")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

test_questions = [
    "请问高血压患者日常饮食需要注意什么？",
    "孩子发烧38度，需要吃退烧药吗？",
    "胃不舒服，总是反酸，应该注意什么？",
    "最近总是头晕，血压有点高，应该怎么办？",
    "膝盖在运动后经常疼痛，需要做什么检查吗？"
]

print("\n" + "="*50)
print("测试模型推理...")
for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question)

print("\n" + "="*50)
print("微调完成！")
print(f"总训练时间：{round(train_time/60, 2)} 分钟")
print(f"使用数据：每科 {samples_per_dept} 条，共 {len(dataset)} 条")
print(f"训练轮次：{total_epochs} 轮")
print(f"产出物：")
print(f"  1. LoRA权重文件：./medical-lora-checkpoint-full/")
print(f"  2. 训练日志文件：./logs_medical_lora_full/")