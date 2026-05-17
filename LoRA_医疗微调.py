#!/usr/bin/env python
# coding: utf-8

# ### 医疗模型LoRA微调脚本
# ### 按照严格标准执行，确保训练完成、产出物完整、效果验证

# 导入必要的库
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
import pandas as pd
import psutil
from tqdm import tqdm
import time
from peft import LoraConfig, PeftModel, get_peft_model

# 设置模型参数
max_seq_length = 512  # 序列长度
samples_per_dept = 500  # 每个科室筛选500条数据
total_epochs = 3  # 训练轮次

# 加载预训练模型和分词器
# 使用distilgpt2作为替代模型
model_name = "distilgpt2"
print(f"加载模型: {model_name}")

# 加载分词器
print("加载分词器中...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 添加padding token
tokenizer.pad_token = tokenizer.eos_token
print("分词器加载完成！")

# 加载模型（使用CPU）
print("加载模型中...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # 使用float32在CPU上运行
    device_map="cpu"
)
print("模型加载完成！")

# 配置LoRA
print("配置LoRA中...")
lora_config = LoraConfig(
    r=8,  # LoRA秩
    lora_alpha=16,  # LoRA alpha
    lora_dropout=0.05,  #  dropout
    bias="none",  #  bias
    task_type="CAUSAL_LM",  # 任务类型
    # 不指定target_modules，让PEFT库自动选择适合的模块
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)
print("LoRA配置完成！")

# 定义医疗对话的提示模板
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

# 获取结束标记
EOS_TOKEN = tokenizer.eos_token

def read_csv_with_encoding(file_path):
    """尝试使用不同的编码读取CSV文件"""
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")

def load_medical_data(data_dir, samples_per_dept=500):
    """加载医疗对话数据，每个科室筛选500条数据"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'OAGD_妇产科': '妇产科',
        'Oncology_肿瘤科': '肿瘤科',
        'Andriatria_男科': '男科'
    }
    
    # 遍历所有科室目录
    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"目录不存在: {dept_path}")
            continue
            
        print(f"\n处理{dept_name}数据...")
        
        # 获取该科室下的所有CSV文件
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        
        dept_sample_count = 0
        
        for csv_file in csv_files:
            if dept_sample_count >= samples_per_dept:
                break
                
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在处理文件: {csv_file}")
            
            try:
                # 读取CSV文件
                df = read_csv_with_encoding(file_path)
                
                # 打印列名，帮助调试
                print(f"文件 {csv_file} 的列名: {df.columns.tolist()}")
                
                # 处理每一行数据
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{dept_name}数据"):
                    if dept_sample_count >= samples_per_dept:
                        break
                        
                    try:
                        # 获取问题和回答（尝试不同的列名）
                        question = None
                        answer = None
                        
                        # 尝试不同的列名
                        if 'question' in row:
                            question = str(row['question']).strip()
                        elif '问题' in row:
                            question = str(row['问题']).strip()
                        elif 'ask' in row:
                            question = str(row['ask']).strip()
                            
                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif '回答' in row:
                            answer = str(row['回答']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()
                        
                        # 过滤无效数据
                        if not question or not answer:
                            continue
                            
                        # 限制长度
                        if len(question) > 100 or len(answer) > 100:
                            continue
                            
                        # 添加到数据列表
                        data.append({
                            "instruction": "请回答以下医疗相关问题",
                            "input": question,
                            "output": answer
                        })
                        
                        dept_sample_count += 1
                        
                    except Exception as e:
                        print(f"处理数据行时出错: {e}")
                        continue
                        
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue
        
        print(f"{dept_name}数据处理完成，共筛选 {dept_sample_count} 条数据")
    
    # 验证数据
    if not data:
        raise ValueError("没有成功处理任何数据！")
        
    print(f"\n成功处理 {len(data)} 条数据")
    return Dataset.from_list(data)

def formatting_prompts_func(examples):
    """格式化提示"""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = medical_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def tokenize_function(examples):
    """分词处理"""
    outputs = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    # 添加labels，与input_ids相同
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# 加载医疗数据集
print("\n加载医疗数据集中...")
start_time = time.time()
dataset = load_medical_data("【数据集】中文医疗数据", samples_per_dept=samples_per_dept)
data_loading_time = time.time() - start_time
print(f"数据加载用时: {round(data_loading_time/60, 2)}分钟")

print("格式化数据中...")
dataset = dataset.map(formatting_prompts_func, batched=True)
print("分词处理中...")
dataset = dataset.map(tokenize_function, batched=True)

# 划分训练集和验证集
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./medical-lora-checkpoint",  # LoRA权重保存路径
    num_train_epochs=total_epochs,  # 训练轮次
    per_device_train_batch_size=1,  # 小批量大小以节省内存
    per_device_eval_batch_size=1,  # 验证批量大小
    gradient_accumulation_steps=2,  # 梯度累积以模拟更大的批量
    warmup_steps=100,  # 预热步骤
    weight_decay=0.01,  # 权重衰减
    logging_dir="logs_medical_lora",  # 日志保存路径
    logging_steps=10,  # 日志记录步长
    learning_rate=2e-4,  # 学习率
    save_strategy="epoch",  # 每轮保存一次
    eval_strategy="epoch",  # 每轮评估一次
    remove_unused_columns=False,
    report_to="tensorboard",  # 报告到TensorBoard
    load_best_model_at_end=True,  # 加载最佳模型
    metric_for_best_model="eval_loss"  # 以验证损失为最佳模型指标
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 显示当前内存状态
memory = psutil.virtual_memory()
print(f"\nCPU mode. Total memory: {round(memory.total / 1024**3, 2)} GB")
print(f"Available memory: {round(memory.available / 1024**3, 2)} GB")

# 开始训练
print("\n开始训练...")
train_start_time = time.time()
trainer_stats = trainer.train()
train_end_time = time.time()
train_time = train_end_time - train_start_time

# 显示训练后的时间统计
print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

memory = psutil.virtual_memory()
print(f"Final available memory: {round(memory.available / 1024**3, 2)} GB")

# 评估模型
print("\n评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 保存LoRA权重
print("\n保存LoRA权重...")
model.save_pretrained("./medical-lora-checkpoint")
tokenizer.save_pretrained("./medical-lora-checkpoint")
print("LoRA权重保存完成！")

# 模型推理示例
def generate_medical_response(question):
    """生成医疗回答"""
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

# 测试问题
test_questions = [
    "请问高血压患者日常饮食需要注意什么？",
    "孩子发烧38度，需要吃退烧药吗？",
    "胃不舒服，总是反酸，应该注意什么？"
]

print("\n测试模型推理...")
for question in test_questions:
    print("\n" + "="*50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question) 

print("\n微调完成！总训练时间：{:.2f}分钟".format(train_time/60))
print("\n微调后的预期结果包括：")
print("1. LoRA权重文件（medical-lora-checkpoint目录）")
print("2. 训练日志文件（logs_medical_lora目录）")
print("3. 模型配置文件")
print("4. 模型推理测试结果（多个医疗问题的回答示例）")
print("5. 训练过程的时间和内存统计")
print("6. 模型评估结果")
print("\n验证步骤：")
print("1. 检查medical-lora-checkpoint目录是否存在adapter_model.bin和adapter_config.json文件")
print("2. 运行推理测试，验证模型是否能回答医疗问题")
print("3. 查看训练日志，确认loss是否持续下降")
print("4. 检查评估结果，确认模型效果")
