#!/usr/bin/env python
# coding: utf-8

"""
医疗模型LoRA微调 - Unsloth + 4bit量化加速版
支持GPU/CPU差异化训练方案

核心优势：
- Unsloth优化：训练速度提升2-5倍
- 4bit量化：显存降低50%-80%
- BLEU自动评估：量化模型效果
"""

import os
import time
import torch
import pandas as pd
import psutil
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 尝试导入Unsloth（如果可用）
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth库已加载，将使用加速训练")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️ Unsloth库未安装，使用标准HuggingFace训练")
    print("   安装命令: pip install unsloth")

# 尝试导入BLEU评估库
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
    print("✅ NLTK库已加载，将使用BLEU评估")
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK库未安装，BLEU评估不可用")
    print("   安装命令: pip install nltk")

# ==================== 配置参数 ====================
# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # 可使用更大的医疗模型
MAX_SEQ_LENGTH = 512
SAMPLES_PER_DEPT = 100  # 测试用100条，实际可用10000条
TOTAL_EPOCHS = 2

# 量化配置
USE_4BIT = True  # 启用4bit量化
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"  # nf4或fp4
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_COMPUTE_DTYPE = torch.float16

# 训练配置
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50

# LoRA配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 输出目录
OUTPUT_DIR = "./medical-lora-checkpoint-unsloth"
LOG_DIR = "./logs_medical_lora_unsloth"


def check_hardware():
    """检测硬件环境并推荐训练方案"""
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"\n{'='*60}")
        print(f"🎮 GPU环境检测")
        print(f"{'='*60}")
        print(f"GPU数量: {gpu_count}")
        print(f"GPU型号: {gpu_name}")
        print(f"显存大小: {gpu_memory:.2f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
        
        if gpu_memory < 8:
            print(f"\n⚠️ 显存较小，强烈建议使用4bit量化")
            recommended方案 = "GPU + 4bit量化 + Unsloth"
        elif gpu_memory < 16:
            recommended方案 = "GPU + 4bit量化"
        else:
            recommended方案 = "GPU + 16bit全精度"
        
        print(f"💡 推荐方案: {recommended方案}")
    else:
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        total_memory = psutil.virtual_memory().total / (1024**3)
        
        print(f"\n{'='*60}")
        print(f"💻 CPU环境检测")
        print(f"{'='*60}")
        print(f"CPU核心数: {cpu_cores}")
        print(f"CPU线程数: {cpu_threads}")
        print(f"总内存: {total_memory:.2f} GB")
        print(f"\n⚠️ 未检测到GPU，将使用CPU训练")
        print(f"💡 建议: 使用小批量数据或考虑云端GPU服务")
    
    return has_gpu


def load_model_with_quantization(model_name, use_4bit=True):
    """
    加载模型并应用4bit量化
    
    Args:
        model_name: 模型名称或路径
        use_4bit: 是否使用4bit量化
    
    Returns:
        model, tokenizer
    """
    print(f"\n{'='*60}")
    print(f"📦 加载模型: {model_name}")
    print(f"{'='*60}")
    
    if UNSLOTH_AVAILABLE and use_4bit:
        # 使用Unsloth加载模型（自动优化）
        print("✨ 使用Unsloth加载模型（加速模式）")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # 自动选择
            load_in_4bit=use_4bit,
        )
        
        # 应用Unsloth的LoRA优化
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth优化的梯度检查点
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("✅ Unsloth模型加载完成")
        
    else:
        # 使用标准HuggingFace加载
        print("📚 使用标准HuggingFace加载模型")
        
        from transformers import BitsAndBytesConfig
        
        if use_4bit:
            # 配置4bit量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=LOAD_IN_4BIT,
                bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
                bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
                bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
            )
            print(f"🔧 启用4bit量化配置:")
            print(f"   - 量化类型: {BNB_4BIT_QUANT_TYPE}")
            print(f"   - 计算精度: {BNB_4BIT_COMPUTE_DTYPE}")
            print(f"   - 双重量化: {BNB_4BIT_USE_DOUBLE_QUANT}")
        else:
            bnb_config = None
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if use_4bit else torch.float32,
        )
        
        # 应用标准LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        print("✅ 标准模型加载完成")
    
    # 打印模型信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n📊 模型参数统计:")
    print(f"   - 总参数量: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    print(f"   - 训练比例: {trainable_params/total_params*100:.2f}%")
    
    return model, tokenizer


def read_csv_with_encoding(file_path):
    """尝试使用不同的编码读取CSV文件"""
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")


def load_medical_data(data_dir, samples_per_dept=100):
    """加载医疗对话数据"""
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
            print(f"⚠️ 目录不存在: {dept_path}")
            continue
            
        print(f"\n处理 {dept_name} 数据...")
        
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]
        dept_sample_count = 0
        
        for csv_file in csv_files:
            if dept_sample_count >= samples_per_dept:
                break
                
            file_path = os.path.join(dept_path, csv_file)
            
            try:
                df = read_csv_with_encoding(file_path)
                
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理{dept_name}", leave=False):
                    if dept_sample_count >= samples_per_dept:
                        break
                        
                    try:
                        question = None
                        answer = None
                        
                        if 'ask' in row:
                            question = str(row['ask']).strip()
                        elif 'question' in row:
                            question = str(row['question']).strip()
                            
                        if 'answer' in row:
                            answer = str(row['answer']).strip()
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
                        
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue
        
        total_per_dept[dept_name] = dept_sample_count
        print(f"{dept_name} 完成，共筛选 {dept_sample_count} 条数据")
    
    if not data:
        raise ValueError("没有成功处理任何数据！")
    
    dept_summary = ", ".join([f"{k}: {v}条" for k, v in total_per_dept.items()])
    print(f"\n{'='*60}")
    print(f"✅ 数据加载完成！总计 {len(data)} 条数据")
    print(f"各科室分布: {dept_summary}")
    
    return Dataset.from_list(data)


def calculate_bleu_scores(model, tokenizer, test_dataset, num_samples=10):
    """
    计算BLEU分数
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        test_dataset: 测试数据集
        num_samples: 评估样本数量
    
    Returns:
        avg_bleu_score: 平均BLEU分数
        bleu_scores: 每个样本的BLEU分数列表
    """
    if not NLTK_AVAILABLE:
        print("⚠️ NLTK未安装，无法计算BLEU分数")
        return None, None
    
    print(f"\n{'='*60}")
    print(f"📊 开始BLEU评估（{num_samples}个样本）")
    print(f"{'='*60}")
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    # 随机选择样本
    import random
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""
    
    for idx in tqdm(indices, desc="BLEU评估"):
        sample = test_dataset[idx]
        question = sample['input']
        reference_answer = sample['output']
        
        # 生成回答
        inputs = tokenizer(
            medical_prompt.format(question, ""),
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的回答部分
        if "### 回答：" in generated_text:
            generated_answer = generated_text.split("### 回答：")[-1].strip()
        else:
            generated_answer = generated_text
        
        # 计算BLEU分数
        reference_tokens = [reference_answer.split()]
        generated_tokens = generated_answer.split()
        
        if len(generated_tokens) == 0:
            bleu_score = 0.0
        else:
            try:
                bleu_score = sentence_bleu(
                    reference_tokens,
                    generated_tokens,
                    smoothing_function=smoothing
                )
            except:
                bleu_score = 0.0
        
        bleu_scores.append(bleu_score)
        
        # 打印示例（仅前3个）
        if idx < 3:
            print(f"\n样本 {idx+1}:")
            print(f"  问题: {question[:50]}...")
            print(f"  参考答案: {reference_answer[:50]}...")
            print(f"  生成答案: {generated_answer[:50]}...")
            print(f"  BLEU分数: {bleu_score:.4f}")
    
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    print(f"\n{'='*60}")
    print(f"📈 BLEU评估结果:")
    print(f"   - 平均BLEU分数: {avg_bleu:.4f}")
    print(f"   - 最高BLEU分数: {max(bleu_scores):.4f}")
    print(f"   - 最低BLEU分数: {min(bleu_scores):.4f}")
    print(f"   - 评估样本数: {len(bleu_scores)}")
    print(f"{'='*60}")
    
    return avg_bleu, bleu_scores


def main():
    """主训练流程"""
    print("\n" + "="*60)
    print("🚀 医疗模型LoRA微调 - Unsloth + 4bit量化版")
    print("="*60)
    
    # 1. 检测硬件环境
    has_gpu = check_hardware()
    
    # 2. 加载模型（带4bit量化）
    model, tokenizer = load_model_with_quantization(MODEL_NAME, use_4bit=USE_4BIT and has_gpu)
    
    # 3. 加载数据
    print(f"\n{'='*60}")
    print("📂 加载医疗数据集")
    print(f"{'='*60}")
    start_time = time.time()
    
    dataset = load_medical_data("【数据集】中文医疗数据", samples_per_dept=SAMPLES_PER_DEPT)
    data_loading_time = time.time() - start_time
    print(f"数据加载用时: {data_loading_time:.2f}秒")
    
    # 4. 数据预处理
    print("\n格式化数据中...")
    
    medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""
    
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = medical_prompt.format(input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 5. 划分训练集和验证集
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    val_dataset = train_test_split["test"]
    
    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 6. 训练前BLEU评估（基线）
    print(f"\n{'='*60}")
    print("📏 训练前BLEU评估（基线）")
    print(f"{'='*60}")
    baseline_bleu, _ = calculate_bleu_scores(model, tokenizer, val_dataset, num_samples=5)
    
    # 7. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TOTAL_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        logging_dir=LOG_DIR,
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        save_strategy="epoch",
        eval_strategy="epoch",
        remove_unused_columns=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=has_gpu and USE_4BIT,  # GPU时使用混合精度
    )
    
    # 8. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # 9. 显示资源状态
    memory = psutil.virtual_memory()
    print(f"\n{'='*60}")
    print(f"💾 系统资源状态")
    print(f"{'='*60}")
    print(f"总内存: {memory.total / 1024**3:.2f} GB")
    print(f"可用内存: {memory.available / 1024**3:.2f} GB")
    
    if has_gpu:
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU已用显存: {gpu_memory_used:.2f} GB")
        print(f"GPU预留显存: {gpu_memory_reserved:.2f} GB")
    
    # 估算训练时间
    total_steps = (len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * TOTAL_EPOCHS
    if UNSLOTH_AVAILABLE and has_gpu:
        estimated_hours = round((total_steps * 1.0) / 3600, 1)  # Unsloth加速
        print(f"\n⚡ 使用Unsloth加速，预计训练时间: 约 {estimated_hours} 小时")
    else:
        estimated_hours = round((total_steps * 2.7) / 3600, 1)
        print(f"\n⏱️  标准训练，预计训练时间: 约 {estimated_hours} 小时")
    
    # 10. 开始训练
    print(f"\n{'='*60}")
    print("🎯 开始训练...")
    print(f"{'='*60}")
    train_start_time = time.time()
    
    trainer_stats = trainer.train()
    
    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    
    print(f"\n✅ 训练完成！")
    print(f"训练用时: {train_time/60:.2f} 分钟 ({train_time/3600:.2f} 小时)")
    print(f"总迭代步数: {trainer_stats.metrics['train_runtime']:.0f} 秒")
    
    # 11. 训练后BLEU评估
    print(f"\n{'='*60}")
    print("📏 训练后BLEU评估")
    print(f"{'='*60}")
    final_bleu, _ = calculate_bleu_scores(model, tokenizer, val_dataset, num_samples=10)
    
    # 12. 计算BLEU提升
    if baseline_bleu is not None and final_bleu is not None:
        improvement = ((final_bleu - baseline_bleu) / baseline_bleu * 100) if baseline_bleu > 0 else 0
        print(f"\n{'='*60}")
        print(f"📈 BLEU提升统计")
        print(f"{'='*60}")
        print(f"训练前BLEU: {baseline_bleu:.4f}")
        print(f"训练后BLEU: {final_bleu:.4f}")
        print(f"绝对提升: {final_bleu - baseline_bleu:.4f}")
        print(f"相对提升: {improvement:.2f}%")
        print(f"{'='*60}")
    
    # 13. 保存模型
    print(f"\n💾 保存LoRA权重...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 模型已保存到: {OUTPUT_DIR}")
    
    # 14. 最终总结
    print(f"\n{'='*60}")
    print(f"🎉 微调完成！总结报告")
    print(f"{'='*60}")
    print(f"✓ 训练数据: {len(dataset)} 条")
    print(f"✓ 训练轮次: {TOTAL_EPOCHS} epochs")
    print(f"✓ 训练时间: {train_time/60:.2f} 分钟")
    print(f"✓ BLEU提升: {improvement:.2f}%" if baseline_bleu else "✓ BLEU评估: 未完成")
    print(f"✓ 量化方案: {'4bit量化' if USE_4BIT else '全精度'}")
    print(f"✓ 加速方案: {'Unsloth加速' if UNSLOTH_AVAILABLE else '标准训练'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
