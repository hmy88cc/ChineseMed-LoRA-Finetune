# 医疗模型LoRA微调项目

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57.6-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目简介

本项目使用 **LoRA (Low-Rank Adaptation)** 技术对中文医疗对话数据进行模型微调，构建能够回答医疗问题的AI助手。项目包含完整的数据处理、模型训练、测试评估流程，支持多科室医疗问答场景。

### ✨ 核心特性

- ✅ **参数高效微调**：使用LoRA技术，仅训练少量参数即可实现领域适配
- ✅ **多科室覆盖**：支持内科、外科、儿科、妇产科、肿瘤科、男科6个科室
- ✅ **完整流程**：从数据准备到模型测试的全链路实现
- ✅ **可视化监控**：TensorBoard实时跟踪训练过程
- ✅ **灵活配置**：提供小规模（3000条）和全量（60000条）两种训练方案
- ✅ **Unsloth加速**：GPU环境训练速度提升2-5倍，显存降低50%-80%
- ✅ **4bit量化**：支持NF4/FP4量化，大幅降低显存需求
- ✅ **BLEU自动评估**：训练前后BLEU分数对比，量化模型效果

---

## 🏗️ 技术架构

### 技术栈

| 类别 | 技术/库 | 版本 | 用途 |
|------|---------|------|------|
| 编程语言 | Python | 3.11 | 项目开发 |
| 深度学习框架 | PyTorch | 2.10.0 | 模型训练和推理 |
| 自然语言处理 | Transformers | 4.57.6 | 模型加载和处理 |
| 参数高效微调 | PEFT | 0.18.1 | LoRA配置和应用 |
| 数据处理 | Datasets | - | 数据集处理 |
| 数据处理 | Pandas | 2.3.3 | CSV文件读取和处理 |
| 进度显示 | tqdm | 4.67.1 | 显示数据处理和训练进度 |
| 内存监控 | psutil | 7.1.3 | 监控系统内存使用情况 |
| 日志记录 | TensorBoard | 2.15.2 | 训练过程可视化 |

### 基础模型

- **模型名称**：distilgpt2
- **参数量**：约355M（GPT-2的蒸馏版本）
- **优势**：内存占用小，推理速度快，适合资源有限的环境

### LoRA配置

```python
LoraConfig(
    r=8,              # LoRA秩
    lora_alpha=16,    # LoRA alpha
    lora_dropout=0.05,# Dropout率
    bias="none",      # Bias配置
    task_type="CAUSAL_LM"  # 任务类型
)
```

---

## 📁 项目结构

```
29-高质量微调数据工程与评估/
├── 【数据集】中文医疗数据/          # 原始医疗对话数据
│   ├── IM_内科/                    # 内科数据
│   ├── Surgical_外科/              # 外科数据
│   ├── Pediatric_儿科/             # 儿科数据
│   ├── OAGD_妇产科/                # 妇产科数据
│   ├── Oncology_肿瘤科/            # 肿瘤科数据
│   └── Andriatria_男科/            # 男科数据
├── medical-lora-checkpoint/        # LoRA权重（3000条数据训练）
│   ├── checkpoint-1350/            # Epoch 1
│   ├── checkpoint-2700/            # Epoch 2
│   └── checkpoint-4050/            # Epoch 3（最终模型）
├── medical_model_simple/           # 基础模型文件
├── test_medical_model/             # 测试模型文件
├── logs_medical_lora/              # 训练日志（TensorBoard）
├── LoRA_医疗微调.py                # 训练脚本（3000条数据）
├── LoRA_医疗微调_6W.py             # 训练脚本（60000条数据）
├── LoRA_医疗微调_Unsloth_4bit.py   # ✨ Unsloth+4bit加速训练脚本
├── BLEU评估工具.py                 # ✨ BLEU自动评估工具
├── 测试_LoRA模型.py                # 模型测试脚本
├── README.md                       # 项目说明文档（中文）
├── README_EN.md                    # 项目说明文档（英文）
├── requirements.txt                # 依赖包清单
├── Unsloth和BLEU使用指南.md        # ✨ 新功能详细指南
└── 医疗模型LoRA微调流程图.md       # 详细流程图
```

---

## 🚀 快速开始

### 环境要求

- **操作系统**：Windows/Linux/macOS
- **Python**：3.8+
- **内存**：建议≥64GB（CPU训练）
- **存储**：至少10GB可用空间

### 安装依赖

```bash
pip install torch==2.10.0
pip install transformers==4.57.6
pip install peft==0.18.1
pip install datasets
pip install pandas==2.3.3
pip install tqdm==4.67.1
pip install psutil==7.1.3
pip install tensorboard==2.15.2
```

### 数据准备

将医疗对话CSV文件放置在 `【数据集】中文医疗数据/` 目录下，每个科室一个子文件夹：

```
【数据集】中文医疗数据/
├── IM_内科/
│   └── 内科5000-33000.csv
├── Surgical_外科/
│   └── 外科5-14000.csv
├── Pediatric_儿科/
│   └── 儿科5-14000.csv
├── OAGD_妇产科/
│   └── 妇产科6-28000.csv
├── Oncology_肿瘤科/
│   └── 肿瘤科5-10000.csv
└── Andriatria_男科/
    └── 男科5-13000.csv
```

**数据格式要求**：
- CSV文件包含列：`ask`（问题）、`answer`（回答）
- 支持多种编码格式：gbk、gb2312、gb18030、utf-8
- 问题和回答长度建议≤200字

---

## 💻 使用方法

### 方案一：标准训练（适合CPU环境）

使用每个科室500条数据，共3000条进行训练：

```bash
python LoRA_医疗微调.py
```

**训练配置**：
- 数据量：3000条（每科500条）
- 训练轮次：3 epochs
- 预计时间：约3小时（CPU）
- 输出目录：`./medical-lora-checkpoint/`

### 方案二：全量数据训练

使用每个科室10000条数据，共60000条进行训练：

```bash
python LoRA_医疗微调_6W.py
```

**训练配置**：
- 数据量：60000条（每科10000条）
- 训练轮次：2 epochs
- 预计时间：约60小时（CPU）
- 输出目录：`./medical-lora-checkpoint-full/`

### 方案三：Unsloth + 4bit加速训练（推荐GPU用户）✨

使用Unsloth优化和4bit量化，大幅提升训练速度：

```bash
python LoRA_医疗微调_Unsloth_4bit.py
```

**核心优势**：
- ⚡ **速度提升**: 2-5倍（GPU环境）
- 💾 **显存降低**: 50%-80%
- 📊 **自动评估**: 内置BLEU分数计算
- 🔄 **前后对比**: 自动对比微调前后效果

**实测效果**（100步训练）：
- BLEU-4从 0.0243 提升至 0.0657（**+170.4%**）
- 训练时间从45分钟缩短至15分钟（RTX 4090）

详细使用说明请查看：[Unsloth和BLEU使用指南.md](Unsloth和BLEU使用指南.md)

### 模型测试

训练完成后，使用测试脚本验证模型效果：

```bash
python 测试_LoRA模型.py
```

测试涵盖6个科室的典型问题：
- 内科：头晕、高血压
- 外科：膝盖疼痛
- 儿科：儿童咳嗽
- 妇产科：孕期呕吐
- 肿瘤科：肺癌症状
- 男科：尿频尿急

### BLEU自动评估 ✨

使用BLEU工具自动评估模型质量：

```bash
python BLEU评估工具.py
```

**功能特性**：
- 📊 多指标评估：BLEU-1/2/3/4
- 🔄 前后对比：自动计算提升百分比
- 📝 详细报告：生成JSON格式评估结果
- 🇨🇳 中文支持：集成jieba分词

---

## 📊 训练结果

### 性能指标

| 指标 | 3000条数据 | 60000条数据 |
|------|-----------|------------|
| 训练集大小 | 2700条 | 54000条 |
| 验证集大小 | 300条 | 6000条 |
| 训练轮次 | 3 epochs | 2 epochs |
| 总迭代步数 | 4050步 | 54000步 |
| 初始损失 | ~7.0 | ~7.0 |
| 最终损失 | ~1.5 | ~1.2 |
| 训练时间 | ~3小时 | ~60小时 |
| 内存峰值 | ~60GB | ~80GB |

### 收敛情况

训练过程中损失值稳定下降，表明模型有效学习了医疗对话模式：

```
Epoch 1: Loss 7.0 → 3.5
Epoch 2: Loss 3.5 → 2.0
Epoch 3: Loss 2.0 → 1.5
```

### 产出物

训练完成后生成以下文件：

1. **LoRA权重文件**
   - `adapter_model.safetensors`：LoRA适配器权重
   - `adapter_config.json`：LoRA配置参数
   - `optimizer.pt`：优化器状态
   - `scheduler.pt`：学习率调度器状态

2. **训练日志**
   - `events.out.tfevents.*`：TensorBoard日志文件

3. **查看训练曲线**
   ```bash
   tensorboard --logdir=logs_medical_lora
   ```

---

## 🔧 核心代码解析

### 1. 数据加载与处理

```python
def load_medical_data(data_dir, samples_per_dept=500):
    """加载医疗对话数据，支持多编码自动检测"""
    data = []
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'OAGD_妇产科': '妇产科',
        'Oncology_肿瘤科': '肿瘤科',
        'Andriatria_男科': '男科'
    }
    
    for dept_dir, dept_name in departments.items():
        # 读取CSV文件（自动检测编码）
        df = read_csv_with_encoding(file_path)
        
        # 提取问题和回答
        for _, row in df.iterrows():
            question = str(row['ask']).strip()
            answer = str(row['answer']).strip()
            
            # 数据过滤和格式化
            if len(question) <= 200 and len(answer) <= 200:
                data.append({
                    "instruction": "请回答以下医疗相关问题",
                    "input": question,
                    "output": answer
                })
    
    return Dataset.from_list(data)
```

### 2. LoRA配置

```python
# 配置LoRA参数
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)
```

### 3. 训练配置

```python
training_args = TrainingArguments(
    output_dir="./medical-lora-checkpoint",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-4,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

### 4. 模型推理

```python
def generate_medical_response(question):
    """生成医疗回答"""
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    )
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 📈 训练监控

### TensorBoard可视化

启动TensorBoard查看训练曲线：

```bash
tensorboard --logdir=logs_medical_lora
```

访问 `http://localhost:6006` 查看：
- 训练损失曲线
- 验证损失曲线
- 学习率变化
- 训练速度

### 内存监控

项目内置psutil监控，实时显示：
- 总内存使用
- 可用内存
- 内存峰值

---

## 🎯 应用场景

### 1. 在线医疗咨询
- 初步症状分析
- 就医建议
- 健康咨询

### 2. 医疗知识问答
- 疾病知识查询
- 用药指导
- 健康科普

### 3. 医疗教育辅助
- 医学生学习
- 病例分析
- 知识测试

### 4. 临床决策支持
- 辅助诊断
- 治疗方案建议
- 文献检索

---

## 🔮 改进方向

### 模型升级
- 🎯 使用更大的医疗专用模型（如Qwen1.5-1.8B、ChatGLM3-6B）
- 🎯 选择支持中文的多语言模型
- 🎯 引入医疗领域预训练模型

### 数据优化
- 🎯 扩大数据集至10000+条/科室
- 🎯 数据增强（同义改写、回译）
- 🎯 专业医疗人员审核数据质量

### 训练优化
- 🎯 调整LoRA参数（r=16, alpha=32）
- 🎯 增加训练轮次至5-10 epochs
- 🎯 使用动态学习率调度

### 评估优化
- 🎯 引入自动评估指标（BLEU、ROUGE）
- 🎯 医疗专家人工评估
- 🎯 与基准模型对比测试

---

## ⚠️ 注意事项

### 重要声明

⚠️ **本项目仅用于学术研究和教学目的**

1. **非医疗建议**：模型生成的回答不能作为专业医疗建议，仅供参考
2. **需要验证**：所有医疗相关回答应由专业医疗人员审核
3. **数据安全**：确保训练数据不包含患者隐私信息
4. **合规使用**：遵守当地医疗AI相关法规

### 硬件要求

- **CPU训练**：建议48线程以上，内存≥64GB
- **GPU训练**（可选）：NVIDIA GPU ≥8GB显存可显著加速训练
- **存储空间**：至少10GB可用空间

### 常见问题

**Q: 训练时内存不足怎么办？**
A: 减小batch_size或增加gradient_accumulation_steps

**Q: 如何提高模型回答质量？**
A: 使用更大的基础模型、增加训练数据、优化LoRA参数

**Q: 如何查看训练进度？**
A: 使用TensorBoard或查看控制台输出的loss值

---

## 📚 参考资料

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT库文档](https://huggingface.co/docs/peft/)
- [DistilGPT2模型](https://huggingface.co/distilbert/distilgpt2)

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 👥 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件

---

## 🌟 致谢

感谢以下开源项目的支持：
- Hugging Face Transformers
- PyTorch
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets

---

**如果这个项目对你有帮助，请给一个 ⭐ Star！**
