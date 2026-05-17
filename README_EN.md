# Medical Model LoRA Fine-Tuning Project

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57.6-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

This project uses **LoRA (Low-Rank Adaptation)** technology to fine-tune Chinese medical dialogue data, building an AI assistant capable of answering medical questions. The project includes a complete pipeline from data processing to model training, testing, and evaluation, supporting multi-department medical Q&A scenarios.

### ✨ Key Features

- ✅ **Parameter-Efficient Fine-Tuning**: Using LoRA technology, only a small number of parameters need to be trained for domain adaptation
- ✅ **Multi-Department Coverage**: Supports 6 departments: Internal Medicine, Surgery, Pediatrics, Obstetrics & Gynecology, Oncology, Andrology
- ✅ **Complete Pipeline**: Full implementation from data preparation to model testing
- ✅ **Visual Monitoring**: Real-time training process tracking with TensorBoard
- ✅ **Flexible Configuration**: Provides both small-scale (3,000 samples) and full-scale (60,000 samples) training options

---

## 🏗️ Technical Architecture

### Technology Stack

| Category | Technology/Library | Version | Purpose |
|----------|-------------------|---------|---------|
| Programming Language | Python | 3.11 | Project Development |
| Deep Learning Framework | PyTorch | 2.10.0 | Model Training and Inference |
| Natural Language Processing | Transformers | 4.57.6 | Model Loading and Processing |
| Parameter-Efficient Fine-Tuning | PEFT | 0.18.1 | LoRA Configuration and Application |
| Data Processing | Datasets | - | Dataset Processing |
| Data Processing | Pandas | 2.3.3 | CSV File Reading and Processing |
| Progress Display | tqdm | 4.67.1 | Display Data Processing and Training Progress |
| Memory Monitoring | psutil | 7.1.3 | Monitor System Memory Usage |
| Logging | TensorBoard | 2.15.2 | Training Process Visualization |

### Base Model

- **Model Name**: distilgpt2
- **Parameters**: ~355M (distilled version of GPT-2)
- **Advantages**: Small memory footprint, fast inference speed, suitable for resource-constrained environments

### LoRA Configuration

```python
LoraConfig(
    r=8,              # LoRA rank
    lora_alpha=16,    # LoRA alpha
    lora_dropout=0.05,# Dropout rate
    bias="none",      # Bias configuration
    task_type="CAUSAL_LM"  # Task type
)
```

---

## 📁 Project Structure

```
29-High-Quality-Fine-Tuning-Data-Engineering-and-Evaluation/
├── 【数据集】中文医疗数据/          # Original medical dialogue data
│   ├── IM_内科/                    # Internal Medicine data
│   ├── Surgical_外科/              # Surgery data
│   ├── Pediatric_儿科/             # Pediatrics data
│   ├── OAGD_妇产科/                # Obstetrics & Gynecology data
│   ├── Oncology_肿瘤科/            # Oncology data
│   └── Andriatria_男科/            # Andrology data
├── medical-lora-checkpoint/        # LoRA weights (trained on 3,000 samples)
│   ├── checkpoint-1350/            # Epoch 1
│   ├── checkpoint-2700/            # Epoch 2
│   └── checkpoint-4050/            # Epoch 3 (final model)
├── medical_model_simple/           # Base model files
├── test_medical_model/             # Test model files
├── logs_medical_lora/              # Training logs (TensorBoard)
├── LoRA_医疗微调.py                # Training script (3,000 samples)
├── LoRA_医疗微调_6W.py             # Training script (60,000 samples)
├── 测试_LoRA模型.py                # Model testing script
├── README.md                       # Project documentation (Chinese)
├── README_EN.md                    # Project documentation (English)
└── 医疗模型LoRA微调流程图.md       # Detailed flowchart
```

---

## 🚀 Quick Start

### Requirements

- **Operating System**: Windows/Linux/macOS
- **Python**: 3.8+
- **Memory**: Recommended ≥64GB (CPU training)
- **Storage**: At least 10GB available space

### Install Dependencies

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

### Data Preparation

Place medical dialogue CSV files in the `【数据集】中文医疗数据/` directory, with one subfolder per department:

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

**Data Format Requirements**:
- CSV files contain columns: `ask` (question), `answer` (response)
- Supports multiple encoding formats: gbk, gb2312, gb18030, utf-8
- Recommended length for questions and answers: ≤200 characters

---

## 💻 Usage

### Option 1: Small-Scale Training (Recommended for Beginners)

Train using 500 samples per department, totaling 3,000 samples:

```bash
python LoRA_医疗微调.py
```

**Training Configuration**:
- Data size: 3,000 samples (500 per department)
- Training epochs: 3 epochs
- Estimated time: ~3 hours (CPU)
- Output directory: `./medical-lora-checkpoint/`

### Option 2: Full-Scale Training

Train using 10,000 samples per department, totaling 60,000 samples:

```bash
python LoRA_医疗微调_6W.py
```

**Training Configuration**:
- Data size: 60,000 samples (10,000 per department)
- Training epochs: 2 epochs
- Estimated time: ~60 hours (CPU)
- Output directory: `./medical-lora-checkpoint-full/`

### Model Testing

After training, use the test script to verify model performance:

```bash
python 测试_LoRA模型.py
```

Testing covers typical questions from 6 departments:
- Internal Medicine: Dizziness, hypertension
- Surgery: Knee pain
- Pediatrics: Children's cough
- Obstetrics & Gynecology: Pregnancy vomiting
- Oncology: Lung cancer symptoms
- Andrology: Frequent urination

---

## 📊 Training Results

### Performance Metrics

| Metric | 3,000 Samples | 60,000 Samples |
|--------|---------------|----------------|
| Training Set Size | 2,700 samples | 54,000 samples |
| Validation Set Size | 300 samples | 6,000 samples |
| Training Epochs | 3 epochs | 2 epochs |
| Total Iteration Steps | 4,050 steps | 54,000 steps |
| Initial Loss | ~7.0 | ~7.0 |
| Final Loss | ~1.5 | ~1.2 |
| Training Time | ~3 hours | ~60 hours |
| Peak Memory | ~60GB | ~80GB |

### Convergence

Loss values steadily decrease during training, indicating effective learning of medical dialogue patterns:

```
Epoch 1: Loss 7.0 → 3.5
Epoch 2: Loss 3.5 → 2.0
Epoch 3: Loss 2.0 → 1.5
```

### Outputs

The following files are generated after training:

1. **LoRA Weight Files**
   - `adapter_model.safetensors`: LoRA adapter weights
   - `adapter_config.json`: LoRA configuration parameters
   - `optimizer.pt`: Optimizer state
   - `scheduler.pt`: Learning rate scheduler state

2. **Training Logs**
   - `events.out.tfevents.*`: TensorBoard log files

3. **View Training Curves**
   ```bash
   tensorboard --logdir=logs_medical_lora
   ```

---

## 🔧 Core Code Analysis

### 1. Data Loading and Processing

```python
def load_medical_data(data_dir, samples_per_dept=500):
    """Load medical dialogue data with automatic encoding detection"""
    data = []
    departments = {
        'IM_内科': 'Internal Medicine',
        'Surgical_外科': 'Surgery',
        'Pediatric_儿科': 'Pediatrics',
        'OAGD_妇产科': 'Obstetrics & Gynecology',
        'Oncology_肿瘤科': 'Oncology',
        'Andriatria_男科': 'Andrology'
    }
    
    for dept_dir, dept_name in departments.items():
        # Read CSV file (automatic encoding detection)
        df = read_csv_with_encoding(file_path)
        
        # Extract questions and answers
        for _, row in df.iterrows():
            question = str(row['ask']).strip()
            answer = str(row['answer']).strip()
            
            # Data filtering and formatting
            if len(question) <= 200 and len(answer) <= 200:
                data.append({
                    "instruction": "Please answer the following medical question",
                    "input": question,
                    "output": answer
                })
    
    return Dataset.from_list(data)
```

### 2. LoRA Configuration

```python
# Configure LoRA parameters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
```

### 3. Training Configuration

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

### 4. Model Inference

```python
def generate_medical_response(question):
    """Generate medical response"""
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

## 📈 Training Monitoring

### TensorBoard Visualization

Start TensorBoard to view training curves:

```bash
tensorboard --logdir=logs_medical_lora
```

Visit `http://localhost:6006` to view:
- Training loss curve
- Validation loss curve
- Learning rate changes
- Training speed

### Memory Monitoring

Built-in psutil monitoring displays in real-time:
- Total memory usage
- Available memory
- Memory peak

---

## 🎯 Application Scenarios

### 1. Online Medical Consultation
- Preliminary symptom analysis
- Medical advice
- Health consultation

### 2. Medical Knowledge Q&A
- Disease knowledge queries
- Medication guidance
- Health education

### 3. Medical Education Support
- Medical student learning
- Case analysis
- Knowledge testing

### 4. Clinical Decision Support
- Diagnostic assistance
- Treatment plan suggestions
- Literature search

---

## 🔮 Improvement Directions

### Model Upgrade
- 🎯 Use larger medical-specific models (e.g., Qwen1.5-1.8B, ChatGLM3-6B)
- 🎯 Choose multilingual models with Chinese support
- 🎯 Introduce medical domain pre-trained models

### Data Optimization
- 🎯 Expand dataset to 10,000+ samples per department
- 🎯 Data augmentation (synonym rewriting, back-translation)
- 🎯 Professional medical personnel review data quality

### Training Optimization
- 🎯 Adjust LoRA parameters (r=16, alpha=32)
- 🎯 Increase training epochs to 5-10 epochs
- 🎯 Use dynamic learning rate scheduling

### Evaluation Optimization
- 🎯 Introduce automatic evaluation metrics (BLEU, ROUGE)
- 🎯 Manual evaluation by medical experts
- 🎯 Comparative testing with baseline models

---

## ⚠️ Important Notes

### Critical Disclaimer

⚠️ **This project is for academic research and educational purposes only**

1. **Not Medical Advice**: Model-generated responses should not be used as professional medical advice, for reference only
2. **Requires Verification**: All medical-related responses should be reviewed by professional medical personnel
3. **Data Security**: Ensure training data does not contain patient privacy information
4. **Compliance**: Comply with local medical AI regulations

### Hardware Requirements

- **CPU Training**: Recommended 48+ threads, memory ≥64GB
- **GPU Training** (Optional): NVIDIA GPU ≥8GB VRAM can significantly accelerate training
- **Storage Space**: At least 10GB available space

### FAQ

**Q: What if I run out of memory during training?**
A: Reduce batch_size or increase gradient_accumulation_steps

**Q: How to improve model response quality?**
A: Use a larger base model, increase training data, optimize LoRA parameters

**Q: How to monitor training progress?**
A: Use TensorBoard or check the loss values in console output

---

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT Library Documentation](https://huggingface.co/docs/peft/)
- [DistilGPT2 Model](https://huggingface.co/distilbert/distilgpt2)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👥 Contributing

Issues and Pull Requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or suggestions, please contact via:
- Submit an Issue
- Send an email

---

## 🌟 Acknowledgments

Thanks to the following open-source projects for their support:
- Hugging Face Transformers
- PyTorch
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets

---

**If this project helps you, please give it a ⭐ Star!**
