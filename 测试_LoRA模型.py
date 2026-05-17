#!/usr/bin/env python
# coding: utf-8

# ### 测试LoRA微调后的医疗模型
# ### 为每个科室生成一个问题并测试模型回答

# 导入必要的库
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载预训练模型和分词器
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 添加padding token
tokenizer.pad_token = tokenizer.eos_token

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 加载LoRA权重
lora_path = "./medical-lora-checkpoint/checkpoint-4050"
model = PeftModel.from_pretrained(model, lora_path)
print(f"成功加载LoRA权重: {lora_path}")

# 定义医疗对话的提示模板
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

# 为每个科室生成一个问题
department_questions = {
    "内科": "我最近总是感觉头晕，血压有点高，应该怎么办？",
    "外科": "我的膝盖在运动后经常疼痛，需要做什么检查吗？",
    "儿科": "我的孩子3岁，最近经常咳嗽，应该吃什么药？",
    "妇产科": "我怀孕3个月，总是感到恶心呕吐，有什么缓解方法吗？",
    "肿瘤科": "肺癌早期有哪些症状需要注意？",
    "男科": "我有尿频、尿急的症状，可能是什么问题？"
}

# 生成医疗回答
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

# 测试每个科室的问题
print("\n测试LoRA微调后的医疗模型...")
for department, question in department_questions.items():
    print("\n" + "="*60)
    print(f"科室：{department}")
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question)

print("\n" + "="*60)
print("测试完成！")
