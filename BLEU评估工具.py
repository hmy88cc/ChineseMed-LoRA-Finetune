#!/usr/bin/env python
# coding: utf-8

"""
医疗模型BLEU自动评估工具

功能：
1. 计算微调前后BLEU分数对比
2. 支持多指标评估（BLEU-1, BLEU-2, BLEU-3, BLEU-4）
3. 生成详细评估报告
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import jieba
import re
from tqdm import tqdm


class MedicalBLEUEvaluator:
    """医疗模型BLEU评估器"""
    
    def __init__(self, model_path, base_model_name="distilgpt2"):
        """
        初始化评估器
        
        Args:
            model_path: LoRA模型路径
            base_model_name: 基础模型名称
        """
        print(f"📦 加载模型: {base_model_name}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # 加载LoRA权重
        print(f"🔧 加载LoRA权重: {model_path}")
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()
        
        print("✅ 模型加载完成")
        
        # 医疗对话模板
        self.medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""
    
    def chinese_tokenizer(self, text):
        """中文分词（使用jieba）"""
        # 使用jieba进行中文分词
        words = jieba.lcut(text)
        # 过滤标点符号和空格
        words = [w for w in words if w.strip() and not re.match(r'^[^\w\u4e00-\u9fff]+$', w)]
        return words
    
    def generate_response(self, question, max_new_tokens=150):
        """
        生成医疗回答
        
        Args:
            question: 问题文本
            max_new_tokens: 最大生成token数
        
        Returns:
            generated_text: 生成的回答
        """
        inputs = self.tokenizer(
            self.medical_prompt.format(question, ""),
            return_tensors="pt"
        ).to("cpu")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "### 回答：" in generated_text:
            answer = generated_text.split("### 回答：")[-1].strip()
        else:
            answer = generated_text
        
        return answer
    
    def calculate_sentence_bleu(self, reference, hypothesis, n_grams=4):
        """
        计算单句BLEU分数
        
        Args:
            reference: 参考答案（字符串）
            hypothesis: 假设答案（字符串）
            n_grams: N-gram的最大值（1-4）
        
        Returns:
            bleu_scores: dict，包含BLEU-1到BLEU-n的分数
        """
        smoothing = SmoothingFunction().method1
        
        # 中文分词
        ref_tokens = self.chinese_tokenizer(reference)
        hyp_tokens = self.chinese_tokenizer(hypothesis)
        
        if len(hyp_tokens) == 0:
            return {f'BLEU-{i}': 0.0 for i in range(1, n_grams + 1)}
        
        bleu_scores = {}
        for n in range(1, n_grams + 1):
            try:
                weights = tuple([1.0/n] * n + [0.0] * (4 - n))
                score = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores[f'BLEU-{n}'] = score
            except:
                bleu_scores[f'BLEU-{n}'] = 0.0
        
        return bleu_scores
    
    def calculate_corpus_bleu(self, references, hypotheses, n_grams=4):
        """
        计算语料库级别的BLEU分数
        
        Args:
            references: 参考答案列表
            hypotheses: 假设答案列表
            n_grams: N-gram的最大值
        
        Returns:
            corpus_bleu_scores: dict，包含各阶BLEU分数
        """
        smoothing = SmoothingFunction().method1
        
        # 中文分词
        ref_tokens_list = [self.chinese_tokenizer(ref) for ref in references]
        hyp_tokens_list = [self.chinese_tokenizer(hyp) for hyp in hypotheses]
        
        corpus_bleu_scores = {}
        for n in range(1, n_grams + 1):
            try:
                weights = tuple([1.0/n] * n + [0.0] * (4 - n))
                score = corpus_bleu(
                    [[ref] for ref in ref_tokens_list],
                    hyp_tokens_list,
                    weights=weights,
                    smoothing_function=smoothing
                )
                corpus_bleu_scores[f'BLEU-{n}'] = score
            except:
                corpus_bleu_scores[f'BLEU-{n}'] = 0.0
        
        return corpus_bleu_scores
    
    def evaluate_dataset(self, test_data, num_samples=20):
        """
        评估测试数据集
        
        Args:
            test_data: 测试数据列表，每个元素为{'input': question, 'output': answer}
            num_samples: 评估样本数量
        
        Returns:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"📊 开始BLEU评估（{num_samples}个样本）")
        print(f"{'='*60}")
        
        import random
        if num_samples < len(test_data):
            sampled_data = random.sample(test_data, num_samples)
        else:
            sampled_data = test_data
        
        results = {
            'samples': [],
            'bleu_scores': [],
            'corpus_references': [],
            'corpus_hypotheses': []
        }
        
        for idx, sample in enumerate(tqdm(sampled_data, desc="评估进度")):
            question = sample['input']
            reference_answer = sample['output']
            
            # 生成回答
            generated_answer = self.generate_response(question)
            
            # 计算BLEU分数
            bleu_scores = self.calculate_sentence_bleu(reference_answer, generated_answer)
            
            # 保存结果
            results['samples'].append({
                'index': idx,
                'question': question,
                'reference': reference_answer,
                'generated': generated_answer,
                'bleu_scores': bleu_scores
            })
            
            results['bleu_scores'].append(bleu_scores)
            results['corpus_references'].append(reference_answer)
            results['corpus_hypotheses'].append(generated_answer)
            
            # 打印前3个示例
            if idx < 3:
                print(f"\n--- 样本 {idx+1} ---")
                print(f"问题: {question[:80]}...")
                print(f"参考: {reference_answer[:80]}...")
                print(f"生成: {generated_answer[:80]}...")
                print(f"BLEU-4: {bleu_scores.get('BLEU-4', 0):.4f}")
        
        # 计算平均BLEU分数
        avg_bleu = {}
        for n in range(1, 5):
            key = f'BLEU-{n}'
            scores = [s[key] for s in results['bleu_scores']]
            avg_bleu[key] = sum(scores) / len(scores) if scores else 0.0
        
        results['avg_bleu'] = avg_bleu
        
        # 计算语料库级别BLEU
        corpus_bleu_scores = self.calculate_corpus_bleu(
            results['corpus_references'],
            results['corpus_hypotheses']
        )
        results['corpus_bleu'] = corpus_bleu_scores
        
        # 打印总结
        print(f"\n{'='*60}")
        print(f"📈 BLEU评估结果总结")
        print(f"{'='*60}")
        print(f"\n平均BLEU分数（句子级别）:")
        for key, value in avg_bleu.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n语料库BLEU分数:")
        for key, value in corpus_bleu_scores.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n评估样本数: {len(sampled_data)}")
        print(f"{'='*60}")
        
        return results
    
    def compare_before_after(self, test_data, baseline_model_path=None, num_samples=20):
        """
        对比微调前后的BLEU分数
        
        Args:
            test_data: 测试数据
            baseline_model_path: 基线模型路径（可选，如果没有则使用未加载LoRA的基础模型）
            num_samples: 评估样本数
        
        Returns:
            comparison_results: 对比结果
        """
        print(f"\n{'='*60}")
        print(f"🔄 微调前后BLEU对比评估")
        print(f"{'='*60}")
        
        # 1. 评估微调后模型（当前已加载）
        print(f"\n【步骤1】评估微调后模型...")
        after_results = self.evaluate_dataset(test_data, num_samples)
        
        # 2. 评估基线模型
        print(f"\n【步骤2】评估基线模型...")
        
        if baseline_model_path:
            # 加载基线LoRA模型
            baseline_model = PeftModel.from_pretrained(self.base_model, baseline_model_path)
            baseline_model.eval()
            old_model = self.model
            self.model = baseline_model
            
            before_results = self.evaluate_dataset(test_data, num_samples)
            
            # 恢复原模型
            self.model = old_model
        else:
            # 使用基础模型（无LoRA）
            print("使用基础模型作为基线")
            old_model = self.model
            self.model = self.base_model
            self.model.eval()
            
            before_results = self.evaluate_dataset(test_data, num_samples)
            
            # 恢复LoRA模型
            self.model = old_model
        
        # 3. 计算提升
        print(f"\n{'='*60}")
        print(f"📊 微调效果对比")
        print(f"{'='*60}")
        
        comparison = {
            'before': before_results,
            'after': after_results,
            'improvement': {}
        }
        
        for key in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
            before_score = before_results['avg_bleu'][key]
            after_score = after_results['avg_bleu'][key]
            
            if before_score > 0:
                improvement_pct = ((after_score - before_score) / before_score) * 100
            else:
                improvement_pct = 0.0
            
            comparison['improvement'][key] = {
                'before': before_score,
                'after': after_score,
                'absolute_improvement': after_score - before_score,
                'relative_improvement_pct': improvement_pct
            }
            
            print(f"\n{key}:")
            print(f"  微调前: {before_score:.4f}")
            print(f"  微调后: {after_score:.4f}")
            print(f"  绝对提升: {after_score - before_score:.4f}")
            print(f"  相对提升: {improvement_pct:.2f}%")
        
        print(f"\n{'='*60}")
        
        return comparison
    
    def save_report(self, results, output_path="bleu_evaluation_report.json"):
        """
        保存评估报告
        
        Args:
            results: 评估结果
            output_path: 输出文件路径
        """
        # 简化结果以便保存（移除过长的文本）
        simplified_results = {
            'avg_bleu': results.get('avg_bleu', {}),
            'corpus_bleu': results.get('corpus_bleu', {}),
            'num_samples': len(results.get('samples', [])),
            'sample_summaries': []
        }
        
        for sample in results.get('samples', [])[:5]:  # 只保存前5个示例
            simplified_results['sample_summaries'].append({
                'index': sample['index'],
                'question': sample['question'][:100],
                'reference': sample['reference'][:100],
                'generated': sample['generated'][:100],
                'bleu_scores': sample['bleu_scores']
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 评估报告已保存到: {output_path}")


def main():
    """主函数 - 演示BLEU评估"""
    
    # 示例测试数据
    test_data = [
        {
            "input": "我最近总是感觉头晕，血压有点高，应该怎么办？",
            "output": "建议您定期监测血压，保持低盐饮食，适当运动，避免情绪激动。如果血压持续偏高，请及时就医，医生可能会根据您的情况开具降压药物。"
        },
        {
            "input": "我的孩子3岁，最近经常咳嗽，应该吃什么药？",
            "output": "儿童咳嗽需要先明确原因，可能是感冒、过敏或其他呼吸道疾病。建议先带孩子到医院儿科就诊，不要自行用药。医生会根据具体情况给出治疗方案。"
        },
        {
            "input": "膝盖在运动后经常疼痛，需要做什么检查吗？",
            "output": "运动后膝盖疼痛可能与半月板损伤、韧带拉伤或关节炎有关。建议进行膝关节X光或MRI检查，同时减少剧烈运动，注意休息。如疼痛持续，请咨询骨科医生。"
        },
    ]
    
    # 更多测试数据可以从CSV文件加载
    print("💡 提示: 可以从您的医疗数据集中加载更多测试样本")
    
    # 初始化评估器
    evaluator = MedicalBLEUEvaluator(
        model_path="./medical-lora-checkpoint/checkpoint-4050",
        base_model_name="distilgpt2"
    )
    
    # 执行评估
    results = evaluator.evaluate_dataset(test_data, num_samples=len(test_data))
    
    # 保存报告
    evaluator.save_report(results, "bleu_evaluation_report.json")
    
    print("\n🎉 BLEU评估完成！")


if __name__ == "__main__":
    main()
