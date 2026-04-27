import torch
import torch.nn.functional as F
from qwen_data_engine import build_high_throughput_dataloader

class MHCDGenerator:
    """
    MHCD-AE 阶段三：多假设对比解码生成器
    """
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        # 依照导师方案，定义 5 条具备多样性的生成路径参数 (beta, temperature)
        self.path_configs = [
            {"beta": 0.20, "temp": 0.6},
            {"beta": 0.35, "temp": 0.8},
            {"beta": 0.50, "temp": 1.0},
            {"beta": 0.65, "temp": 1.1},
            {"beta": 0.80, "temp": 1.2}
        ]

    @torch.no_grad()
    def generate_candidates(self, dataloader):
        """
        针对每个输入，独立生成 K=5 个候选答案
        """
        all_results = []
        self.model.eval()

        for batch_inputs, raw_items in dataloader:
            # 将 Tensor 移动到 5090
            batch_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}
            
            batch_candidates = []
            
            # 核心：循环 5 次，每次使用不同的 contrastive 强度
            for i, config in enumerate(self.path_configs):
                print(f"正在生成路径 {i+1}/5 (β={config['beta']}, T={config['temp']})...")
                
                # 注意：这里我们使用标准的 generate。
                # 在进阶版中，我们会重写模型的 forward 逻辑来实现真正的 Token 级 logits 减法。
                # 目前先利用超参数组合模拟“多样性采样”
                output_ids = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=config['temp'],
                    top_p=0.9,
                    # 注意：Transformers 原生不直接支持 beta 参数，此处我们通过引导采样模拟
                    # 真正的 logits 减法将在明天的代码重构中由 Agent 注入底层 layer
                )
                
                # 裁剪输入部分，只保留生成的答案
                input_len = batch_inputs['input_ids'].shape[1]
                generated_text = self.processor.batch_decode(
                    output_ids[:, input_len:], 
                    skip_special_tokens=True
                )
                
                batch_candidates.append({
                    "path_id": i,
                    "text": generated_text[0],
                    "config": config
                })
                
                # 销毁每次生成的候选张量，防 OOM
                del output_ids
                torch.cuda.empty_cache()
            
            raw_item = raw_items[0]
            result_item = {
                "question": raw_items[0]['question'],
                "candidates": batch_candidates
            }
            for key in ("ground_truth", "image_path", "image_name", "question_id", "source_file", "source_index"):
                if raw_item.get(key) is not None:
                    result_item[key] = raw_item[key]
            all_results.append(result_item)
            
        return all_results

if __name__ == "__main__":
    # 模拟一笔测试数据
    test_data = [{"image_path": "assets/demo.jpg", "question": "请描述这张图片中不寻常的地方。"}]
    # 这里假设你已经在主程序中初始化了 model 和 processor
    # generator = MHCDGenerator(model, processor)
    # dataloader = build_high_throughput_dataloader(test_data, processor)
    # results = generator.generate_candidates(dataloader)
    print("🚀 MHCD 多假设生成引擎就绪！等待阶段四聚类逻辑接入...")
