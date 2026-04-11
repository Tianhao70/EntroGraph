import torch
from torch.utils.data import Dataset, DataLoader
from qwen_vl_utils import process_vision_info

class QwenEvalDataset(Dataset):
    """
    高性能 Qwen 视觉数据集：将磁盘 IO 和预处理彻底下放给 CPU
    """
    def __init__(self, data_list, processor):
        self.data_list = data_list  # 期待格式: [{'image_path': '...', 'question': '...'}, ...]
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 构造 Qwen2.5-VL 专属的视觉消息字典
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item['image_path']},
                    {"type": "text", "text": item['question']}
                ]
            }
        ]
        
        # 2. CPU 在后台多线程完成模板渲染
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 返回 CPU 预处理好的数据（此时还没有转成 Tensor，交给 collate_fn 统一做）
        return {
            "text": text,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "raw_item": item # 保留原始信息以便后续打分对比
        }

def qwen_collate_fn(batch, processor):
    """
    将一页的数据打包成 Tensor，准备锁页传输
    """
    texts = [b["text"] for b in batch]
    image_inputs = [img for b in batch for img in (b["image_inputs"] or [])]
    video_inputs = [vid for b in batch for vid in (b["video_inputs"] or [])]
    raw_items = [b["raw_item"] for b in batch]

    # CPU 密集型操作：统一 tokenization 和图像张量化
    inputs = processor(
        text=texts,
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt"
    )
    return inputs, raw_items

def build_high_throughput_dataloader(data_list, processor, batch_size=1):
    """
    核心武器：48GB 内存锁页预抓取 DataLoader
    """
    dataset = QwenEvalDataset(data_list, processor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,           # 召唤 8 个 CPU 进程后台疯狂读图
        pin_memory=True,         # 开启 DMA 高速通道，直接把内存映射到 5090 显存
        collate_fn=lambda b: qwen_collate_fn(b, processor),
        drop_last=False,
        prefetch_factor=2        # 每个 worker 提前预取 2 个 batch
    )
    return dataloader

if __name__ == "__main__":
    print("✅ Qwen 高性能数据流引擎 (Data Engine) 编译就绪！")