import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print("🚀 正在将 Qwen2.5-VL-7B 装载至 RTX 5090...")

# 1. 极速加载模型 (强制 BF16 和 FA2)
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda"
)

processor = AutoProcessor.from_pretrained(model_id)

# 2. 构造一个最简单的纯文本测试 (不加图，先测测显存)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用一句话证明你已经开启了FlashAttention-2加速。"}
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")

# 3. 预热推理
print("⚡ 正在进行首次推理预热...")
generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print("\n🎯 模型回复:", output_text[0])
print(f"💽 当前 5090 显存峰值占用: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")