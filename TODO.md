# Qwen2.5-VL-7B Migration TODOs

## Technical Debt & Performance Optimizations (Recorded from Day 2)

**Issue Description:**
The legacy evaluation pipeline (e.g., in `inference_i2t_scores.py` and `playground/models.py`) deeply couples the data loading logic with the model inference loop. The benchmark datasets (`POPE`, `CHAIR`, `MME`, etc.) only return image paths from their `__getitem__` methods, and the image reading (`Image.open()`) and preprocessing (`process_images`) are executed synchronously within the model's `.eval()` phase.

**Impact:**
Because of this synchronous design, we are unable to use `DataLoader` correctly, leading to blocked GPU wait times during IO and preprocessing operations, severely bottlenecking the tensor transfer to the RTX 5090 GPU and wasting the 48GB of available RAM.

**Action Plan (Next Steps for New Pipeline):**
When building the brand new inference pipeline for **Qwen2.5-VL-7B**, ensure the following design from the ground up:
- [ ] **Decouple Data Logic from Inference Loop:** Rewrite dataset `__getitem__` functions to natively read images from disk and perform all preprocessing steps, directly returning PyTorch Tensors.
- [ ] **Implement High-Performance DataLoader:** Initialize a `DataLoader` for the new pipeline enforcing `num_workers=8` and `pin_memory=True`.
- [ ] **Max Throughput to 5090:** Use the 48GB RAM for memory-pinned asynchronous prefetching, feeding the RTX 5090 with zero-wait data streams. 
