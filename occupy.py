import torch
import time

def occupy_memory(gpu_id=0, target_memory_gb=24):
    # 设置当前使用的 GPU
    torch.cuda.set_device(gpu_id)
    
    # 计算需要分配的内存大小（以GB为单位）
    memory_in_bytes = target_memory_gb * 1024 ** 3
    
    # 获取 float32 类型的张量所需的元素数量
    num_elements = memory_in_bytes // 4
    
    # 分配内存的张量
    try:
        print(f"Attempting to allocate {target_memory_gb} GB of memory on GPU {gpu_id}...")
        # 分配一个大的张量，但不进行任何计算操作
        tensor = torch.empty(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")
        print(f"Successfully allocated {tensor.numel() * 4 / (1024 ** 3):.2f} GB of memory on GPU {gpu_id}.")
    except RuntimeError as e:
        print(f"Error during memory allocation: {e}")
        return
    
    # 持续占用显存，尽量减少功耗
    try:
        while True:
            time.sleep(60 * 60)  # 每小时休眠一次，极大减少CPU/GPU负载
    except KeyboardInterrupt:
        print("Program terminated manually.")

if __name__ == "__main__":
    occupy_memory(gpu_id=0, target_memory_gb=21)
