import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")