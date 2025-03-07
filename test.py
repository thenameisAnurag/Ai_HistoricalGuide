import torch

print("CUDA Available:", torch.cuda.is_available())  
print("CUDA Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current CUDA Device:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA is NOT detected by PyTorch")
