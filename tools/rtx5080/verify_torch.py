import json

import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")

device = torch.device("cuda")
x = torch.randn((2048, 2048), device=device)
y = x @ x
torch.cuda.synchronize()
print(
    json.dumps(
        {
            "checksum": y.float().mean().item(),
            "compute_capability": torch.cuda.get_device_capability(0),
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        sort_keys=True,
    )
)
