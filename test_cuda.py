import torch
import sys

print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print('is_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

device_arg = -1
if torch.cuda.is_available():
    try:
        name = torch.cuda.get_device_name(0)
        maj, minr = torch.cuda.get_device_capability(0)
        print(f"CUDA available on: {name} (capability {maj}.{minr})")
        if (maj, minr) >= (5, 2):
            device_arg = 0
            print("Using CUDA:0")
        else:
            print("GPU capability too low, using CPU")
    except Exception as e:
        print("Error probing CUDA:", e)
else:
    print("CUDA not available (likely wheel without support for your GPU).")

print("device_arg =", device_arg)

