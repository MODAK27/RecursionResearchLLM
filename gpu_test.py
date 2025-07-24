import torch
import time

# Choose device: MPS (Apple GPU) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create random tensors
a = torch.randn(5000, 5000)
b = torch.randn(5000, 5000)

# Move to device
a = a.to(device)
b = b.to(device)

# Warm-up run (MPS backend needs a warm-up)
_ = torch.mm(a, b)

# Timed run
start = time.time()
c = torch.mm(a, b)
end = time.time()

print(f"Time taken for matrix multiply on {device}: {end - start:.4f} seconds")
