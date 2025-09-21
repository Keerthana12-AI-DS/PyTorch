import torch

# ==============================================================
# 1. Creating Tensors
# ==============================================================
# A tensor is a multi-dimensional array (core data structure in PyTorch)

tensor_1d = torch.tensor([1, 2, 3])   # 1D tensor (vector)
print("1D Tensor (Vector):")
print(tensor_1d, "\n")

tensor_2d = torch.tensor([[1, 2], [3, 4]])   # 2D tensor (matrix)
print("2D Tensor (Matrix):")
print(tensor_2d, "\n")

random_tensor = torch.rand(2, 3)   # Random values between 0 and 1
print("Random Tensor (2x3):")
print(random_tensor, "\n")

zeros_tensor = torch.zeros(2, 3)   # Tensor filled with zeros
print("Zeros Tensor (2x3):")
print(zeros_tensor, "\n")

ones_tensor = torch.ones(2, 3)     # Tensor filled with ones
print("Ones Tensor (2x3):")
print(ones_tensor, "\n")


# ==============================================================
# 2. Tensor Operations: Indexing, Slicing, Reshaping
# ==============================================================
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])   # Shape: (3, 2)

# Indexing → fetch single element
element = tensor[1, 0]   # Row 1, Column 0
print(f"Indexed Element (Row 1, Column 0): {element}\n")

# Slicing → get a portion of the tensor
slice_tensor = tensor[:2, :]   # First 2 rows, all columns
print(f"Sliced Tensor (First two rows): \n{slice_tensor}\n")

# Reshaping → change shape without changing data
reshaped_tensor = tensor.view(2, 3)   # Reshape (3x2) → (2x3)
print(f"Reshaped Tensor (2x3): \n{reshaped_tensor}\n")


# ==============================================================
# 3. Common Tensor Functions: Broadcasting & Matrix Multiplication
# ==============================================================
tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])   # Shape (2,3)
tensor_b = torch.tensor([[10, 20, 30]])           # Shape (1,3)

# Broadcasting → smaller tensor expands automatically to match shape
broadcasted_result = tensor_a + tensor_b
print(f"Broadcasted Addition Result: \n{broadcasted_result}\n")

# Matrix Multiplication → essential in neural networks
matrix_multiplication_result = torch.matmul(tensor_a, tensor_a.T)
print(f"Matrix Multiplication Result (tensor_a * tensor_a^T): \n{matrix_multiplication_result}\n")


# ==============================================================
# 4. GPU Acceleration
# ==============================================================
# Deep learning models are heavy on matrix operations → GPUs are much faster than CPUs.
# PyTorch makes it easy to transfer tensors to GPU.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# Create very large tensors directly on GPU (if available)
tensor_size = (10000, 10000)
a = torch.randn(tensor_size, device=device)
b = torch.randn(tensor_size, device=device)

# Perform operation on GPU
c = a + b

# Move result back to CPU for printing
print("Result shape (moved to CPU for printing):", c.cpu().shape, "\n")

# Check current GPU memory usage
if device.type == 'cuda':
    print("Current GPU memory usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
    print(f"Cached:    {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
