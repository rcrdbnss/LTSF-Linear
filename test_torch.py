import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch is running on GPU.")

        # Print GPU name
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Print GPU memory stats
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)} bytes")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0)} bytes")
    else:
        print("CUDA is not available. PyTorch is running on CPU.")
