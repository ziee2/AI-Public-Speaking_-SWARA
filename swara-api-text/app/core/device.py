"""
Device Detection Utility
Auto-detect dan konfigurasi device (CPU/GPU) untuk model ML
"""

import torch
import os


def get_device() -> str:
    """
    Deteksi device yang tersedia (CPU atau CUDA GPU)
    
    Returns:
        str: 'cuda' jika GPU tersedia, 'cpu' jika tidak
    """
    # Check environment variable override
    device_override = os.getenv("DEVICE", "").lower()
    if device_override in ["cpu", "cuda"]:
        print(f"🔧 Device override from env: {device_override}")
        return device_override
    
    # Auto-detect
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = "cpu"
        print("💻 No GPU detected, using CPU")
    
    return device


def get_device_info() -> dict:
    """
    Get detailed device information
    
    Returns:
        dict: Device information
    """
    device = get_device()
    
    info = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if device == "cuda":
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count()
        })
    else:
        info.update({
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads()
        })
    
    return info


def optimize_for_device(device: str):
    """
    Optimize PyTorch settings based on device
    
    Args:
        device: 'cpu' or 'cuda'
    """
    if device == "cpu":
        # Optimize CPU performance
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(min(cpu_count, 4))  # Limit threads to avoid overhead
        print(f"⚙️  PyTorch threads: {torch.get_num_threads()}")
    
    elif device == "cuda":
        # Optimize GPU performance
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
        print("⚡ GPU optimizations enabled")
