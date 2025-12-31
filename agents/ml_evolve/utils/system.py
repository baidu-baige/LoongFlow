# -*- coding: utf-8 -*-
"""
This file define
"""
import platform
import shutil
import subprocess

import psutil


def get_hardware_context() -> dict:
    """
    Get System Info
    """
    info = []

    # cpu info
    try:
        cpu_count = os.cpu_count()
        cpu_freq = psutil.cpu_freq()
        freq_str = f" @ {cpu_freq.max:.2f}Mhz" if cpu_freq else ""
        info.append(f"CPU: {cpu_count} Cores{freq_str} ({platform.processor()})")
    except Exception as e:
        info.append(f"CPU: {os.cpu_count()} Cores (Detailed info unavailable: {e})")

    # ram info
    try:
        vm = psutil.virtual_memory()
        total_gb = round(vm.total / (1024 ** 3), 2)
        info.append(f"RAM: Total {total_gb} GB")
    except Exception:
        info.append("RAM: Detection failed.")

    # gpu info
    gpu_available = False
    if shutil.which("nvidia-smi"):
        try:
            count_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"], encoding="utf-8"
            )
            gpu_count = int(count_out.strip())

            details_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], encoding="utf-8"
            )

            info.append(f"GPU Available: True (Count: {gpu_count})")

            gpu_lines = details_out.strip().split('\n')
            for idx, line in enumerate(gpu_lines):
                name, mem = line.split(',')
                info.append(f"  - GPU {idx}: {name.strip()} ({mem.strip()})")

            gpu_available = True

        except Exception as e:
            info.append(f"GPU Check Failed: nvidia-smi found but execution error ({str(e)}).")
    else:
        info.append("GPU: None (nvidia-smi not found).")

    return {
        "hardware_info": "\n".join(info),
        "gpu_available": gpu_available
    }


import os


def get_directory_structure(path: str, max_files: int = 50) -> str:
    """
    get directory structure
    """
    if not os.path.exists(path):
        return f"Error: The path '{path}' does not exist."

    info = [f"Directory contents of: {path}"]

    try:
        with os.scandir(path) as entries:
            sorted_entries = sorted(entries, key=lambda e: (not e.is_dir(), e.name))

            file_count = 0
            for entry in sorted_entries:
                if file_count >= max_files:
                    info.append(f"... (Truncated, showing first {max_files} items)")
                    break

                if entry.is_dir():
                    info.append(f"  [DIR]  {entry.name}/")
                else:
                    size_bytes = entry.stat().st_size
                    size_str = _format_size(size_bytes)
                    info.append(f"  [FILE] {entry.name:<30} ({size_str})")

                file_count += 1

    except PermissionError:
        return f"Error: Permission denied accessing '{path}'."
    except Exception as e:
        return f"Error scanning directory: {str(e)}"

    if len(info) == 1:
        info.append("  (Empty directory)")

    return "\n".join(info)


def _format_size(size_bytes: int) -> str:
    """helper function to format size in bytes"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
