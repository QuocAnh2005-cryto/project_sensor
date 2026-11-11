#!/usr/bin/env python3
"""
setup_env.py
────────────
Tự động tạo virtual environment (.env) trong thư mục app/
và cài đặt toàn bộ thư viện từ requirements.txt

Cách dùng:
    python setup_env.py                  # tạo .env và cài đặt
    python setup_env.py --no-cupy        # bỏ qua CuPy (nếu không có GPU)
    python setup_env.py --cuda 114       # chỉ định phiên bản CUDA cho CuPy
"""

import os
import sys
import subprocess
import argparse
import platform

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ENV_DIR      = os.path.join(SCRIPT_DIR, ".env")
REQ_FILE     = os.path.join(SCRIPT_DIR, "requirements.txt")

# Python executable inside the new venv
if platform.system() == "Windows":
    VENV_PYTHON = os.path.join(ENV_DIR, "Scripts", "python.exe")
    VENV_PIP    = os.path.join(ENV_DIR, "Scripts", "pip.exe")
else:
    VENV_PYTHON = os.path.join(ENV_DIR, "bin", "python3")
    VENV_PIP    = os.path.join(ENV_DIR, "bin", "pip")


def run(cmd: list, check: bool = True):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, check=check)
    return result.returncode


def create_venv():
    if os.path.exists(ENV_DIR):
        print(f"[INFO] Virtual env already exists at: {ENV_DIR}")
        print("[INFO] Skipping creation. Delete .env folder to recreate.")
    else:
        print(f"[INFO] Creating virtual environment at: {ENV_DIR}")
        run([sys.executable, "-m", "venv", ENV_DIR])
        print("[OK]  Virtual environment created.")


def upgrade_pip():
    print("\n[INFO] Upgrading pip inside venv...")
    run([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"])


def install_requirements():
    print(f"\n[INFO] Installing packages from: {REQ_FILE}")
    run([VENV_PIP, "install", "-r", REQ_FILE])
    print("[OK]  Core packages installed.")


def install_cupy(cuda_version: str):
    """
    Install the matching cupy-cudaXXX wheel.
    cuda_version: '102', '114', '12x', etc.
    """
    pkg = f"cupy-cuda{cuda_version}"
    print(f"\n[INFO] Installing CuPy for CUDA {cuda_version}: {pkg}")
    rc = run([VENV_PIP, "install", pkg], check=False)
    if rc != 0:
        print(f"[WARN] CuPy install failed. You may need to install it manually:")
        print(f"       {VENV_PIP} install {pkg}")
    else:
        print("[OK]  CuPy installed.")


def print_summary():
    print("\n" + "="*60)
    print("  Setup Complete!")
    print("="*60)
    print(f"  Env location : {ENV_DIR}")
    print(f"  Python       : {VENV_PYTHON}")
    print()
    print("  To activate the environment:")
    if platform.system() == "Windows":
        activate = os.path.join(ENV_DIR, "Scripts", "activate.bat")
        print(f"    {activate}")
    else:
        activate = os.path.join(ENV_DIR, "bin", "activate")
        print(f"    source {activate}")
    print()
    print("  To run the application:")
    print(f"    {VENV_PYTHON} {os.path.join(SCRIPT_DIR, 'main.py')}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Setup .env for Predictive Maintenance app")
    parser.add_argument("--no-cupy", action="store_true",
                        help="Skip CuPy installation (use CPU-only mode)")
    parser.add_argument("--cuda", default=None, metavar="VER",
                        help="CUDA version suffix for cupy (e.g. 102, 114, 12x)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Predictive Maintenance – Environment Setup")
    print("  DX-DMVN | TwinXcelerator-V30")
    print("="*60)

    create_venv()
    upgrade_pip()
    install_requirements()

    if not args.no_cupy:
        cuda_ver = args.cuda
        if cuda_ver is None:
            # Auto-detect (simple heuristic)
            try:
                out = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL).decode()
                if "10.2" in out:
                    cuda_ver = "102"
                elif "11.4" in out or "11.3" in out or "11.2" in out:
                    cuda_ver = "114"
                elif "12" in out:
                    cuda_ver = "12x"
                else:
                    cuda_ver = "12x"
                print(f"[INFO] Auto-detected CUDA → cupy-cuda{cuda_ver}")
            except Exception:
                print("[WARN] nvcc not found. Defaulting to cupy-cuda12x.")
                print("       Override with: python setup_env.py --cuda 102")
                cuda_ver = "12x"
        install_cupy(cuda_ver)
    else:
        print("\n[INFO] Skipping CuPy (--no-cupy flag set).")

    print_summary()


if __name__ == "__main__":
    main()
