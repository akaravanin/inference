#!/usr/bin/env bash
# Install NVIDIA drivers + Docker GPU support on Ubuntu 22.04 / 24.04
set -euo pipefail

# ── 1. NVIDIA driver ──────────────────────────────────────────────────────────
echo "==> Installing NVIDIA driver..."
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# ── 2. NVIDIA Container Toolkit ───────────────────────────────────────────────
echo "==> Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# ── 3. Configure Docker runtime ───────────────────────────────────────────────
echo "==> Configuring Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "Done. REBOOT required for the NVIDIA driver to take effect:"
echo "  sudo reboot"
echo ""
echo "After reboot, verify with:"
echo "  nvidia-smi"
echo "  docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
