#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run with sudo:"
  echo "  sudo bash scripts/install_cuda_toolkit_wsl.sh"
  exit 1
fi

if ! grep -qi microsoft /proc/version; then
  echo "This script is intended for NVIDIA CUDA Toolkit installation inside WSL."
  exit 1
fi

echo "Installing CUDA Toolkit 12.8 for WSL-Ubuntu."
echo "This installs the toolkit only, not the Linux NVIDIA display driver."

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb

apt-get update
apt-get install -y cuda-toolkit-12-8

profile_file="/etc/profile.d/cuda-12-8.sh"
cat > "$profile_file" <<'EOF'
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
EOF

echo
echo "CUDA Toolkit installed."
echo "Open a new WSL shell or run:"
echo "  source /etc/profile.d/cuda-12-8.sh"
echo
/usr/local/cuda-12.8/bin/nvcc --version
