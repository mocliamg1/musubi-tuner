#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/wan2.2-i2v}"
CUDA_FLAVOR="${1:-${CUDA_FLAVOR:-cu124}}"

case "$CUDA_FLAVOR" in
    cu124|cu128|cu130)
        ;;
    *)
        echo "Unsupported CUDA flavor: $CUDA_FLAVOR" >&2
        echo "Use one of: cu124, cu128, cu130" >&2
        exit 1
        ;;
esac

TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_FLAVOR}"

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

need_cmd python3
need_cmd wget

mkdir -p "$MODEL_DIR"/diffusion_models "$MODEL_DIR"/text_encoders "$MODEL_DIR"/vae

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
python3 -m pip install -e "$ROOT_DIR"
python3 -m pip install ascii-magic matplotlib tensorboard prompt-toolkit

download_file() {
    local url="$1"
    local output="$2"

    if [[ -f "$output" ]]; then
        echo "Already present: $output"
        return 0
    fi

    wget -c --show-progress -O "$output" "$url"
}

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
    "$MODEL_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
    "$MODEL_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors" \
    "$MODEL_DIR/text_encoders/umt5_xxl_fp16.safetensors"

download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "$MODEL_DIR/vae/wan_2.1_vae.safetensors"

cat <<EOF
Install complete.

Python environment: $(python3 -c 'import sys; print(sys.executable)')
Models:
  $MODEL_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
  $MODEL_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
  $MODEL_DIR/text_encoders/umt5_xxl_fp16.safetensors
  $MODEL_DIR/vae/wan_2.1_vae.safetensors

Example usage:
  python src/musubi_tuner/wan_generate_video.py --task i2v-A14B \\
    --dit "$MODEL_DIR/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \\
    --dit_high_noise "$MODEL_DIR/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \\
    --t5 "$MODEL_DIR/text_encoders/umt5_xxl_fp16.safetensors" \\
    --vae "$MODEL_DIR/vae/wan_2.1_vae.safetensors"
EOF
