#!/usr/bin/env python3
"""
Downloads Wan2.2 TI2V-5B model files from HuggingFace into ComfyUI's model
directories. Runs at container startup via the entrypoint before ComfyUI starts.
"""
import os
import shutil
import sys

COMFYUI_MODELS = "/app/ComfyUI/models"
REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
TEMP_DIR = "/tmp/wan22_cache"

# (path within HuggingFace repo, local ComfyUI model subdirectory)
MODELS = [
    ("split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "diffusion_models"),
    ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
    ("split_files/vae/wan2.2_vae.safetensors", "vae"),
]

def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[wan22-models] huggingface_hub not available, skipping.")
        sys.exit(0)

    for repo_path, subdir in MODELS:
        filename = os.path.basename(repo_path)
        dest_dir = os.path.join(COMFYUI_MODELS, subdir)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.exists(dest_path):
            size_gb = os.path.getsize(dest_path) / 1e9
            print(f"[wan22-models] Already present: {filename} ({size_gb:.1f} GB), skipping.")
            continue

        print(f"[wan22-models] Downloading {filename} ...")
        os.makedirs(dest_dir, exist_ok=True)

        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                local_dir=TEMP_DIR,
            )
            shutil.move(cached, dest_path)
            size_gb = os.path.getsize(dest_path) / 1e9
            print(f"[wan22-models] Saved: {dest_path} ({size_gb:.1f} GB)")
        except Exception as e:
            print(f"[wan22-models] ERROR downloading {filename}: {e}")
            sys.exit(1)

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("[wan22-models] All models ready.")

if __name__ == "__main__":
    main()
