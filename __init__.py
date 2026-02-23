import os
import shutil
import folder_paths

REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
TEMP_DIR = "/tmp/wan22_cache"
VAE_LOCAL_DIR = "/tmp/wan22_vae"

# All models download from HuggingFace to local disk on every cold start.
# Faster than GCS FUSE (~60-80 MB/s vs ~5-9 MB/s). Same pattern as LayerDiffusion.
#
# The VAE can't go to models/vae/ because the entrypoint symlinks that to GCS.
# Instead it goes to a local path and we register that path with folder_paths
# so ComfyUI finds it there.
MODELS = [
    (
        "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "/app/ComfyUI/models/diffusion_models",
    ),
    (
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "/app/ComfyUI/models/text_encoders",
    ),
    (
        "split_files/vae/wan2.2_vae.safetensors",
        VAE_LOCAL_DIR,
    ),
]

folder_paths.add_model_folder_path("vae", VAE_LOCAL_DIR)


def _download_models():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[wan22-models] huggingface_hub not available, skipping.")
        return

    for repo_path, dest_dir in MODELS:
        filename = os.path.basename(repo_path)
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

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("[wan22-models] All models ready.")


_download_models()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


