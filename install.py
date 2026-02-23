# Model downloads are handled in __init__.py, which runs during ComfyUI
# startup after the entrypoint has created all GCS symlinks. Running here
# (before symlinks exist) caused the vae/ download to be deleted when the
# entrypoint symlinked vae/ -> GCS.
