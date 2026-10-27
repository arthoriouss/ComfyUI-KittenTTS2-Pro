# Security

This nodepack does not start a network server or upload reference audio. It uses the locally installed `kittenml` backend.
Model weights are downloaded by the upstream library/Hugging Face when loading with an online model ID.
Only install dependencies from sources you trust; never expose ComfyUI to the public internet without access control.
Issues should not include private voice samples or system credentials.
