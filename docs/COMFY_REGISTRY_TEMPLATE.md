# ComfyUI Registry publication checklist

- Create or choose an authorized ComfyUI Registry PublisherId.
- Replace `replace-with-your-comfy-publisher-id` in `pyproject.toml` with the actual PublisherId.
- Provide a real repository URL once the GitHub repository is created and published.
- Validate installation on a clean Windows Portable ComfyUI environment with a matching CUDA PyTorch and `torchaudio`.
- Test `Load Model`, French TTS, 5–30s voice clone, expression, longer audio, memory release, and audio preview **on actual model weights**.
- Confirm compatibility of `transformers` and `kittenml==0.9.7` with your declared supported ComfyUI versions.
- Confirm the weight license and avoid implying that the weights are Apache-2.0.
- Publish a signed GitHub release and attach SHA-256 checksums. Existing GitHub workflow creates a release ZIP after a `v*` tag.
- Update the README badge and changelog only after testing; remove the developer-preview warning after hardware validation.
