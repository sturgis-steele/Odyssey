# Models Directory

This folder contains large GGUF model files for the llama.cpp backend.
These files are **not** stored in Git (see .gitignore).

## Currently Used Models

**Primary Model (Instruct)**
- Filename: `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`
- Repository: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF
- Quantization: Q4_K_M (~731 MB)
- Purpose: General conversation and voice assistant

**Secondary Model (Tool)**
- Filename: `LFM2-1.2B-Tool-Q4_K_M.gguf`
- Repository: https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF
- Quantization: Q4_K_M (~731 MB)
- Purpose: Future tool-calling features

## How to Re-download These Models

```bash
cd ~/odyssey/models

# Primary model
python -c '
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
    filename="LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
    local_dir=".",
    local_dir_use_symlinks=False
)
'

# Tool model (optional)
python -c '
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="LiquidAI/LFM2-1.2B-Tool-GGUF",
    filename="LFM2-1.2B-Tool-Q4_K_M.gguf",
    local_dir=".",
    local_dir_use_symlinks=False
)
