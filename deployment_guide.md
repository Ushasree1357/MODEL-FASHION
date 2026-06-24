# 🚀 Deployment Guide — EA-F-001 Hana Kim Character Consistency System

## Overview

This guide covers deploying the Hana Kim generation pipeline in three environments:
1. **Local** — ComfyUI on Windows/Mac
2. **RunPod** — Cloud GPU inference endpoint
3. **Lightning AI** — Managed Jupyter + GPU

---

## 📍 Option 1: Local ComfyUI Deployment

### Prerequisites
- Windows 10/11 or Ubuntu 20.04+
- NVIDIA GPU with 8GB+ VRAM (16GB+ recommended)
- Python 3.10+

### Step 1: Install ComfyUI
```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### Step 2: Install Custom Nodes
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git
```

### Step 3: Download Required Models

| Model | Destination | Download From |
|---|---|---|
| `flux1-dev.safetensors` | `models/unet/` | HuggingFace FLUX.1-dev |
| `ae.safetensors` | `models/vae/` | HuggingFace FLUX.1-dev |
| `clip_l.safetensors` | `models/clip/` | HuggingFace FLUX.1-dev |
| `t5xxl_fp16.safetensors` | `models/clip/` | HuggingFace FLUX.1-dev |
| `flux-controlnet-openpose.safetensors` | `models/controlnet/` | HuggingFace |
| `ip-adapter-flux_dev.safetensors` | `models/ipadapter/` | HuggingFace |
| `clip_vision_vit_h.safetensors` | `models/clip_vision/` | HuggingFace |
| `4x-UltraSharp.pth` | `models/upscale_models/` | OpenModelDB |
| `ea_f_001_hana_kim_lora.safetensors` | `models/loras/` | **Your trained LoRA** |

```bash
# Download FLUX.1-dev components (requires HF auth)
pip install huggingface_hub
huggingface-cli login

python -c "
from huggingface_hub import hf_hub_download
import shutil, os

files = [
    ('black-forest-labs/FLUX.1-dev', 'flux1-dev.safetensors', 'models/unet/'),
    ('black-forest-labs/FLUX.1-dev', 'ae.safetensors', 'models/vae/'),
    ('comfyanonymous/flux_text_encoders', 'clip_l.safetensors', 'models/clip/'),
    ('comfyanonymous/flux_text_encoders', 't5xxl_fp16.safetensors', 'models/clip/'),
]
for repo, filename, dest in files:
    os.makedirs(dest, exist_ok=True)
    path = hf_hub_download(repo_id=repo, filename=filename)
    shutil.copy(path, os.path.join(dest, filename))
    print(f'Downloaded: {filename}')
"
```

### Step 4: Copy Your LoRA
```bash
# Copy your trained LoRA from training output
copy "ea_f_001_hana_kim_lora.safetensors" "ComfyUI\models\loras\"
```

### Step 5: Launch ComfyUI
```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188

# Low VRAM mode (8–12GB GPU):
python main.py --lowvram --listen 0.0.0.0 --port 8188
```

### Step 6: Load Workflow
1. Open `http://localhost:8188`
2. Click **Load** → Select `hana_comfyui_workflow.json`
3. Adjust pose/outfit reference images in nodes 8 & 9
4. Click **Queue Prompt**

### Step 7: Batch Generation
```bash
# Install requests if needed
pip install requests

# Run batch generator (ComfyUI must be running)
python generate_all.py --comfyui-url http://127.0.0.1:8188 --max-queue 3
```

---

## 📍 Option 2: RunPod Cloud GPU Deployment

### When to Use
- Need A100 80GB for faster generation
- Running batch of 75+ images
- Shareable inference endpoint for team

### Step 1: Setup RunPod Pod
1. Go to [runpod.io](https://runpod.io)
2. **Deploy** → **GPU Cloud**
3. Select template: **RunPod PyTorch 2.1 CUDA 12.1**
4. GPU: **RTX A6000 48GB** (~$0.79/hr) or **A100 80GB** (~$2.49/hr)
5. Disk: **100GB** (FLUX models are large)
6. Click **Deploy**

### Step 2: Install Inside Pod
```bash
# Connect via SSH or RunPod terminal
git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
cd /workspace/ComfyUI
pip install -r requirements.txt

# Install custom nodes
cd custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
cd ..

# Download models (run the hf_hub_download script from Step 3 above)
```

### Step 3: Upload Your Files
```bash
# From your local machine, use RunPod's file upload or rsync:
rsync -avz --progress ea_f_001_hana_kim_lora.safetensors \
  root@<pod-ip>:/workspace/ComfyUI/models/loras/

rsync -avz --progress hana_comfyui_workflow.json \
  root@<pod-ip>:/workspace/

rsync -avz --progress generate_all.py \
  root@<pod-ip>:/workspace/
```

### Step 4: Expose ComfyUI Port
In RunPod dashboard → **Connect** → **HTTP Service** → Port **8188**
Note the public URL (e.g., `https://abc123-8188.proxy.runpod.net`)

```bash
# Launch ComfyUI on RunPod
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 &
```

### Step 5: Run Batch Generation
```bash
# From your local machine:
python generate_all.py \
  --comfyui-url https://abc123-8188.proxy.runpod.net \
  --max-queue 5

# Or run from inside the pod:
python /workspace/generate_all.py \
  --comfyui-url http://127.0.0.1:8188 \
  --max-queue 8
```

### Step 6: Download Results
```bash
# Download generated images from RunPod
rsync -avz root@<pod-ip>:/workspace/ComfyUI/output/HANA/ ./generations/
```

---

## 📍 Option 3: Lightning AI Studio Deployment

Already covered in `lightning_ai_guide.md`.
Quick recap:

```
1. lightning.ai → Create Studio → L4 GPU
2. Upload: character_id.zip + train_production_pipeline.ipynb
3. Run all notebook cells
4. Download LORA from /teamspace/studios/this_studio/DOWNLOAD_ME/
5. Open ComfyUI locally → Load workflow → Generate
```

---

## 🔌 ComfyUI API Integration

For programmatic generation without the web UI:

```python
import requests, json

COMFYUI_URL = 'http://127.0.0.1:8188'

def generate(prompt: str, seed: int = 42) -> str:
    """Generate an image via ComfyUI API. Returns output path."""
    workflow = json.load(open('hana_comfyui_workflow.json'))
    
    # Override prompt and seed
    workflow['11']['inputs']['text'] = prompt
    workflow['17']['inputs']['seed'] = seed
    workflow['28']['inputs']['seed'] = seed
    
    # Queue
    resp = requests.post(f'{COMFYUI_URL}/prompt', json={'prompt': workflow})
    prompt_id = resp.json()['prompt_id']
    
    # Poll for completion
    import time
    while True:
        history = requests.get(f'{COMFYUI_URL}/history/{prompt_id}').json()
        if prompt_id in history:
            outputs = history[prompt_id]['outputs']
            for node_id, output in outputs.items():
                if 'images' in output:
                    img_info = output['images'][0]
                    return f"ComfyUI/output/{img_info['subfolder']}/{img_info['filename']}"
        time.sleep(2)

# Example usage:
result = generate(
    "ea_f_001_hana_kim, Hana Kim, wearing emerald dress, outdoor garden, photorealistic",
    seed=12345
)
print(f"Generated: {result}")
```

---

## 📊 Inference Performance Benchmarks

| GPU | Steps | Time/Image | Cost/75 Images |
|---|---|---|---|
| RTX 3090 (24GB local) | 28 | ~35s | ~44 min |
| L4 (Lightning AI) | 28 | ~25s | ~31 min |
| A10G (Lightning AI) | 28 | ~18s | ~23 min |
| A100 80GB (RunPod) | 28 | ~10s | ~13 min |
| H100 (RunPod) | 28 | ~7s | ~9 min |

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| OOM (out of memory) | Add `--lowvram` flag to ComfyUI, or reduce batch size |
| LoRA not loading | Check filename matches node 11 `lora_name` exactly |
| IPAdapter node not found | Install `ComfyUI-IPAdapter-plus` via ComfyUI Manager |
| DWPose node missing | Install `comfyui_controlnet_aux` and restart ComfyUI |
| Face looks different each time | Increase LoRA strength to 0.90, face IP-Adapter to 0.40 |
| Outfit not transferring | Increase outfit IP-Adapter weight to 0.80 |
| CLIP/T5 not found | Check `models/clip/` has both `clip_l.safetensors` and `t5xxl_fp16.safetensors` |

---

## ✅ Deployment Checklist

```
[ ] LoRA file in models/loras/
[ ] All FLUX model files in correct subdirs
[ ] Custom nodes installed (IPAdapter, ControlNet aux, UltraSharp)
[ ] Workflow JSON loaded in ComfyUI
[ ] Test: Queue single generation, verify output
[ ] Test: Face identity consistent with reference images
[ ] Test: Outfit transfer working (node 9 outfit image)
[ ] Test: Pose control working (node 8 pose image)
[ ] Batch: Run generate_all.py with --dry-run first
[ ] Batch: Run generate_all.py live
[ ] Evaluate: python evaluate_consistency.py
```
