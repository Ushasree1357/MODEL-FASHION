# EA-F-001 HANA KIM — Character Consistency AI System
## A Production-Grade Generative AI Pipeline for Fashion Model Identity Preservation

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FLUX.1-dev](https://img.shields.io/badge/Model-FLUX.1--dev-black?logo=huggingface)](https://huggingface.co/black-forest-labs/FLUX.1-dev)
[![LoRA Rank](https://img.shields.io/badge/LoRA-Rank%2032-purple)](https://arxiv.org/abs/2106.09685)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Author**: Gowni | **Project ID**: EA-F-001 | **Status**: Production-Ready

---

## 📌 Problem Statement

Standard text-to-image models produce a **different face every time**, making them unusable for commercial fashion applications that require brand-consistent, repeatable model identities.

This project solves the problem of **character consistency in AI-generated fashion imagery** using a multi-layer identity locking system. The character "Hana Kim" (EA-F-001) must be identically recognizable across:

- 5 distinct body poses
- 5 different outfits
- 3 environments (indoor studio, outdoor garden, art gallery)
- Variable lighting conditions

**Result**: 75 unique, photorealistic images of the same character with measurable identity preservation.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CHARACTER CONSISTENCY PIPELINE                    │
├─────────────┬────────────────┬──────────────────┬───────────────────┤
│   DATASET   │   TRAINING     │   GENERATION     │   EVALUATION      │
│             │                │                  │                   │
│ Raw Images  │  FLUX.1-dev    │  ComfyUI         │  CLIP Score       │
│    ↓        │      +         │  Workflow:       │  DINOv2 Similarity│
│ Dedup (pHash│  LoRA Rank 32  │  LoRA            │  LPIPS            │
│    ↓        │  (ostris/ai-   │    +             │  Sharpness        │
│ QC Filter   │   toolkit)     │  IPAdapter x2    │  Radar Chart      │
│    ↓        │      +         │    +             │                   │
│ Resize 1024 │  FlowMatch     │  ControlNet      │  📊 Report        │
│    ↓        │  Scheduler     │  (DWPose)        │                   │
│ BLIP-2      │      +         │    +             │                   │
│ Captions    │  Cosine LR     │  4x Upscale      │                   │
│    ↓        │      +         │    ↓             │                   │
│ Metadata    │  W&B + MLflow  │  2048×3072       │                   │
└─────────────┴────────────────┴──────────────────┴───────────────────┘
```

---

## 📦 Repository Structure

```
MODEL/
├── 📓 Notebooks
│   ├── auto_dataset_prep.ipynb          ← Step 1: Dataset cleaning pipeline
│   ├── train_production_pipeline.ipynb  ← Step 2: Full training + tracking
│   └── inference_workflow.ipynb         ← Step 3: Inference & batch generation
│
├── 🐍 Python Modules
│   ├── evaluate_consistency.py          ← Evaluation framework (CLIP+DINO+LPIPS)
│   ├── generate_all.py                  ← Batch 75-combo generator (ComfyUI API)
│   └── experiment_tracking.py          ← W&B + MLflow utilities
│
├── 🖥️ ComfyUI
│   └── hana_comfyui_workflow.json       ← Complete 32-node generation workflow
│
├── 📂 character_id/
│   ├── train/                           ← Training images + captions (.txt pairs)
│   ├── identity_anchors/                ← 10 canonical reference images
│   ├── validation/                      ← 15% holdout for evaluation
│   ├── rejects/                         ← QC-failed images
│   ├── identity.yaml                    ← Character DNA definition
│   ├── master_prompt.txt                ← Full identity prompt
│   └── negative_prompt.txt              ← Negative identity guard
│
├── 📊 reports/                          ← Auto-generated reports and dashboards
└── 📖 README.md                         ← This file
```

---

## 🔬 Technical Architecture Deep-Dive

### Layer 1 — LoRA (Primary Identity Lock)

| Parameter | Value | Rationale |
|---|---|---|
| Base Model | `FLUX.1-dev` | Superior realism, best fashion/skin understanding |
| LoRA Rank | **32** | 2× capacity vs standard 16 — captures micro-details |
| LoRA Alpha | **16** | Effective scale 0.5 — prevents overfitting |
| Noise Scheduler | **FlowMatch** | Flux-native (not DDPM). Critical for quality |
| Optimizer | **AdamW8bit** | Memory-efficient on 24GB VRAM |
| Learning Rate | **4e-4 → 0** | Cosine decay for smooth convergence |
| Training Steps | **3,000** | Optimized for 19–80 image datasets |
| LoRA Modules | **UNET only** | Text encoder NOT trained — prevents drift |

### Layer 2 — IP-Adapter (Face Reinforcement + Outfit Transfer)

| Stream | Purpose | Weight | End % |
|---|---|---|---|
| **Outfit IP-Adapter** | Transfer fabric, color, silhouette from reference | 0.72 | 90% |
| **Face Lock IP-Adapter** | Reinforce identity on top of LoRA | 0.35 | 60% |

### Layer 3 — ControlNet (Pose Precision)

- **Model**: Flux-specific OpenPose ControlNet
- **Preprocessor**: DWPose (133 body keypoints)
- **Strength**: 0.78 | **End**: 85% (releases early for natural detail formation)

---

## 📊 Dataset Preparation

```
Raw Images → Perceptual Hash Dedup → Quality Filter → Smart Resize → BLIP-2 Caption → Metadata
```

### Dataset Requirements

| Tier | Images | Expected Quality |
|---|---|---|
| Minimum | 30 | Usable — some face drift |
| Professional | 60–80 | Good consistency |
| Elite | 100–120 | Near-perfect identity lock |

### Caption Formula

```
{trigger_token}, {character_name}, {age_ethnicity}, {face_anchors}, {eye_anchors},
{hair_anchors}, {skin_texture}, {identity_markers}, {pose}, {outfit}, {location},
{style_keywords}
```

**Example**:
```
ea_f_001_hana_kim, Hana Kim, 28-year-old South Korean fashion model,
balanced oval face, deep brown almond eyes, beauty mark below left jawline,
dark chestnut mid-back hair, light beige skin, visible pores, full body,
wearing emerald green halterneck dress, outdoor garden, luxury editorial
```

---

## 🚀 Training Methodology

### Phase 1: Data Preparation
Run `auto_dataset_prep.ipynb` — 7-step pipeline:
1. Scan and inventory raw images
2. Remove duplicates (perceptual hash with threshold=10)
3. Quality filter (resolution ≥ 512px, aspect ratio ≤ 3:1)
4. Smart resize to 1024px longest edge
5. Auto-caption with BLIP-2 + identity template
6. Build metadata JSON
7. Readiness validation

### Phase 2: Training
Run `train_production_pipeline.ipynb` — 9-stage pipeline:
1. GPU environment validation
2. Dataset health dashboard
3. Auto-captioning (BLIP-2)
4. Train/validation split (85/15)
5. LoRA config generation
6. W&B + MLflow experiment tracking setup
7. Training with live metric capture
8. Training dashboard visualization
9. Best checkpoint selection + export

### Key Hyperparameters

```yaml
lora_rank: 32
lora_alpha: 16
steps: 3000
learning_rate: 4e-4
lr_scheduler: cosine
lr_warmup_steps: 200
optimizer: adamw8bit
gradient_accumulation_steps: 2    # Effective batch = 2
ema_decay: 0.99
noise_scheduler: flowmatch        # Flux-specific!
dtype: bf16
resolutions: [512, 768, 1024]     # Multi-resolution training
train_text_encoder: false         # Critical: keep text encoder frozen
```

---

## 📈 Evaluation Framework

Run `evaluate_consistency.py` to measure:

| Metric | Method | Target Score |
|---|---|---|
| **Identity Preservation** | CLIP ViT-L/14 + DINOv2 face cosine sim | ≥ 0.85 |
| **Prompt Adherence** | CLIP text-image cosine similarity | ≥ 0.27 |
| **Image Quality** | Laplacian variance (sharpness proxy) | ≥ 0.60 |
| **Outfit Variation** | DINOv2 clothing-region diversity index | ≥ 0.25 |
| **Background Variation** | DINOv2 background-region diversity | ≥ 0.30 |

```bash
python evaluate_consistency.py \
  --gen-dir ./generations \
  --ref-dir ./character_id/identity_anchors \
  --prompts ./prompts.json \
  --out-dir ./reports
```

**Output**: JSON report + radar chart visual + per-image identity bar chart.

---

## 🖥️ ComfyUI Workflow

Load `hana_comfyui_workflow.json` in ComfyUI. The 32-node workflow:

```
[Flux.1-dev] → [Hana LoRA 0.88] → [Outfit IPAdapter 0.72]
                                 → [Face Lock IPAdapter 0.35]
                                 → [DWPose ControlNet 0.78]
                                 → [KSampler 28 steps CFG 3.5]
                                 → [VAE Decode]
                                 → [4x UltraSharp Upscale]
                                 → [Save 2048×3072]
```

### Required ComfyUI Custom Nodes
Install via **ComfyUI Manager**:
- `ComfyUI-IPAdapter-plus`
- `comfyui-controlnet-aux` (for DWPose preprocessor)
- `ComfyUI_UltimateSDUpscale`

### Required Model Files
```
ComfyUI/models/
├── unet/flux1-dev.safetensors
├── clip/clip_l.safetensors
├── clip/t5xxl_fp16.safetensors
├── vae/ae.safetensors
├── loras/ea_f_001_hana_kim_lora.safetensors   ← Trained output
├── controlnet/flux-controlnet-openpose.safetensors
└── ipadapter/ip-adapter-flux_dev.safetensors
```

---

## 🔁 Experiment Tracking

### Weights & Biases
```python
import wandb
wandb.init(project='hana-kim-flux-lora', name='rank32_cosine_3k')
wandb.log({'train_loss': loss, 'lr': lr, 'step': step})
```

### MLflow
```bash
mlflow ui --backend-store-uri file://./mlruns
# Open http://127.0.0.1:5000
```

Tracked metrics: `train_loss`, `learning_rate`, `total_training_time_min`, `best_checkpoint_step`

---


## ⚡ Challenges & Solutions

| Challenge | Solution |
|---|---|
| Face drift across different outfits | Dual IP-Adapter: outfit + face streams |
| Identity lost in complex poses | ControlNet end_percent=0.85 (not 1.0) |
| Skin texture degraded at high LoRA strength | Reduced to 0.88; face lock at 0.35 |
| VRAM OOM on L4 (24GB) | Quantized model + gradient checkpointing |
| Over-fitted face (plasticky) | Alpha=16 with Rank=32 → scale=0.5 |
| Caption token drift | shuffle_tokens=False, tag_dropout=0.0 |

---

## 🔮 Future Improvements

1. **Face Swap Post-Processing**: Run InsightFace on generated images to swap in exact face for maximum identity lock
2. **Video Consistency**: Extend pipeline to AnimateDiff for consistent fashion video
3. **Multi-Character Support**: Character registry system for 10+ models
4. **SDXL Fallback**: Secondary pipeline for faster inference on smaller GPUs
5. **Auto-Pose Library**: Generate pose references programmatically with OpenPose skeleton builder
6. **Production API**: FastAPI wrapper for programmatic generation requests

---

## 🛠️ Deployment

See `deployment_guide.md` for full instructions on:
- Local ComfyUI deployment
- RunPod cloud GPU inference
- ComfyUI API integration
- Docker containerization

---

## 📚 References

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) — Black Forest Labs
- [ai-toolkit](https://github.com/ostris/ai-toolkit) — ostris (Flux LoRA trainer)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — comfyanonymous
- [IP-Adapter](https://arxiv.org/abs/2308.06721) — Zhang et al., 2023
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [DINOv2](https://arxiv.org/abs/2304.07193) — Meta AI, 2023
- [CLIP](https://arxiv.org/abs/2103.00020) — OpenAI, 2021

---

*Built with precision for the Generative AI fashion industry. This pipeline represents production-grade AI engineering practices applicable to commercial model identity management.*
