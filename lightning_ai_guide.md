# 🧬 HANA KIM (EA-F-001) — Complete Character Identity System
## Technical Mastery Guide for Shahida's Submission
*Prepared by: Gowni | Lighting AI Training System | June 2026*

---

## 🎯 Executive Summary

This document covers the **complete technical pipeline** for creating a reusable AI fashion model character using:

| Technology | Purpose | Tool Used |
|---|---|---|
| **Flux LoRA Training** | Lock Hana's face identity into the model | ostris/ai-toolkit |
| **ComfyUI** | Full generation workflow UI | ComfyUI + custom nodes |
| **IPAdapter** | Transfer outfit appearance | IPAdapter + CLIP Vision |
| **ControlNet** | Control body pose precisely | OpenPose ControlNet for Flux |
| **RunPod / Lightning AI** | Cloud GPU training | L4 / A10G / A100 |
| **Flux.1-dev** | Base image generation model | Black Forest Labs |

---

## 🔬 PART 1: Character Consistency — How It Works

### The Problem We're Solving
Standard image generation produces a **different face every time**, even with detailed prompts. To use Hana Kim consistently across:
- 5 poses × 5 outfits × 3 locations = **75 unique generations**

…we need **identity locking** — the model must remember Hana's exact face DNA.

### The 3-Layer Identity Lock System

```
Layer 1: LoRA (Primary)
  └── Trained on 19+ images of Hana
  └── Burns her face into the model weights
  └── Activated by trigger token: ea_f_001_hana_kim
  └── Weight: 0.85–0.90 at inference

Layer 2: IP-Adapter (Secondary Face Lock)
  └── Provides a reference image of Hana's face
  └── Guides generation toward that face even without LoRA
  └── Weight: 0.30–0.40 (subtle, doesn't override LoRA)
  └── Combined with LoRA = near-perfect face replication

Layer 3: Detailed Text Prompt (Tertiary)
  └── Describes every facial feature in precise language
  └── beauty mark below left jawline, deep brown almond eyes, etc.
  └── Prevents model drift in complex scenes
```

### Why Flux + LoRA is Best for Fashion Models

- **Flux.1-dev** has superior understanding of clothing textures, human anatomy, and lighting
- **LoRA rank 32** captures fine facial details (iris texture, skin grain, beauty marks)
- **FlowMatch scheduler** (Flux-specific) produces sharper, more realistic skin
- Better than SDXL for: skin translucency, hair strand detail, fabric drape

---

## 🎓 PART 2: LoRA Training — Deep Technical Explanation

### What is LoRA?
**Low-Rank Adaptation** modifies a small set of model weights (instead of all 12B parameters) to specialize the model for a specific concept. For face identity:

```
Original Flux weights: 12 Billion parameters (frozen)
          +
LoRA Delta weights:    ~100 Million parameters (trained)
          =
Result: Flux that "knows" Hana Kim's face
```

### Our Training Configuration (Elite Settings)

```yaml
network:
  type: lora
  linear: 32        # Rank 32 = 2x more capacity than standard rank 16
  linear_alpha: 16  # Effective scale = alpha/rank = 0.5 (prevents overfit)
```

**Why Rank 32?** 
- Rank 16 captures general face shapes but misses fine details (beauty mark, iris flecks, brow asymmetry)
- Rank 32 captures **all the micro-details** that make Hana unique
- Alpha 16 with Rank 32 = conservative effective LR, prevents overfitting the face

### Learning Rate Strategy
```yaml
learning_rate: 4e-4     # Aggressive enough to learn quickly
lr_scheduler: cosine    # Starts high, decays smoothly to 0
lr_warmup_steps: 200    # Gentle warmup prevents early instability
```

**Cosine Decay Visualization:**
```
LR: 4e-4 ─────────╮
                    ╮
                     ╮
                      ╮
                       ╮─── final: ~0
Step:  0    500   1000  1500  2000  2500  3000
```

### Training Priority Weights (Caption Strategy)

Our captions are structured to match the training priority:

```
40% face identity  → trigger + face descriptors in EVERY caption
20% camera angles  → "front-facing", "3/4 left", "profile right"
15% expressions    → "calm direct gaze", "subtle smile"
10% lighting       → "soft Rembrandt lighting", "overcast daylight"
10% body coverage  → "full body", "waist up", "closeup"
5%  environment    → "indoor studio", "outdoor garden"
```

---

## 🖥️ PART 3: ComfyUI Workflow — Node-by-Node Explanation

### Workflow Architecture

```
[FLUX.1-dev]──────────────────────────────────────────────┐
      │                                                     │
      ↓                                                     │
[LoRA Loader] ← ea_f_001_hana_kim_lora.safetensors        │
      │         strength: 0.88                             │
      ↓                                                     │
[IP-Adapter (Outfit)] ← outfit_reference.jpg               │
      │  weight: 0.72 | style: linear                     │
      ↓                                                     │
[IP-Adapter (Face Lock)] ← hana_face_anchor.jpg            │
      │  weight: 0.35 | style: style transfer             │
      ↓                                                     │
[ControlNet Apply] ← DWPose extracted from pose_XX.png     │
      │  strength: 0.78 | start: 0.0 | end: 0.85         │
      ↓                                                     │
[KSampler]                                                  │
      │  steps: 28 | cfg: 3.5 | sampler: euler            │
      │  scheduler: simple | denoise: 1.0                  │
      ↓                                                     │
[VAE Decode]                                               │
      ↓                                                     │
[4x UltraSharp Upscale] → 4096×6144 resolution             │
      ↓                                                     │
[Save Image: HANA/pose{N}_outfit{N}_{location}]            │
```

### Critical Node Settings Explained

| Node | Setting | Value | Reason |
|---|---|---|---|
| LoRA Loader | strength_model | 0.88 | High enough for identity lock |
| LoRA Loader | strength_clip | 0.88 | Matches model strength |
| IP-Adapter Outfit | weight | 0.72 | Strong outfit transfer without bleeding |
| IP-Adapter Face | weight | 0.35 | Subtle — LoRA does heavy lifting |
| ControlNet | strength | 0.78 | Strong pose control |
| ControlNet | end_percent | 0.85 | Release at 85% so fine details can form |
| KSampler | cfg | 3.5 | Flux optimal — higher = over-saturated |
| KSampler | steps | 28 | 28 is Flux sweet spot (more = diminishing returns) |

---

## 👗 PART 4: IP-Adapter — Outfit Transfer Explained

### How IP-Adapter Works
IP-Adapter uses a **CLIP Vision encoder** to convert an outfit reference image into an "embedding" that guides generation:

```
Outfit Photo → CLIP Vision Encoder → 257 embedding tokens
                                           ↓
                                    Injected into attention layers
                                           ↓
                                    Model generates with outfit features
```

### The Two-Stream Strategy (Our Workflow)

```
Stream 1: Outfit IP-Adapter (weight 0.72, end at 0.90)
  Purpose: Transfer fabric texture, color, silhouette from reference
  
Stream 2: Face Lock IP-Adapter (weight 0.35, end at 0.60)
  Purpose: Reinforce Hana's face identity alongside the LoRA
  
Both streams combine additively in attention:
  Total influence = LoRA identity + Outfit style + Face reference
```

### IP-Adapter Weight Guide

```
Outfit IP-Adapter weight:
  0.50–0.60 = Subtle suggestion of outfit (prompt dominates)
  0.65–0.75 = Good outfit transfer (RECOMMENDED)
  0.80–0.90 = Strong outfit (may bleed into face/background)

Face IP-Adapter weight:
  0.25–0.35 = Subtle identity reinforcement (RECOMMENDED with LoRA)
  0.40–0.50 = Stronger face push (use if LoRA results drift)
  0.55–0.70 = Very strong (use WITHOUT LoRA as standalone)
```

---

## 🎭 PART 5: ControlNet (Pose Control) — Explained

### DWPose Preprocessing Pipeline
```
Input: Any pose reference image (your 5 poses)
       ↓
DWPreprocessor → Detects 133 body keypoints
       ↓
Skeleton overlay image (colored lines = body structure)
       ↓
ControlNet reads skeleton → Forces model to match that pose
```

### The 5 Poses for Hana

| Code | Pose | Best For |
|---|---|---|
| P1 | Standing front, arms relaxed | Product-style full-body shots |
| P2 | 3/4 turn left, hand on hip | Editorial confidence shots |
| P3 | Walking stride, natural motion | Dynamic campaign images |
| P4 | Sitting elegantly, legs crossed | Indoor luxury shots |
| P5 | Profile right, glancing back | Dramatic editorial, outerwear |

### ControlNet Timing (start/end_percent)

```
0.0 → 0.85 means:
  - Pose enforced from step 0 (beginning of generation)
  - Released at step 85% (allows final details to form naturally)
  
If end_percent = 1.0: Pose too rigid, skin looks rubbery
If end_percent = 0.70: Pose may drift in complex scenes
0.85 is the sweet spot ✓
```

---

## ☁️ PART 6: RunPod — Alternative GPU Training

### Why RunPod?
- **Cheaper A100 access** than Lightning AI for long runs
- **Persistent storage** — your files don't vanish between sessions
- **More GPU options**: A100 80GB, H100, RTX 4090

### RunPod Setup Steps

1. Go to [runpod.io](https://runpod.io) → Create Account
2. Deploy **Template**: `RunPod Pytorch 2.1 / CUDA 12.1`
3. Select GPU: **A100 SXM 80GB** or **RTX A6000** (best price/performance)
4. Volume: **50GB network volume** (persistent storage)
5. SSH into pod:

```bash
# Inside RunPod terminal:
git clone --recursive https://github.com/ostris/ai-toolkit.git /workspace/ai-toolkit
cd /workspace/ai-toolkit
pip install -r requirements.txt
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

6. Upload your `character_id.zip` via RunPod file manager
7. Run training:
```bash
python run.py config/hana_kim_elite_config.yaml
```

### Cost Comparison

| GPU | Cost/hr | Training Time | Total Cost |
|---|---|---|---|
| L4 24GB (Lightning AI) | ~$0.42/hr | ~75 min | ~$0.52 |
| A10G 24GB (Lightning AI) | ~$0.75/hr | ~50 min | ~$0.62 |
| RTX A6000 (RunPod) | ~$0.79/hr | ~45 min | ~$0.59 |
| A100 80GB (RunPod) | ~$2.49/hr | ~20 min | ~$0.83 |

**Recommendation: L4 on Lightning AI** for best cost-efficiency for 3,000 steps.

---

## 🗂️ PART 7: Generation Matrix — 75 Combinations

### Naming Convention
Format: `pose{P}_outfit{O}_{location}.jpg`

```
P = 1-5 (pose number)
O = 1-5 (outfit number)
L = indoor_studio | outdoor_garden | indoor_gallery
```

### The 75 Generations Planned

| | Outfit 1 (Emerald Halterneck) | Outfit 2 (Camel Coat) | Outfit 3 (Cream Knit) | Outfit 4 (Black Gown) | Outfit 5 (White Poplin) |
|---|---|---|---|---|---|
| **P1 Front** | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 |
| **P2 3/4 Left** | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 |
| **P3 Walking** | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 |
| **P4 Sitting** | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 |
| **P5 Profile** | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 | L1, L2, L3 |

**L1** = Indoor Luxury Studio | **L2** = Outdoor Parisian Garden | **L3** = Indoor Art Gallery

### Prompt Templates for Each Location

**L1 — Indoor Luxury Studio:**
```
ea_f_001_hana_kim, Hana Kim, 28-year-old East Asian Korean woman,
[POSE], [OUTFIT], luxury marble photography studio, soft Rembrandt lighting
from left window, warm white light, smooth gradient background, 
editorial fashion photography, Hasselblad quality, 8K
```

**L2 — Outdoor Garden:**
```
ea_f_001_hana_kim, Hana Kim, 28-year-old East Asian Korean woman,
[POSE], [OUTFIT], Parisian garden, lush green foliage, overcast daylight,
golden hour edge lighting, natural bokeh, cinematic fashion editorial,
outdoor luxury campaign
```

**L3 — Indoor Art Gallery:**
```
ea_f_001_hana_kim, Hana Kim, 28-year-old East Asian Korean woman,
[POSE], [OUTFIT], minimalist art gallery, white walls, directional
track lighting, clean architectural background, modern luxury aesthetic,
editorial fashion, Vogue-quality lighting
```

---

## ✅ PART 8: Reusable Workflow Design

### How to Reuse This System for Any Model

1. **Create new character folder**: Copy `character_id/` structure
2. **Replace training images**: Swap images in `train/`
3. **Update captions**: Change trigger token and identity descriptors
4. **Update `identity.yaml`**: New character DNA
5. **Retrain LoRA**: Run notebook with new config
6. **Update ComfyUI workflow**: Change LoRA name and face anchor image
7. **Generate**: Same workflow, new model in minutes

### Files to Change Per Character

| File | What to Change |
|---|---|
| `identity.yaml` | Name, age, ethnicity, physical features |
| `master_prompt.txt` | Full detailed identity description |
| `train/*.txt` captions | Trigger token + image-specific details |
| `config.json` | LoRA name, trigger token |
| `hana_kim_elite_config.yaml` | trigger_word, name fields |
| ComfyUI node 11 | Positive prompt with new identity |
| ComfyUI node 26 | New face anchor reference image |

---

## 📋 2-Day Delivery Checklist

### Day 1 (Today)
- [x] Dataset prepared (19 images + captions in `train/`)
- [x] ComfyUI workflow built (`hana_comfyui_workflow.json`)
- [x] Training notebook ready (`train_lora_lightning_ai.ipynb`)
- [ ] **Upload to Lightning AI** and start training
- [ ] Monitor validation samples at step 250, 500, 750...
- [ ] Download final LoRA (`ea_f_001_hana_kim_lora.safetensors`)

### Day 2 (Tomorrow)
- [ ] Load LoRA into ComfyUI
- [ ] Generate all 75 combinations (5×5×3)
- [ ] Curate best results (select top 25 for presentation)
- [ ] Create before/after comparison: "base model vs trained LoRA"
- [ ] Package deliverable: LoRA file + ComfyUI workflow + generation guide
- [ ] Submit to Shahida

### Deliverable Package Structure
```
HANA_KIM_EA_F_001_DELIVERABLE/
├── 📦 ea_f_001_hana_kim_lora.safetensors  ← The trained model
├── 🖥️  hana_comfyui_workflow.json          ← Drop into ComfyUI
├── 📖 technical_guide.md                  ← This document
├── 🖼️  generations/
│   ├── pose01_outfit01_indoor_studio.jpg
│   ├── pose01_outfit01_outdoor_garden.jpg
│   ├── ... (75 total images)
├── 📊 identity.yaml                       ← Character DNA file
└── 📝 generation_prompts.txt              ← All 75 prompts used
```

---

## 🚀 Quick Start: 5-Minute Setup

```bash
# Step 1: Zip your dataset
# Right-click character_id folder → Compress → character_id.zip

# Step 2: Go to lightning.ai → Create Studio → Blank PyTorch
# Set hardware to: L4 GPU

# Step 3: Upload files via sidebar:
#   - character_id.zip
#   - train_lora_lightning_ai.ipynb

# Step 4: Open notebook, run cells 1-7

# Step 5: Wait ~60-75 minutes

# Step 6: Download ea_f_001_hana_kim_lora.safetensors

# Step 7: Open ComfyUI, load hana_comfyui_workflow.json, generate!
```

---

*This system was designed to be a portfolio-quality demonstration of end-to-end AI fashion model generation. The LoRA, once trained, is fully reusable across any ComfyUI workflow and can generate unlimited consistent images of Hana Kim.*
