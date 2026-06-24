"""
inference_workflow.py — Inference & Batch Generation Notebook
EA-F-001 Hana Kim | ComfyUI API + Diffusers Inference
"""

# This is also delivered as a .ipynb — content below is the Python source
# Run as: python inference_workflow.py or open inference_workflow.ipynb

NOTEBOOK_CELLS = """
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# 🧬 Hana Kim (EA-F-001) — Inference & Generation Workflow\\n",
        "**Post-Training**: Load your trained LoRA and generate all 75 combinations\\n\\n",
        "## Two Modes:\\n",
        "- **Mode A**: Quick single-image test with diffusers (no ComfyUI needed)\\n",
        "- **Mode B**: Full batch generation via ComfyUI API (all 75 combos)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ── SETUP ─────────────────────────────────────────────────────\\n",
        "!pip install -q diffusers transformers accelerate safetensors\\n",
        "import torch, os\\n",
        "from pathlib import Path\\n",
        "\\n",
        "STUDIO_ROOT = os.getcwd()\\n",
        "LORA_PATH   = f'{STUDIO_ROOT}/output/ea_f_001_hana_kim_lora_v3/ea_f_001_hana_kim_lora_v3.safetensors'\\n",
        "OUTPUT_DIR  = f'{STUDIO_ROOT}/inference_output'\\n",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)\\n",
        "\\n",
        "print(f'LoRA path exists: {Path(LORA_PATH).exists()}')\\n",
        "print(f'Output dir: {OUTPUT_DIR}')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ── MODE A: Quick Diffusers Test ──────────────────────────────\\n",
        "from diffusers import FluxPipeline\\n",
        "import torch\\n",
        "\\n",
        "print('Loading FLUX.1-dev...')\\n",
        "pipe = FluxPipeline.from_pretrained(\\n",
        "    'black-forest-labs/FLUX.1-dev',\\n",
        "    torch_dtype=torch.bfloat16,\\n",
        ").to('cuda')\\n",
        "\\n",
        "print('Loading Hana Kim LoRA...')\\n",
        "pipe.load_lora_weights(LORA_PATH, adapter_name='hana_kim')\\n",
        "pipe.set_adapters(['hana_kim'], adapter_weights=[0.88])\\n",
        "\\n",
        "# Test prompt\\n",
        "PROMPT = (\\n",
        "    'ea_f_001_hana_kim, Hana Kim, 28-year-old South Korean Vogue editorial model, '\\n",
        "    'balanced oval face, beauty mark below left jawline, deep brown almond eyes, '\\n",
        "    'wearing tailored camel coat over ivory silk blouse, outdoor Parisian garden, '\\n",
        "    'overcast natural daylight, 3/4 angle, 105mm lens, photorealistic'\\n",
        ")\\n",
        "NEGATIVE = 'blurry, low quality, plastic skin, CGI, bad anatomy, distorted'\\n",
        "\\n",
        "image = pipe(\\n",
        "    prompt=PROMPT,\\n",
        "    height=1024, width=1024,\\n",
        "    num_inference_steps=28,\\n",
        "    guidance_scale=3.5,\\n",
        "    generator=torch.Generator('cuda').manual_seed(420691337),\\n",
        ").images[0]\\n",
        "\\n",
        "out_path = f'{OUTPUT_DIR}/test_camel_coat_outdoor.png'\\n",
        "image.save(out_path)\\n",
        "from IPython.display import Image as IPImage, display\\n",
        "display(IPImage(out_path, width=512))\\n",
        "print(f'✅ Saved: {out_path}')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ── MODE B: Full 75-Combination Matrix ────────────────────────\\n",
        "# Uses diffusers (no ComfyUI needed on Lightning AI)\\n",
        "from diffusers import FluxPipeline\\n",
        "import torch, os, time\\n",
        "from pathlib import Path\\n",
        "from IPython.display import Image as IPImage, display\\n",
        "\\n",
        "POSES = {\\n",
        "    'P1': 'standing front, arms relaxed at sides, full body, direct camera gaze',\\n",
        "    'P2': 'three-quarter angle left, one hand on hip, confident pose, full body',\\n",
        "    'P3': 'walking stride, natural dynamic motion, full body, editorial energy',\\n",
        "    'P4': 'sitting elegantly, legs crossed, upright, three-quarter body',\\n",
        "    'P5': 'profile right, looking over shoulder, dramatic silhouette, full body',\\n",
        "}\\n",
        "OUTFITS = {\\n",
        "    'O1': 'wearing a pleated emerald green silk halterneck maxi dress, floor-length, luxury drape',\\n",
        "    'O2': 'wearing a tailored camel coat over ivory silk blouse and wide-leg trousers',\\n",
        "    'O3': 'wearing a sleeveless cream ribbed knit midi dress, minimalist, small hoop earrings',\\n",
        "    'O4': 'wearing a black asymmetric one-shoulder evening gown, editorial luxury',\\n",
        "    'O5': 'wearing a white organic cotton poplin halterneck maxi dress, fresh daytime',\\n",
        "}\\n",
        "LOCATIONS = {\\n",
        "    'L1': 'luxury marble photography studio, soft Rembrandt lighting, warm directional light',\\n",
        "    'L2': 'Parisian garden, lush foliage, overcast daylight, golden hour edge light',\\n",
        "    'L3': 'minimalist art gallery, white walls, directional track lighting',\\n",
        "}\\n",
        "\\n",
        "CHAR_IDENTITY = (\\n",
        "    'ea_f_001_hana_kim, Hana Kim, 28-year-old South Korean Vogue editorial model, '\\n",
        "    'balanced oval face, elegant high cheekbones, beauty mark below left jawline, '\\n",
        "    'deep brown almond eyes, dark chestnut mid-back hair, light beige skin, '\\n",
        "    'natural skin grain, visible pores'\\n",
        ")\\n",
        "STYLE_SUFFIX = 'luxury editorial fashion photography, shot on Hasselblad, photorealistic, 8K'\\n",
        "\\n",
        "total = len(POSES) * len(OUTFITS) * len(LOCATIONS)\\n",
        "print(f'Generating {total} images (5 poses × 5 outfits × 3 locations)...')\\n",
        "\\n",
        "gen_dir = Path(f'{STUDIO_ROOT}/inference_output/all_75')\\n",
        "gen_dir.mkdir(parents=True, exist_ok=True)\\n",
        "\\n",
        "idx, errors = 0, 0\\n",
        "base_seed = 420691337\\n",
        "\\n",
        "for p_key, pose in POSES.items():\\n",
        "    for o_key, outfit in OUTFITS.items():\\n",
        "        for l_key, location in LOCATIONS.items():\\n",
        "            idx += 1\\n",
        "            seed = base_seed + idx\\n",
        "            filename = gen_dir / f'{p_key}_{o_key}_{l_key}.png'\\n",
        "\\n",
        "            if filename.exists():\\n",
        "                print(f'  [{idx:02d}/75] ⏭️  Skip (exists): {filename.name}')\\n",
        "                continue\\n",
        "\\n",
        "            prompt = f'{CHAR_IDENTITY}, {pose}, {outfit}, {location}, {STYLE_SUFFIX}'\\n",
        "\\n",
        "            try:\\n",
        "                t0 = time.time()\\n",
        "                image = pipe(\\n",
        "                    prompt=prompt,\\n",
        "                    height=1024, width=1024,\\n",
        "                    num_inference_steps=28,\\n",
        "                    guidance_scale=3.5,\\n",
        "                    generator=torch.Generator('cuda').manual_seed(seed),\\n",
        "                ).images[0]\\n",
        "                image.save(filename, quality=95)\\n",
        "                elapsed = time.time() - t0\\n",
        "                print(f'  [{idx:02d}/75] ✅ {filename.name} ({elapsed:.1f}s)')\\n",
        "                if idx % 10 == 0:\\n",
        "                    display(IPImage(str(filename), width=400))\\n",
        "            except Exception as e:\\n",
        "                print(f'  [{idx:02d}/75] ❌ {p_key}+{o_key}+{l_key}: {e}')\\n",
        "                errors += 1\\n",
        "\\n",
        "print(f'\\\\n✅ Done! Generated: {idx-errors} | Errors: {errors}')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ── STEP: Select Best Images & Create Presentation Grid ───────\\n",
        "from PIL import Image\\n",
        "import glob\\n",
        "from pathlib import Path\\n",
        "\\n",
        "gen_dir = Path(f'{STUDIO_ROOT}/inference_output/all_75')\\n",
        "all_imgs = sorted(glob.glob(str(gen_dir / '*.png')))\\n",
        "\\n",
        "# Build a 5×5 grid of best images (one per outfit across poses)\\n",
        "grid_size = 5\\n",
        "thumb_size = 400\\n",
        "grid_img = Image.new('RGB', (thumb_size * grid_size, thumb_size * grid_size), (20, 20, 20))\\n",
        "\\n",
        "# One from each outfit (5) × one from each pose (5)\\n",
        "grid_paths = [p for p in all_imgs if 'L1' in p][:25]  # Indoor studio shots\\n",
        "\\n",
        "for i, img_path in enumerate(grid_paths[:25]):\\n",
        "    row = i // grid_size\\n",
        "    col = i %  grid_size\\n",
        "    try:\\n",
        "        with Image.open(img_path) as img:\\n",
        "            thumb = img.resize((thumb_size, thumb_size), Image.LANCZOS)\\n",
        "            grid_img.paste(thumb, (col * thumb_size, row * thumb_size))\\n",
        "    except Exception:\\n",
        "        pass\\n",
        "\\n",
        "grid_path = f'{STUDIO_ROOT}/DOWNLOAD_ME/hana_generation_grid_5x5.jpg'\\n",
        "Path(grid_path).parent.mkdir(exist_ok=True)\\n",
        "grid_img.save(grid_path, quality=90)\\n",
        "\\n",
        "from IPython.display import Image as IPImage, display\\n",
        "display(IPImage(grid_path, width=800))\\n",
        "print(f'✅ 5×5 presentation grid saved: {grid_path}')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ── STEP: Run Evaluation on Generated Images ──────────────────\\n",
        "import subprocess, sys\\n",
        "\\n",
        "GEN_DIR = f'{STUDIO_ROOT}/inference_output/all_75'\\n",
        "REF_DIR = f'{STUDIO_ROOT}/character_id/identity_anchors'\\n",
        "OUT_DIR = f'{STUDIO_ROOT}/reports'\\n",
        "EVAL_SCRIPT = f'{STUDIO_ROOT}/evaluate_consistency.py'\\n",
        "\\n",
        "# Run evaluation framework\\n",
        "result = subprocess.run(\\n",
        "    [sys.executable, EVAL_SCRIPT,\\n",
        "     '--gen-dir', GEN_DIR,\\n",
        "     '--ref-dir', REF_DIR,\\n",
        "     '--out-dir', OUT_DIR],\\n",
        "    capture_output=True, text=True\\n",
        ")\\n",
        "print(result.stdout)\\n",
        "if result.returncode != 0:\\n",
        "    print('STDERR:', result.stderr[:500])\\n",
        "else:\\n",
        "    print('✅ Evaluation complete! Check reports/evaluation_report.png')"
      ]
    }
  ]
}
"""

# Write as proper notebook
import json

cells_raw = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧬 Hana Kim (EA-F-001) — Inference & Generation Workflow\n",
            "**Post-Training**: Load your trained LoRA and generate all 75 combinations\n\n",
            "## Two Modes:\n",
            "- **Mode A**: Quick single-image test with diffusers (no ComfyUI needed)\n",
            "- **Mode B**: Full batch generation (all 75 combos)\n",
            "- **Mode C**: Evaluation report generation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q diffusers transformers accelerate safetensors Pillow\n",
            "import torch, os\n",
            "from pathlib import Path\n",
            "STUDIO_ROOT = os.getcwd()\n",
            "LORA_PATH   = f'{STUDIO_ROOT}/output/ea_f_001_hana_kim_lora_v3/ea_f_001_hana_kim_lora_v3.safetensors'\n",
            "OUTPUT_DIR  = f'{STUDIO_ROOT}/inference_output'\n",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
            "print(f'LoRA exists: {Path(LORA_PATH).exists()}')\n",
            "print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# MODE A: Quick Test\n",
            "from diffusers import FluxPipeline\n",
            "import torch\n",
            "\n",
            "pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16).to('cuda')\n",
            "pipe.load_lora_weights(LORA_PATH, adapter_name='hana_kim')\n",
            "pipe.set_adapters(['hana_kim'], adapter_weights=[0.88])\n",
            "\n",
            "PROMPT = ('ea_f_001_hana_kim, Hana Kim, 28-year-old South Korean fashion model, '\n",
            "          'beauty mark below left jawline, deep brown almond eyes, '\n",
            "          'wearing tailored camel coat, outdoor Parisian garden, 3/4 angle, photorealistic')\n",
            "\n",
            "image = pipe(prompt=PROMPT, height=1024, width=1024, num_inference_steps=28,\n",
            "             guidance_scale=3.5, generator=torch.Generator('cuda').manual_seed(420691337)).images[0]\n",
            "image.save(f'{OUTPUT_DIR}/mode_a_test.png', quality=95)\n",
            "from IPython.display import Image as IPImage, display\n",
            "display(IPImage(f'{OUTPUT_DIR}/mode_a_test.png', width=512))\n",
            "print('✅ Test image saved!')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# MODE B: Full 75 Combinations\n",
            "import time\n",
            "POSES = {'P1': 'standing front, arms relaxed, full body', 'P2': '3/4 angle left, hand on hip, full body',\n",
            "         'P3': 'walking stride, dynamic, full body', 'P4': 'sitting elegantly, legs crossed',\n",
            "         'P5': 'profile right, glancing over shoulder'}\n",
            "OUTFITS = {'O1': 'emerald green silk halterneck maxi dress', 'O2': 'tailored camel coat over ivory blouse',\n",
            "           'O3': 'cream ribbed knit midi dress', 'O4': 'black asymmetric evening gown',\n",
            "           'O5': 'white poplin halterneck maxi dress'}\n",
            "LOCATIONS = {'L1': 'luxury marble studio, Rembrandt lighting', 'L2': 'Parisian garden, overcast daylight',\n",
            "             'L3': 'minimalist art gallery, track lighting'}\n",
            "BASE = ('ea_f_001_hana_kim, Hana Kim, 28-year-old South Korean fashion model, '\n",
            "        'beauty mark below left jawline, deep brown almond eyes, light beige skin, natural pores')\n",
            "\n",
            "gen_dir = Path(f'{OUTPUT_DIR}/all_75')\n",
            "gen_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "idx, errors = 0, 0\n",
            "for pk, pose in POSES.items():\n",
            "    for ok, outfit in OUTFITS.items():\n",
            "        for lk, location in LOCATIONS.items():\n",
            "            idx += 1\n",
            "            fname = gen_dir / f'{pk}_{ok}_{lk}.png'\n",
            "            if fname.exists(): print(f'[{idx:02d}/75] ⏭️  {fname.name}'); continue\n",
            "            prompt = f'{BASE}, {pose}, wearing {outfit}, {location}, luxury editorial, photorealistic 8K'\n",
            "            try:\n",
            "                t0 = time.time()\n",
            "                img = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=28,\n",
            "                           guidance_scale=3.5, generator=torch.Generator('cuda').manual_seed(420691337+idx)).images[0]\n",
            "                img.save(fname, quality=95)\n",
            "                print(f'[{idx:02d}/75] ✅ {fname.name} ({time.time()-t0:.1f}s)')\n",
            "            except Exception as e:\n",
            "                print(f'[{idx:02d}/75] ❌ {e}'); errors += 1\n",
            "print(f'\\n✅ Generated {idx-errors} images | Errors: {errors}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# MODE C: 5x5 Presentation Grid\n",
            "from PIL import Image\n",
            "import glob\n",
            "gen_dir = Path(f'{OUTPUT_DIR}/all_75')\n",
            "all_imgs = sorted(glob.glob(str(gen_dir / '*.png')))\n",
            "\n",
            "THUMB = 400; COLS = 5\n",
            "rows = (len(all_imgs) + COLS - 1) // COLS\n",
            "grid = Image.new('RGB', (THUMB * COLS, THUMB * rows), (15, 15, 15))\n",
            "\n",
            "for i, p in enumerate(all_imgs[:25]):\n",
            "    with Image.open(p) as img:\n",
            "        thumb = img.resize((THUMB, THUMB), Image.LANCZOS)\n",
            "        grid.paste(thumb, ((i % COLS) * THUMB, (i // COLS) * THUMB))\n",
            "\n",
            "grid_path = f'{STUDIO_ROOT}/DOWNLOAD_ME/hana_5x5_grid.jpg'\n",
            "Path(grid_path).parent.mkdir(exist_ok=True)\n",
            "grid.save(grid_path, quality=90)\n",
            "from IPython.display import Image as IPImage, display\n",
            "display(IPImage(grid_path, width=900))\n",
            "print(f'✅ Grid saved: {grid_path}')"
        ]
    }
]

notebook_json = {
    "cells": cells_raw,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

print(json.dumps(notebook_json, indent=2)[:200])
print("... (notebook structure defined)")
