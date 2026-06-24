# ComfyUI / Lightning AI Runbook — EA-F-001 Hana Kim

Goal: provide ComfyUI access and a repeatable Lightning AI training workflow for the `EA-F-001` Hana Kim identity LoRA.

1) Granting access to Gowni
- Option A — Lightning Studio: add Gowni as a collaborator/member of your Studio teamspace and share the project folder (`/teamspace/studios/this_studio`). Upload `character_id.zip` to the teamspace. Gowni can then run notebooks and access GPU.
- Option B — GitHub: invite Gowni as a collaborator to this repository or share a branch. For large datasets prefer sharing a zipped dataset via cloud storage (Drive, S3) and provide the download link in the Studio.

2) Environment & dependencies
- Install repository deps (non-torch) from the top-level `requirements.txt`:

```bash
pip install -r requirements.txt
```

- Install `torch`/`torchvision` with the CUDA wheel matching your GPU. Example for CUDA 12.1 (Lightning Studio L4/A10G):

```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes
```

- The notebook `train_lora_lightning_ai.ipynb` clones `ostris/ai-toolkit` and uses its `requirements.txt`; pinning a commit is recommended.

3) Hugging Face access
- Set `HF_TOKEN` as an environment variable in Lightning Studio (do NOT hardcode):

```bash
export HF_TOKEN="hf_xxx"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
```

4) Dataset
- Recommended: Do NOT keep raw images in repo history. Instead upload `character_id.zip` to the Studio teamspace or a cloud bucket and extract it into `/teamspace/studios/this_studio/character_id`.
- Verified repository path: `character_id/` contains `train/`, `captions/`, `identity_anchors/`.

5) Running training (high level)
- In Lightning Studio, open a blank PyTorch session and run the `train_lora_lightning_ai.ipynb` notebook.
- Steps inside the notebook:
  - Clone ai-toolkit
  - Install requirements (ai-toolkit `requirements.txt`) and torch wheel
  - Upload/unzip dataset to `STUDIO_ROOT` or ensure `character_id` exists there
  - Set `HF_TOKEN` and accept FLUX.1-dev license
  - Write training config and run ai-toolkit trainer

6) ComfyUI & post-training
- After training, download the generated LoRA `.safetensors` and place it under your ComfyUI `models/loras/` folder.
- For inference, add `IPAdapter` + `ControlNet` nodes in ComfyUI and point to the trained LoRA. Use the `master_prompt.txt` as the conditioning guidance.

7) Quick tips for identity consistency (from dataset spec)
- Maintain the `trigger_token` (ea_f_001_hana_kim) in captions.
- Ensure at least 60 images for high quality; 100+ for elite identity-lock.
- Keep anchor shots (identity_anchors) in neutral lighting and angles.

8) Next actions I can perform for you
- Create a Git LFS + history-clean plan to remove images from repo (I will only proceed after confirmation).
- Pin `ai-toolkit` to a tested commit in the notebook and update `requirements.txt` to exact versions.
- Add a minimal `train.sh` or `run.py` wrapper for non-notebook training.
