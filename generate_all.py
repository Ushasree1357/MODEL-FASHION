"""
HANA KIM — Generation Matrix Builder
Generates all 75 ComfyUI API calls (5 poses × 5 outfits × 3 locations)
Run this script to batch-queue all generations via ComfyUI API.

Usage: python generate_all.py --comfyui-url http://127.0.0.1:8188
"""

import argparse
import json
import random
import requests
import time
import os
from pathlib import Path

# ── Character Identity ─────────────────────────────────────────────────────
TRIGGER_TOKEN = "ea_f_001_hana_kim"
CHARACTER_BASE = (
    "Hana Kim, 28-year-old South Korean Vogue editorial fashion model, "
    "balanced oval face, elegant high cheekbones, feminine soft jawline, "
    "deep brown almond eyes, warm amber iris flecks, beauty mark below left jawline, "
    "dark chestnut mid-back hair, light beige skin, visible pores, natural skin grain, "
    "peach fuzz, poised confidence, calm direct gaze"
)

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, extra limbs, bad anatomy, bad eyes, "
    "disfigured, deformed, plastic skin, over-retouched, CGI, 3D render, illustration, "
    "animation, cartoon, watermark, logo, text overlay, heavy sunglasses, big hats, "
    "hands covering face, strong filters, double face, mutated, airbrushed skin, "
    "mannequin, wax figure, fake skin"
)

# ── Poses ──────────────────────────────────────────────────────────────────
POSES = {
    "P1": {
        "file": "pose_01_standing_front.png",
        "prompt": "standing front, arms relaxed at sides, full body",
    },
    "P2": {
        "file": "pose_02_three_quarter_left.png",
        "prompt": "three-quarter angle left, one hand on hip, weight shifted, full body",
    },
    "P3": {
        "file": "pose_03_walking.png",
        "prompt": "walking stride, natural confident motion, dynamic pose, full body",
    },
    "P4": {
        "file": "pose_04_sitting.png",
        "prompt": "sitting elegantly on a chair, legs crossed, upright posture, three-quarter body",
    },
    "P5": {
        "file": "pose_05_profile_glance.png",
        "prompt": "profile right, looking over shoulder, slight turn, dramatic silhouette, full body",
    },
}

# ── Outfits ────────────────────────────────────────────────────────────────
OUTFITS = {
    "O1": {
        "file": "outfit_01_emerald_halterneck.jpg",
        "prompt": "wearing a pleated emerald green silk halterneck maxi dress, floor-length, elegant drape",
    },
    "O2": {
        "file": "outfit_02_camel_coat.jpg",
        "prompt": "wearing a tailored camel coat over ivory silk blouse and wide-leg camel trousers, understated luxury",
    },
    "O3": {
        "file": "outfit_03_cream_knit.jpg",
        "prompt": "wearing a sleeveless cream ribbed knit midi dress, clean minimalist, small hoop earrings",
    },
    "O4": {
        "file": "outfit_04_black_gown.jpg",
        "prompt": "wearing a black asymmetric one-shoulder evening gown, sophisticated luxury, editorial",
    },
    "O5": {
        "file": "outfit_05_white_poplin.jpg",
        "prompt": "wearing a white organic cotton poplin halterneck maxi dress, fresh daytime energy, breezy",
    },
}

# ── Locations ──────────────────────────────────────────────────────────────
LOCATIONS = {
    "L1": {
        "code": "indoor_studio",
        "prompt": (
            "luxury marble photography studio, soft Rembrandt lighting from left window, "
            "warm directional light, smooth gradient background, professional lighting setup"
        ),
    },
    "L2": {
        "code": "outdoor_garden",
        "prompt": (
            "Parisian garden setting, lush green foliage background, overcast natural daylight, "
            "subtle golden hour edge lighting, natural bokeh background, outdoor luxury campaign"
        ),
    },
    "L3": {
        "code": "indoor_gallery",
        "prompt": (
            "minimalist modern art gallery, white walls, directional track lighting, "
            "clean architectural space, contemporary luxury aesthetic, gallery interior"
        ),
    },
}

# ── Build Full Prompts ─────────────────────────────────────────────────────
def build_prompt(pose_key, outfit_key, location_key):
    pose = POSES[pose_key]
    outfit = OUTFITS[outfit_key]
    location = LOCATIONS[location_key]

    return (
        f"{TRIGGER_TOKEN}, {CHARACTER_BASE}, "
        f"{pose['prompt']}, "
        f"{outfit['prompt']}, "
        f"{location['prompt']}, "
        "luxury editorial fashion photography, cinematic lighting, "
        "shot on Hasselblad H6D, 85mm lens, ultra-detailed skin texture, "
        "photorealistic, 8K resolution, Vogue editorial quality"
    )


# ── ComfyUI API Caller ─────────────────────────────────────────────────────
def queue_comfyui_prompt(comfyui_url: str, workflow_override: dict) -> str:
    """Queue a prompt to ComfyUI API. Returns prompt_id."""
    payload = {"prompt": workflow_override}
    response = requests.post(f"{comfyui_url}/prompt", json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("prompt_id", "unknown")


def get_queue_status(comfyui_url: str) -> dict:
    """Check ComfyUI queue status."""
    response = requests.get(f"{comfyui_url}/queue", timeout=10)
    return response.json()


def load_base_workflow(workflow_path: str) -> dict:
    """Load the base ComfyUI workflow."""
    with open(workflow_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_generation_workflow(
    base_workflow: dict,
    pose_key: str,
    outfit_key: str,
    location_key: str,
    seed: int = None,
) -> dict:
    """
    Modify the base workflow for a specific pose/outfit/location combo.
    Returns a modified copy of the workflow.
    """
    import copy
    wf = copy.deepcopy(base_workflow)

    pose = POSES[pose_key]
    outfit = OUTFITS[outfit_key]
    location_code = LOCATIONS[location_key]["code"]

    if seed is None:
        seed = random.randint(1, 999999999)

    full_prompt = build_prompt(pose_key, outfit_key, location_key)
    filename_prefix = f"HANA/{pose_key}_{outfit_key}_{location_code}"

    # Update nodes
    wf["8"]["inputs"]["image"] = pose["file"]          # Pose reference
    wf["9"]["inputs"]["image"] = outfit["file"]        # Outfit reference
    wf["11"]["inputs"]["text"] = full_prompt           # Positive prompt
    wf["12"]["inputs"]["text"] = NEGATIVE_PROMPT       # Negative prompt

    # Update seeds (use same seed for reproducibility)
    if "17" in wf:
        wf["17"]["inputs"]["seed"] = seed
    if "28" in wf:
        wf["28"]["inputs"]["seed"] = seed

    # Update save paths
    if "32" in wf:
        wf["32"]["inputs"]["filename_prefix"] = filename_prefix + "_facelocked"
    if "24" in wf:
        wf["24"]["inputs"]["filename_prefix"] = filename_prefix
    if "25" in wf:
        wf["25"]["inputs"]["filename_prefix"] = filename_prefix + "_refined"

    return wf, seed


# ── Main Generation Runner ─────────────────────────────────────────────────
def run_generation_matrix(
    comfyui_url: str = "http://127.0.0.1:8188",
    workflow_path: str = None,
    dry_run: bool = False,
    max_queue: int = 5,
    delay_seconds: float = 3.0,
):
    """
    Run the complete 75-combination generation matrix.

    Args:
        comfyui_url: URL to your running ComfyUI instance
        workflow_path: Path to hana_comfyui_workflow.json
        dry_run: If True, only print prompts without sending to ComfyUI
        max_queue: Max items to queue at once (prevents overload)
        delay_seconds: Delay between API calls
    """
    if workflow_path is None:
        # Default to same directory as this script
        script_dir = Path(__file__).parent
        workflow_path = str(script_dir / "hana_comfyui_workflow.json")

    if not dry_run:
        base_workflow = load_base_workflow(workflow_path)
    else:
        base_workflow = {}

    print("=" * 60)
    print("🧬 HANA KIM (EA-F-001) — Generation Matrix")
    print("=" * 60)
    print(f"  Poses    : {len(POSES)}")
    print(f"  Outfits  : {len(OUTFITS)}")
    print(f"  Locations: {len(LOCATIONS)}")
    print(f"  Total    : {len(POSES) * len(OUTFITS) * len(LOCATIONS)} generations")
    print(f"  Mode     : {'DRY RUN (no API calls)' if dry_run else f'LIVE → {comfyui_url}'}")
    print("=" * 60)
    print()

    # Build generation list
    generations = []
    base_seed = 420691337  # Fixed for reproducibility

    for pose_key in POSES:
        for outfit_key in OUTFITS:
            for loc_key in LOCATIONS:
                seed = base_seed + len(generations)
                generations.append({
                    "pose": pose_key,
                    "outfit": outfit_key,
                    "location": loc_key,
                    "seed": seed,
                })

    # Export generation manifest
    manifest_path = Path(workflow_path).parent / "generation_manifest.json"
    manifest = []
    for gen in generations:
        full_prompt = build_prompt(gen["pose"], gen["outfit"], gen["location"])
        loc_code = LOCATIONS[gen["location"]]["code"]
        manifest.append({
            "filename": f"{gen['pose']}_{gen['outfit']}_{loc_code}_facelocked.jpg",
            "pose": gen["pose"],
            "outfit": gen["outfit"],
            "location": gen["location"],
            "seed": gen["seed"],
            "prompt": full_prompt,
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"📋 Generation manifest saved: {manifest_path}")
    print()

    # Run generations
    queued = 0
    errors = 0

    for i, gen in enumerate(generations):
        pose_key = gen["pose"]
        outfit_key = gen["outfit"]
        loc_key = gen["location"]
        seed = gen["seed"]
        loc_code = LOCATIONS[loc_key]["code"]

        label = f"[{i+1:03d}/75] {pose_key} + {outfit_key} + {loc_code}"
        prompt = build_prompt(pose_key, outfit_key, loc_key)

        if dry_run:
            print(f"  📝 {label}")
            print(f"     Prompt: {prompt[:80]}...")
            print()
            continue

        try:
            wf, _ = build_generation_workflow(base_workflow, pose_key, outfit_key, loc_key, seed)

            # Wait if queue is getting full
            while True:
                status = get_queue_status(comfyui_url)
                queue_size = len(status.get("queue_running", [])) + len(status.get("queue_pending", []))
                if queue_size < max_queue:
                    break
                print(f"  ⏳ Queue full ({queue_size} items), waiting...")
                time.sleep(5)

            prompt_id = queue_comfyui_prompt(comfyui_url, wf)
            queued += 1
            print(f"  ✅ {label} → queued (id: {prompt_id[:8]}...)")

        except Exception as e:
            errors += 1
            print(f"  ❌ {label} → ERROR: {e}")

        time.sleep(delay_seconds)

    print()
    print("=" * 60)
    if dry_run:
        print(f"✅ DRY RUN complete. {len(generations)} prompts listed.")
    else:
        print(f"✅ Done! Queued: {queued} | Errors: {errors}")
    print("=" * 60)


# ── CLI Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hana Kim (EA-F-001) ComfyUI Generation Matrix"
    )
    parser.add_argument(
        "--comfyui-url",
        default="http://127.0.0.1:8188",
        help="ComfyUI server URL (default: http://127.0.0.1:8188)",
    )
    parser.add_argument(
        "--workflow",
        default=None,
        help="Path to hana_comfyui_workflow.json (default: same directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all prompts without sending to ComfyUI",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=5,
        help="Max items to have in ComfyUI queue at once (default: 5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls (default: 2.0)",
    )

    args = parser.parse_args()

    run_generation_matrix(
        comfyui_url=args.comfyui_url,
        workflow_path=args.workflow,
        dry_run=args.dry_run,
        max_queue=args.max_queue,
        delay_seconds=args.delay,
    )
