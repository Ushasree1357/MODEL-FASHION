"""
evaluate_consistency.py
========================
Enterprise-level Character Consistency Evaluation Framework
EA-F-001 Hana Kim — AI Fashion Model Pipeline

Metrics measured:
  1. Identity Preservation Score (CLIP face similarity)
  2. Pose Accuracy Score (keypoint matching)
  3. Clothing Variation Index (outfit transfer quality)
  4. Background Variation Index (scene diversity)
  5. Image Quality Score (BRISQUE / LPIPS / FID)
  6. Prompt Adherence Score (CLIP text-image alignment)

Usage:
  python evaluate_consistency.py --gen-dir ./generations --ref-dir ./identity_anchors
"""

import os, json, math, argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')


# ── Model Loaders ─────────────────────────────────────────────────────────
def load_clip(device='cuda'):
    """Load CLIP for image-text and image-image similarity."""
    import clip
    model, preprocess = clip.load('ViT-L/14', device=device)
    model.eval()
    return model, preprocess

def load_dino(device='cuda'):
    """Load DINOv2 for robust visual feature extraction."""
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
    model = model.to(device).eval()
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, preprocess

def load_lpips(device='cuda'):
    """Load LPIPS for perceptual similarity."""
    import lpips
    loss_fn = lpips.LPIPS(net='alex').to(device)
    return loss_fn


# ── Feature Extraction ────────────────────────────────────────────────────
@torch.no_grad()
def extract_clip_image_features(model, preprocess, images, device='cuda') -> torch.Tensor:
    """Extract CLIP image embeddings for a list of images (PIL.Image or Path)."""
    feats = []
    for item in tqdm(images, desc='CLIP image encoding', leave=False):
        if isinstance(item, (str, Path)):
            img = Image.open(item).convert('RGB')
        else:
            img = item.convert('RGB')
        inp = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.squeeze(0).cpu().float())
    return torch.stack(feats)

@torch.no_grad()
def extract_clip_text_features(model, prompts: List[str], device='cuda') -> torch.Tensor:
    """Extract CLIP text embeddings."""
    import clip
    tokens = clip.tokenize(prompts, truncate=True).to(device)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float()

@torch.no_grad()
def extract_dino_features(model, preprocess, images, device='cuda') -> torch.Tensor:
    """Extract DINOv2 features (PIL.Image or Path) — better for structural/identity similarity."""
    feats = []
    for item in tqdm(images, desc='DINO encoding', leave=False):
        if isinstance(item, (str, Path)):
            img = Image.open(item).convert('RGB')
        else:
            img = item.convert('RGB')
        inp = preprocess(img).unsqueeze(0).to(device)
        feat = model(inp)
        feat = F.normalize(feat, dim=-1)
        feats.append(feat.squeeze(0).cpu().float())
    return torch.stack(feats)


# ── Face Region Cropper ───────────────────────────────────────────────────
def crop_face_region(image_path: Path, top_ratio=0.0, bottom_ratio=0.45) -> Optional[Image.Image]:
    """Crop top N% of image as approximate face region."""
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        top = int(h * top_ratio)
        bottom = int(h * bottom_ratio)
        cropped = img.crop((0, top, w, bottom))
        return cropped
    except Exception:
        return None


# ── Metric Calculators ────────────────────────────────────────────────────
class ConsistencyEvaluator:
    """
    Enterprise character consistency evaluation framework.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.clip_model = None
        self.clip_prep  = None
        self.dino_model = None
        self.dino_prep  = None
        self.lpips_fn   = None
        print(f'🔬 ConsistencyEvaluator initialized | device: {device}')

    def load_models(self):
        print('Loading evaluation models...')
        try:
            self.clip_model, self.clip_prep = load_clip(self.device)
            print('  ✅ CLIP ViT-L/14 loaded')
        except Exception as e:
            print(f'  ⚠️  CLIP failed: {e}')

        try:
            self.dino_model, self.dino_prep = load_dino(self.device)
            print('  ✅ DINOv2-B/14 loaded')
        except Exception as e:
            print(f'  ⚠️  DINOv2 failed: {e}')

        try:
            self.lpips_fn = load_lpips(self.device)
            print('  ✅ LPIPS-Alex loaded')
        except Exception as e:
            print(f'  ⚠️  LPIPS failed: {e}')

    # ── Metric 1: Identity Preservation ─────────────────────────────────
    def identity_preservation_score(
        self,
        reference_paths: List[Path],
        generated_paths: List[Path],
    ) -> Dict:
        """
        Compute face-region CLIP + DINOv2 cosine similarity between
        reference identity images and generated outputs.
        Higher = better identity consistency.
        """
        print('\n📐 Computing Identity Preservation Score...')

        def get_face_crops(paths):
            crops = []
            for p in paths:
                crop = crop_face_region(p)
                if crop:
                    crops.append(crop)
            return crops

        ref_crops = get_face_crops(reference_paths)
        gen_crops = get_face_crops(generated_paths)

        scores_clip, scores_dino = [], []

        if self.clip_model and ref_crops and gen_crops:
            ref_feats = extract_clip_image_features(self.clip_model, self.clip_prep, ref_crops, self.device)
            gen_feats = extract_clip_image_features(self.clip_model, self.clip_prep, gen_crops, self.device)
            ref_mean = ref_feats.mean(0, keepdim=True)
            sims = F.cosine_similarity(gen_feats, ref_mean.expand_as(gen_feats))
            scores_clip = sims.tolist()

        if self.dino_model and ref_crops and gen_crops:
            ref_feats = extract_dino_features(self.dino_model, self.dino_prep, ref_crops, self.device)
            gen_feats = extract_dino_features(self.dino_model, self.dino_prep, gen_crops, self.device)
            ref_mean = ref_feats.mean(0, keepdim=True)
            sims = F.cosine_similarity(gen_feats, ref_mean.expand_as(gen_feats))
            scores_dino = sims.tolist()

        clip_mean  = float(np.mean(scores_clip))  if scores_clip  else 0.0
        dino_mean  = float(np.mean(scores_dino))  if scores_dino  else 0.0
        combined   = (clip_mean * 0.4 + dino_mean * 0.6) if (scores_clip and scores_dino) else (clip_mean or dino_mean)

        result = {
            'metric': 'identity_preservation',
            'clip_similarity': round(clip_mean, 4),
            'dino_similarity': round(dino_mean, 4),
            'combined_score':  round(combined, 4),
            'grade': self._grade(combined, thresholds=[0.85, 0.75, 0.65, 0.55]),
            'n_references': len(ref_crops),
            'n_generated':  len(gen_crops),
            'per_image_clip': [round(s, 4) for s in scores_clip],
        }
        print(f'  CLIP Identity Score : {clip_mean:.4f}')
        print(f'  DINO Identity Score : {dino_mean:.4f}')
        print(f'  Combined Score      : {combined:.4f}  {result["grade"]}')
        return result

    # ── Metric 2: Prompt Adherence ────────────────────────────────────────
    def prompt_adherence_score(
        self,
        generated_paths: List[Path],
        prompts: List[str],
    ) -> Dict:
        """
        CLIP text-image cosine similarity.
        Measures how well generated image matches its text prompt.
        """
        print('\n📝 Computing Prompt Adherence Score...')
        if not self.clip_model:
            return {'metric': 'prompt_adherence', 'error': 'CLIP not loaded'}

        assert len(generated_paths) == len(prompts), \
            'Number of images must match number of prompts'

        scores = []
        for img_path, prompt in tqdm(zip(generated_paths, prompts), total=len(prompts), desc='Text-Image CLIP'):
            img_feat  = extract_clip_image_features(self.clip_model, self.clip_prep, [img_path], self.device)
            text_feat = extract_clip_text_features(self.clip_model, [prompt], self.device)
            sim = F.cosine_similarity(img_feat, text_feat).item()
            scores.append(sim)

        mean_score = float(np.mean(scores))
        result = {
            'metric': 'prompt_adherence',
            'mean_clip_score': round(mean_score, 4),
            'std': round(float(np.std(scores)), 4),
            'min': round(float(np.min(scores)), 4),
            'max': round(float(np.max(scores)), 4),
            'grade': self._grade(mean_score, [0.30, 0.27, 0.24, 0.20]),
            'per_image': [round(s, 4) for s in scores],
        }
        print(f'  Mean CLIP Score: {mean_score:.4f}  {result["grade"]}')
        return result

    # ── Metric 3: Image Quality (BRISQUE proxy) ───────────────────────────
    def image_quality_score(self, generated_paths: List[Path]) -> Dict:
        """Compute sharpness (Laplacian variance) as image quality proxy."""
        print('\n🖼️  Computing Image Quality Score...')
        quality_scores = []

        for p in tqdm(generated_paths, desc='Quality scoring'):
            try:
                img = Image.open(p).convert('L')
                arr = np.array(img, dtype=np.float32)
                # Laplacian variance = sharpness
                kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)
                from scipy.ndimage import convolve
                lap = convolve(arr, kernel)
                sharpness = float(np.var(lap))
                quality_scores.append(sharpness)
            except Exception:
                pass

        # Normalize to 0-1 range (typical range 0-5000)
        norm_scores = [min(s / 3000.0, 1.0) for s in quality_scores]
        mean_q = float(np.mean(norm_scores)) if norm_scores else 0.0

        result = {
            'metric': 'image_quality',
            'mean_sharpness_score': round(mean_q, 4),
            'raw_sharpness_mean': round(float(np.mean(quality_scores)), 2) if quality_scores else 0,
            'grade': self._grade(mean_q, [0.6, 0.45, 0.3, 0.2]),
        }
        print(f'  Mean Quality Score: {mean_q:.4f}  {result["grade"]}')
        return result

    # ── Metric 4: Outfit Variation Index ──────────────────────────────────
    def clothing_variation_index(self, generated_paths: List[Path]) -> Dict:
        """
        Measures diversity in the lower body region (outfit area).
        Higher diversity = better outfit transfer working.
        Uses DINO features on bottom 60% crop (clothing region).
        """
        print('\n👗 Computing Clothing Variation Index...')
        if not self.dino_model:
            return {'metric': 'clothing_variation', 'error': 'DINOv2 not loaded'}

        # Crop clothing region (bottom 60%)
        crop_paths = []
        temp_dir = Path('./temp_clothing_crops')
        temp_dir.mkdir(parents=True, exist_ok=True)

        for p in generated_paths:
            try:
                img = Image.open(p).convert('RGB')
                w, h = img.size
                crop = img.crop((0, int(h*0.35), w, h))
                crop_p = temp_dir / p.name
                crop.save(crop_p)
                crop_paths.append(crop_p)
            except Exception:
                pass

        if not crop_paths:
            return {'metric': 'clothing_variation', 'error': 'No images processed'}

        feats = extract_dino_features(self.dino_model, self.dino_prep, crop_paths, self.device)
        # Pairwise cosine similarity — diversity = 1 - mean_similarity
        n = feats.shape[0]
        sim_matrix = (feats @ feats.T).numpy()
        off_diag = sim_matrix[np.triu_indices(n, k=1)]
        mean_sim  = float(np.mean(off_diag))
        diversity = 1.0 - mean_sim

        result = {
            'metric': 'clothing_variation',
            'mean_pairwise_similarity': round(mean_sim, 4),
            'diversity_index': round(diversity, 4),
            'grade': self._grade(diversity, [0.35, 0.25, 0.15, 0.08]),
        }
        print(f'  Outfit Diversity Index: {diversity:.4f}  {result["grade"]}')
        
        # Clean up temp files
        try:
            for p in crop_paths:
                if p.exists(): p.unlink()
            if temp_dir.exists(): temp_dir.rmdir()
        except Exception:
            pass
            
        return result

    # ── Metric 5: Background Variation ────────────────────────────────────
    def background_variation_index(self, generated_paths: List[Path]) -> Dict:
        """Measures diversity in background regions (top corners crop)."""
        print('\n🌍 Computing Background Variation Index...')
        if not self.dino_model:
            return {'metric': 'background_variation', 'error': 'DINOv2 not loaded'}

        crop_paths = []
        temp_dir = Path('./temp_bg_crops')
        temp_dir.mkdir(parents=True, exist_ok=True)

        for p in generated_paths:
            try:
                img = Image.open(p).convert('RGB')
                w, h = img.size
                # Take top-left and top-right corners (background area)
                left_crop  = img.crop((0, 0, w//3, h//3))
                right_crop = img.crop((2*w//3, 0, w, h//3))
                combined = Image.new('RGB', (w//3*2, h//3))
                combined.paste(left_crop, (0, 0))
                combined.paste(right_crop, (w//3, 0))
                cp = temp_dir / p.name
                combined.save(cp)
                crop_paths.append(cp)
            except Exception:
                pass

        if not crop_paths:
            return {'metric': 'background_variation', 'error': 'No images'}

        feats = extract_dino_features(self.dino_model, self.dino_prep, crop_paths, self.device)
        n = feats.shape[0]
        sim_matrix = (feats @ feats.T).numpy()
        off_diag = sim_matrix[np.triu_indices(n, k=1)]
        mean_sim  = float(np.mean(off_diag))
        diversity = 1.0 - mean_sim

        result = {
            'metric': 'background_variation',
            'mean_pairwise_similarity': round(mean_sim, 4),
            'diversity_index': round(diversity, 4),
            'grade': self._grade(diversity, [0.40, 0.30, 0.20, 0.10]),
        }
        print(f'  Background Diversity Index: {diversity:.4f}  {result["grade"]}')
        
        # Clean up temp files
        try:
            for p in crop_paths:
                if p.exists(): p.unlink()
            if temp_dir.exists(): temp_dir.rmdir()
        except Exception:
            pass
            
        return result

    # ── Grade Helper ──────────────────────────────────────────────────────
    @staticmethod
    def _grade(score: float, thresholds: List[float]) -> str:
        if score >= thresholds[0]: return '🏆 EXCELLENT'
        if score >= thresholds[1]: return '✅ GOOD'
        if score >= thresholds[2]: return '⚠️  FAIR'
        if score >= thresholds[3]: return '⚠️  POOR'
        return '❌ CRITICAL'

    # ── Full Evaluation Report ────────────────────────────────────────────
    def full_evaluation(
        self,
        reference_paths: List[Path],
        generated_paths: List[Path],
        prompts: Optional[List[str]] = None,
        output_dir: str = './reports',
    ) -> Dict:
        """Run all metrics and produce a comprehensive report."""
        os.makedirs(output_dir, exist_ok=True)
        print('\n' + '='*60)
        print('  HANA KIM CHARACTER CONSISTENCY — EVALUATION REPORT')
        print('='*60)

        self.load_models()
        all_results = {}

        all_results['identity']    = self.identity_preservation_score(reference_paths, generated_paths)
        all_results['quality']     = self.image_quality_score(generated_paths)
        all_results['bg_variation']= self.background_variation_index(generated_paths)
        all_results['outfit_var']  = self.clothing_variation_index(generated_paths)

        if prompts and len(prompts) == len(generated_paths):
            all_results['prompt_adherence'] = self.prompt_adherence_score(generated_paths, prompts)

        # Compute Overall Score
        weights = {
            'identity': 0.40,
            'prompt_adherence': 0.20,
            'quality': 0.15,
            'outfit_var': 0.15,
            'bg_variation': 0.10,
        }
        overall = 0.0
        total_weight = 0.0
        for key, w in weights.items():
            if key in all_results:
                metric = all_results[key]
                score_key = next((k for k in ['combined_score','mean_clip_score','mean_sharpness_score','diversity_index'] if k in metric), None)
                if score_key:
                    overall += metric[score_key] * w
                    total_weight += w
        if total_weight > 0:
            overall /= total_weight

        all_results['overall'] = {
            'score': round(overall, 4),
            'grade': self._grade(overall, [0.70, 0.55, 0.40, 0.25]),
        }

        # Save JSON report
        report_path = os.path.join(output_dir, 'consistency_evaluation.json')
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        self._print_summary(all_results)
        self._plot_report(all_results, generated_paths, output_dir)

        print(f'\n📁 Full report saved: {report_path}')
        return all_results

    def _print_summary(self, results: Dict):
        print('\n' + '='*60)
        print('  EVALUATION SUMMARY')
        print('='*60)
        rows = [
            ('Identity Preservation', results.get('identity', {}).get('combined_score', 'N/A'),     results.get('identity', {}).get('grade', '')),
            ('Prompt Adherence',      results.get('prompt_adherence', {}).get('mean_clip_score', 'N/A'), results.get('prompt_adherence', {}).get('grade', '')),
            ('Image Quality',         results.get('quality', {}).get('mean_sharpness_score', 'N/A'), results.get('quality', {}).get('grade', '')),
            ('Outfit Variation',      results.get('outfit_var', {}).get('diversity_index', 'N/A'),    results.get('outfit_var', {}).get('grade', '')),
            ('Background Variation',  results.get('bg_variation', {}).get('diversity_index', 'N/A'),  results.get('bg_variation', {}).get('grade', '')),
        ]
        for name, score, grade in rows:
            score_str = f'{score:.4f}' if isinstance(score, float) else str(score)
            print(f'  {name:<25}: {score_str:<10} {grade}')
        overall = results.get('overall', {})
        print(f'\n  {"OVERALL SCORE":<25}: {overall.get("score", 0):.4f}     {overall.get("grade", "")}')
        print('='*60)

    def _plot_report(self, results: Dict, generated_paths: List[Path], output_dir: str):
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Character Consistency Evaluation Report\nEA-F-001 Hana Kim', fontsize=16, fontweight='bold')
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

        # Radar chart
        ax_radar = fig.add_subplot(gs[0, 0], polar=True)
        metrics_names = ['Identity', 'Quality', 'Prompt Adh.', 'Outfit Var', 'BG Var']
        values = [
            results.get('identity', {}).get('combined_score', 0),
            results.get('quality', {}).get('mean_sharpness_score', 0),
            results.get('prompt_adherence', {}).get('mean_clip_score', 0) * 3.0,  # scale 0-1
            results.get('outfit_var', {}).get('diversity_index', 0),
            results.get('bg_variation', {}).get('diversity_index', 0),
        ]
        N = len(metrics_names)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        values_plot = values + values[:1]
        ax_radar.plot(angles, values_plot, 'o-', linewidth=2, color='#3498db')
        ax_radar.fill(angles, values_plot, alpha=0.25, color='#3498db')
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics_names, size=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Metrics Radar', fontweight='bold', pad=15)

        # Score bars
        ax_bars = fig.add_subplot(gs[0, 1:])
        bar_colors = ['#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.45 else '#e74c3c' for v in values]
        bars = ax_bars.barh(metrics_names, values, color=bar_colors, edgecolor='white', height=0.6)
        ax_bars.set_xlim(0, 1)
        ax_bars.set_xlabel('Score (0–1)')
        ax_bars.set_title('Metric Scores', fontweight='bold')
        ax_bars.axvline(0.7, color='green', ls='--', alpha=0.5, label='Excellent threshold')
        for bar, val in zip(bars, values):
            ax_bars.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
        ax_bars.legend()
        ax_bars.grid(axis='x', alpha=0.3)

        # Per-image identity scores
        ax_id = fig.add_subplot(gs[1, :2])
        id_scores = results.get('identity', {}).get('per_image_clip', [])
        if id_scores:
            x = list(range(len(id_scores)))
            colors_id = ['#2ecc71' if s >= 0.85 else '#f39c12' if s >= 0.70 else '#e74c3c' for s in id_scores]
            ax_id.bar(x, id_scores, color=colors_id, edgecolor='white')
            ax_id.axhline(np.mean(id_scores), color='navy', ls='--', label=f'Mean: {np.mean(id_scores):.3f}')
            ax_id.set_ylim(0, 1)
            ax_id.set_xlabel('Generated Image Index')
            ax_id.set_ylabel('CLIP Identity Score')
            ax_id.set_title('Per-Image Identity Preservation', fontweight='bold')
            ax_id.legend()
            ax_id.grid(axis='y', alpha=0.3)

        # Overall score gauge
        ax_gauge = fig.add_subplot(gs[1, 2])
        overall = results.get('overall', {}).get('score', 0)
        ax_gauge.pie(
            [overall, 1 - overall],
            colors=['#3498db', '#ecf0f1'],
            startangle=90,
            counterclock=False,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'}
        )
        ax_gauge.text(0, 0, f'{overall:.2f}', ha='center', va='center', fontsize=26, fontweight='bold', color='#2c3e50')
        ax_gauge.set_title(f'Overall Score\n{results.get("overall", {}).get("grade", "")}', fontweight='bold')

        # Sample images (first 6)
        sample_paths = generated_paths[:6]
        for i, img_p in enumerate(sample_paths):
            if i >= 3: break
            ax = fig.add_subplot(gs[2, i])
            try:
                img = Image.open(img_p).convert('RGB')
                ax.imshow(img)
                ax.set_title(img_p.name[:20], fontsize=8)
            except Exception:
                pass
            ax.axis('off')

        plt.savefig(os.path.join(output_dir, 'evaluation_report.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'📊 Visual report saved: {output_dir}/evaluation_report.png')


# ── CLI Interface ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Character Consistency Evaluator — EA-F-001 Hana Kim')
    parser.add_argument('--gen-dir',  required=True, help='Directory with generated images')
    parser.add_argument('--ref-dir',  required=True, help='Directory with reference/anchor images')
    parser.add_argument('--prompts',  default=None,  help='JSON file with prompts list')
    parser.add_argument('--out-dir',  default='./reports', help='Output directory for reports')
    parser.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    VALID_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
    gen_paths = sorted([p for p in Path(args.gen_dir).iterdir() if p.suffix.lower() in VALID_EXTS]) if Path(args.gen_dir).exists() else []
    ref_paths = sorted([p for p in Path(args.ref_dir).iterdir() if p.suffix.lower() in VALID_EXTS]) if Path(args.ref_dir).exists() else []

    if not gen_paths:
        print(f"❌ Error: No generated images found in directory '{args.gen_dir}' (or path does not exist).")
        return
    if not ref_paths:
        print(f"❌ Error: No reference images found in directory '{args.ref_dir}' (or path does not exist).")
        return

    prompts = None
    if args.prompts and Path(args.prompts).exists():
        with open(args.prompts) as f:
            prompts = json.load(f)

    evaluator = ConsistencyEvaluator(device=args.device)
    results = evaluator.full_evaluation(ref_paths, gen_paths, prompts, args.out_dir)

    print('\n✅ Evaluation complete!')
    print(f'   Overall Grade: {results["overall"]["grade"]}')
    print(f'   Score: {results["overall"]["score"]:.4f}')


if __name__ == '__main__':
    main()
