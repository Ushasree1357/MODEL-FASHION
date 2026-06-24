import json
import os
import glob

def fix_notebook(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell.get('source', []):
                old_line = line
                # Fix STUDIO_ROOT
                if "STUDIO_ROOT = '/teamspace/studios/this_studio'" in line or "STUDIO_ROOT   = '/teamspace/studios/this_studio'" in line:
                    line = line.replace("'/teamspace/studios/this_studio'", "os.getcwd()")
                
                # Fix EXTRA_SCAN
                if "EXTRA_SCAN = '/teamspace/studios/this_studio/character_id/train'" in line:
                    line = line.replace("'/teamspace/studios/this_studio/character_id/train'", "f'{STUDIO_ROOT}/character_id/train'")
                
                # Fix MLFLOW_DIR in train notebook
                if "MLFLOW_DIR = f'/teamspace/studios/this_studio/mlruns'" in line:
                    line = line.replace("f'/teamspace/studios/this_studio/mlruns'", "f'{STUDIO_ROOT}/mlruns'")
                
                # Fix WANDB Placeholder
                if "WANDB_API_KEY = 'YOUR_WANDB_API_KEY'" in line:
                    line = line.replace("'YOUR_WANDB_API_KEY'", "os.environ.get('WANDB_API_KEY', '')")
                
                # Fix report json paths
                if "with open(f'/teamspace/studios/this_studio/reports/" in line:
                    line = line.replace("f'/teamspace/studios/this_studio/", "f'{STUDIO_ROOT}/")
                
                new_source.append(line)
                if old_line != line:
                    modified = True
            
            cell['source'] = new_source
            
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Fixed {filepath}")

for nb_file in glob.glob('*.ipynb'):
    fix_notebook(nb_file)
