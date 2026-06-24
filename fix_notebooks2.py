import glob

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    # For ipynb cells that have the string:
    content = content.replace("'/teamspace/studios/this_studio'", "os.getcwd()")
    content = content.replace("'/teamspace/studios/this_studio/character_id/train'", "f'{STUDIO_ROOT}/character_id/train'")
    content = content.replace("f'/teamspace/studios/this_studio/mlruns'", "f'{STUDIO_ROOT}/mlruns'")
    content = content.replace("'YOUR_WANDB_API_KEY'", "os.environ.get('WANDB_API_KEY', '')")
    content = content.replace("f'/teamspace/studios/this_studio/", "f'{STUDIO_ROOT}/")
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for nb_file in glob.glob('*.ipynb'):
    fix_file(nb_file)
