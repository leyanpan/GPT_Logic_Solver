import os
debug = False

def debug_log(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

def get_dataset_path(dir, split='train', ext='txt'):
    raw_path = os.path.join(dir, f'{split}.{ext}')
    if not os.path.exists(raw_path):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(dir)
        try:
            os.system(f'python prepare.py')
        finally:
            os.chdir(cur_path)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f'File {raw_path} not found and could not be downloaded.')
    return raw_path