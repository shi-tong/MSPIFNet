import os
import pandas as pd
from pathlib import Path
import time
def make_file(args, file_path, file_name = None):
    child_path = file_path
    parent_path = args.acc_loss
    child_path = Path(child_path).resolve()
    parent_path = Path(parent_path).resolve()
    isin = parent_path in child_path.parents
    
    if isin:
        return child_path
    else:
        try:
            parent_path.mkdir(parents=True, exist_ok=True)
            if file_name is None:
                raise ValueError("Facing problem when creating a file, please set the file_name")
            file_to_create = parent_path / file_name
            
            return file_to_create
        except Exception as e:
            print(f"Error creating file: {e}")
            return False

def create_acc_loss_txt(file_path):
    # defination
    acc_loss_files = [
        'train_acc.txt',
        'valid_acc.txt',
        'test_acc.txt',
        'train_loss.txt',
        'valid_loss.txt',
        'test_loss.txt'
    ]
    all_path = []
    for filename in acc_loss_files:
        file_full_path = os.path.join(file_path, filename)
        if not os.path.exists(file_full_path):
            with open(file_full_path, 'w') as f:
                f.write("")
            print(f"[Success]: {file_full_path}")
            all_path.append(file_full_path)
        else:
            print(f"[Warning]: {file_full_path}")
    return all_path

def create_file(args):
    model_name = args.model
    parent_path = args.acc_loss
    time_str = time.strftime("%Y_%-m_%-d_%-H_%-M", time.localtime())
    folder_name = f"{model_name}_{time_str}"
    path = os.path.join(parent_path, folder_name)
    try:
        if os.path.exists(path):
            print(f"[Warning]: {path}")
        else:
            os.makedirs(path, exist_ok=True)
            print(f"[Success]: {path}")
            all_path = create_acc_loss_txt(path)
        return all_path
    
    except Exception as e:
        raise ValueError(f"[Error]: {e}")
    
def write_acc_loss(value, path, write_mode='a'):
    try:
        with open(path, write_mode) as f:
            f.write(f"{value}\n")
    except Exception as e:
        raise ValueError(f"[Error]: {e}")