import os
import shutil

def get_all_files(path, prefix="", suffix="", contains=""):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")
    files = []
    for pre, dirs, basenames in os.walk(path):
        for name in basenames:
            if name.startswith(prefix) and name.endswith(suffix) and contains in name:
                files.append(os.path.join(pre, name))
    return files

def remove_files_in_dir(dir):
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)