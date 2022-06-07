import shutil

from pathlib import Path
from typing import List


def get_all_files_in_folder(folder: str, types: List) -> List[Path]:
    files_grabbed = []
    for t in types:
        files_grabbed.extend(Path(folder).rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def recreate_folder(path):
    output_dir = Path(path)
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
