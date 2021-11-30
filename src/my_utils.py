from pathlib import Path
from typing import List


def get_all_files_in_folder(folder: Path, types: List) -> List[Path]:
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed
