from pathlib import Path
import fnmatch

AVOID_FILE = "avoid_files.txt"

def load_avoid_patterns():
    avoid_path = Path(AVOID_FILE)
    if avoid_path.exists():
        with open(avoid_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
            return [line for line in lines if line and not line.startswith("#")]
    return []

def path_matches_any_pattern(path_str, patterns):
    for pattern in patterns:
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False

def filter_files(files, base_folder):
    patterns = load_avoid_patterns()
    filtered = []
    for f in files:
        try:
            rel_path = f.relative_to(base_folder).as_posix()
        except ValueError:
            rel_path = f.resolve().as_posix()
        if not path_matches_any_pattern(rel_path, patterns):
            filtered.append(f)
    return filtered
