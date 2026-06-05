import os
import tempfile
import subprocess
import shutil
import difflib

def fetch_original_file(repo_url: str, file_path: str) -> str:
    """Fetches the original file from the target repository via git clone."""
    temp_dir = tempfile.mkdtemp(prefix="buginsight_fetch_")
    try:
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], capture_output=True, text=True, check=True)
        target_file_path = os.path.join(temp_dir, file_path)
        if os.path.exists(target_file_path):
            with open(target_file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise FileNotFoundError(f"File {file_path} not found in repo {repo_url}")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch original file: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return ""

def generate_unified_diff(original: str, patched: str, file_path: str) -> str:
    """Programmatically generates a unified git diff from two source strings."""
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(
        original_lines, 
        patched_lines, 
        fromfile=f"a/{file_path}", 
        tofile=f"b/{file_path}",
        n=3
    ))
    return "".join(diff)
