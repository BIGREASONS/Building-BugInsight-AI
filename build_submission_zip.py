import os
import zipfile

# ALLOWLIST APPROACH: Maximum Polish
ALLOWED_DIRS = {'swarm', 'frontend', 'configs', 'docker', 'data', 'evaluation', 'docs', 'buginsight-demo'}
ALLOWED_FILES = {'README.md', 'requirements.txt', 'LICENSE'}

# Still exclude caches even within allowed dirs
EXCLUDE_DIRS = {'.git', '.pytest_cache', '__pycache__', 'node_modules', '.next'}
EXCLUDE_EXTS = {'.pyc', '.pyo', '.pyd', '.env'}

def should_exclude_file(root, filename):
    # If it's a root-level file, it must be in ALLOWED_FILES
    if root == '.':
        if filename not in ALLOWED_FILES:
            return True
            
    if filename.startswith('.') and filename not in {'.gitignore', '.gitattributes'}:
        return True
        
    _, ext = os.path.splitext(filename)
    if ext in EXCLUDE_EXTS:
        return True
        
    return False

def build_zip(zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # If we are at the root level, ONLY allow the explicitly ALLOWED_DIRS
            if root == '.':
                dirs[:] = [d for d in dirs if d in ALLOWED_DIRS]
            else:
                # Inside allowed dirs, just filter out standard caches
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            
            for file in files:
                if not should_exclude_file(root, file):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)

if __name__ == '__main__':
    build_zip('buginsight-swarm-submission-polished.zip')
    print("Maximum polish zip created successfully.")
