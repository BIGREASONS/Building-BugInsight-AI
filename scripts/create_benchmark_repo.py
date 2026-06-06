import os
from github import Github, GithubException

token = os.environ.get("GITHUB_TOKEN")
g = Github(token)
user = g.get_user()

repo_name = "buginsight-benchmark"

try:
    repo = user.create_repo(
        repo_name,
        description="Benchmark suite for BugInsight vulnerabilities.",
        private=False,
        auto_init=True
    )
    print(f"Created repository: {repo.full_name}")
except GithubException as e:
    if e.status == 422:
        print(f"Repository {repo_name} already exists. Using existing.")
        repo = user.get_repo(f"{user.login}/{repo_name}")
    else:
        raise

# Now create files in the repo
files = {
    "README.md": "# BugInsight Benchmark Suite\n\nA suite of vulnerable and safe files for evaluating BugInsight.",
    
    # Vulnerable files
    "vulnerable/sql_injection.py": '''import sqlite3

def get_user(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}'"
    cursor.execute(query)
    return cursor.fetchall()
''',

    "vulnerable/command_injection.py": '''import subprocess

def ping_host(host):
    # Vulnerable to command injection
    cmd = f"ping -c 4 {host}"
    return subprocess.check_output(cmd, shell=True)
''',

    "vulnerable/path_traversal.py": '''import os

def read_file(filename):
    # Vulnerable to path traversal
    filepath = os.path.join("/var/www/html", filename)
    with open(filepath, "r") as f:
        return f.read()
''',

    "vulnerable/hardcoded_secret.py": '''def get_api_client():
    # Vulnerable to hardcoded secrets
    api_key = "AKIAIOSFODNN7EXAMPLE"
    secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    return {"key": api_key, "secret": secret_key}
''',

    "vulnerable/weak_crypto.py": '''import hashlib

def hash_password(password):
    # Vulnerable: uses weak MD5 hash
    hasher = hashlib.md5()
    hasher.update(password.encode('utf-8'))
    return hasher.hexdigest()
''',

    "vulnerable/xss.py": '''from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/hello')
def hello():
    name = request.args.get('name', 'World')
    # Vulnerable to XSS
    response = make_response(f"<h1>Hello {name}</h1>")
    return response
''',

    "vulnerable/unsafe_deserialization.py": '''import pickle
import base64

def load_data(serialized_data):
    # Vulnerable: unpickles untrusted data
    data = base64.b64decode(serialized_data)
    return pickle.loads(data)
''',

    "vulnerable/multi_vulnerability.py": '''import sqlite3
import hashlib

def authenticate_user(username, password):
    # 1. Hardcoded secret (secret key)
    admin_token = "super_secret_admin_token_123"
    
    # 2. Weak crypto (MD5)
    hashed_pw = hashlib.md5(password.encode()).hexdigest()
    
    # 3. SQL Injection
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{hashed_pw}'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user and username == "admin":
        return admin_token
    return user
''',

    # Safe files
    "safe/safe_auth.py": '''import sqlite3
import hashlib
import os

def authenticate_user(username, password):
    # Uses strong crypto and parameterized queries
    salt = os.urandom(32)
    hashed_pw = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username=? AND password=?"
    cursor.execute(query, (username, hashed_pw))
    return cursor.fetchone()
''',

    "safe/safe_api.py": '''import os

def get_api_client():
    # Safe: loads from environment variables
    api_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not api_key or not secret_key:
        raise ValueError("Missing AWS credentials")
    return {"key": api_key, "secret": secret_key}
''',

    "safe/safe_file_io.py": '''import os

def read_safe_file(filename):
    # Safe: prevents path traversal
    base_dir = "/var/www/html"
    filepath = os.path.abspath(os.path.join(base_dir, filename))
    if not filepath.startswith(base_dir):
        raise ValueError("Path traversal detected")
        
    with open(filepath, "r") as f:
        return f.read()
''',

    "safe/safe_crypto.py": '''from cryptography.fernet import Fernet
import os

def encrypt_data(data):
    # Safe: uses strong encryption and loads key from env
    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        raise ValueError("Missing ENCRYPTION_KEY")
        
    f = Fernet(key.encode())
    return f.encrypt(data.encode())
''',

    "safe/safe_web.py": '''from flask import Flask, request, make_response
import html

app = Flask(__name__)

@app.route('/hello')
def hello():
    name = request.args.get('name', 'World')
    # Safe: escapes HTML characters
    safe_name = html.escape(name)
    response = make_response(f"<h1>Hello {safe_name}</h1>")
    return response
'''
}

for path, content in files.items():
    try:
        repo.get_contents(path)
        print(f"File {path} already exists. Updating...")
        file_info = repo.get_contents(path)
        repo.update_file(path, f"Update {path}", content, file_info.sha)
    except GithubException as e:
        if e.status == 404:
            print(f"Creating {path}...")
            repo.create_file(path, f"Create {path}", content)
        else:
            raise

print("Benchmark repository setup complete.")
