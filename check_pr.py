import os
from github import Github

token = os.environ.get("GITHUB_TOKEN")
g = Github(token)
repo = g.get_repo("BIGREASONS/buginsight-actions-test")

for pr in repo.get_pulls(state='all'):
    print(f"PR #{pr.number}: {pr.title} by {pr.user.login}")
