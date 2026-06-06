import os
from github import Github

token = os.environ.get("GITHUB_TOKEN")
g = Github(token)
repo = g.get_repo("BIGREASONS/buginsight-actions-test")

for branch in repo.get_branches():
    print(branch.name)
