from github import Github
import os

try:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is missing!")
        exit(1)
        
    g = Github(token)
    
    # 1. Verify token works by getting user
    user = g.get_user()
    print(f"Token authenticated as user: {user.login}")
    
    # 2. Get repository
    repo_name = "BIGREASONS/buginsight-actions-test"
    print(f"Attempting to fetch repo: {repo_name}")
    repo = g.get_repo(repo_name)
    print(f"Repo fetched successfully: {repo.full_name}")
    
    # 3. Get pull request
    pr_number = 1
    print(f"Attempting to fetch PR #{pr_number}")
    pr = repo.get_pull(pr_number)
    print(f"PR fetched successfully: {pr.title}")
    
    # 4. Attempt to create comment
    print("Attempting to create issue comment...")
    comment = pr.create_issue_comment("BugInsight test comment - verifying API integration works")
    print(f"SUCCESS! Comment created at: {comment.html_url}")
    
except Exception as e:
    print(f"EXCEPTION: {type(e).__name__} - {str(e)}")
