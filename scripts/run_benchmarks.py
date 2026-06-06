import os
import time
import requests
import json
import argparse
from sseclient import SSEClient

TARGET_REPO = "https://github.com/BIGREASONS/buginsight-benchmark"

VULNERABLE_FILES = [
    "vulnerable/sql_injection.py",
    "vulnerable/command_injection.py",
    "vulnerable/path_traversal.py",
    "vulnerable/hardcoded_secret.py",
    "vulnerable/weak_crypto.py",
    "vulnerable/xss.py",
    "vulnerable/unsafe_deserialization.py",
    "vulnerable/multi_vulnerability.py",
]

SAFE_FILES = [
    "safe/safe_auth.py",
    "safe/safe_api.py",
    "safe/safe_file_io.py",
    "safe/safe_crypto.py",
    "safe/safe_web.py",
]

REGRESSION_FILES = [
    "benchmarks/regression/fixed_sqli.py",
    "benchmarks/regression/fixed_path_traversal.py",
]

API_URL = "http://127.0.0.1:8002"
WEBHOOK_SECRET = "test-secret"  # Assuming test_local environment

def get_failure_category(reason):
    if not reason:
        return None
    reason_lower = reason.lower()
    if "syntaxerror" in reason_lower:
        return "Syntax Error"
    elif "undefined name" in reason_lower:
        return "Undefined Variable"
    elif "functional preservation failed" in reason_lower:
        return "Functional Preservation Failure"
    elif "truncated" in reason_lower:
        return "Truncation"
    elif "rescan failed" in reason_lower:
        return "Rescan Failure"
    else:
        return "Other"

def run_benchmark(target_file, is_vulnerable, model, prompt_version):
    print(f"\n--- Testing: {target_file} ---")
    headers = {
        "Authorization": f"Bearer {WEBHOOK_SECRET}",
        "Content-Type": "application/json"
    }
    payload = {
        "repo_url": TARGET_REPO,
        "pr_number": 1,
        "branch": "main",
        "target_file": target_file
    }

    start_time = time.time()
    res = requests.post(f"{API_URL}/api/v1/analyze_pr", json=payload, headers=headers)
    if res.status_code != 200:
        print(f"Error starting job: {res.text}")
        return None
        
    job_id = res.json()["job_id"]
    print(f"Job ID: {job_id}. Listening to SSE stream...")

    sse_url = f"{API_URL}/api/swarm/stream/{job_id}"
    sse_response = requests.get(sse_url, stream=True)
    messages = SSEClient(sse_response)
    
    final_data = None
    agent_runtimes = {}
    last_agent_time = start_time
    
    for msg in messages.events():
        if not msg.data:
            continue
        try:
            event = json.loads(msg.data)
            if event["type"] == "agent_complete":
                agent_name = event['data'].get('agent')
                print(f"[{agent_name}] Complete")
                current_time = time.time()
                agent_runtimes[agent_name] = current_time - last_agent_time
                last_agent_time = current_time
            elif event["type"] == "swarm_complete":
                final_data = event["data"]
                break
        except Exception as e:
            print(f"Error parsing SSE: {e}")

    duration = time.time() - start_time
    
    if not final_data:
        print("Failed to get swarm_complete payload.")
        return None

    findings = final_data.get("scanner_findings", [])
    detected = len(findings) > 0
    patch = final_data.get("patch", "")
    patch_generated = bool(patch and not patch.startswith("Failed to generate patch:"))
    val_passed = final_data.get("is_patch_valid", False)
    rescan_passed = final_data.get("rescan_passed", False)
    
    first_failed_gate = final_data.get("first_failed_gate")
    failure_reason = final_data.get("validation_reasoning")
    
    # BugInsight might not attempt fix if nothing detected
    if not detected:
        patch_generated = False
        val_passed = False
        rescan_passed = False

    if not val_passed and not first_failed_gate:
        first_failed_gate = "unknown"
        
    generated_lines = final_data.get("generated_patch_lines", 0)
    original_lines = final_data.get("original_lines", 0)
    patch_minimality = 0
    if original_lines > 0:
        # Assuming patch line count represents total file lines, changed lines approx
        # For simplicity, we just track the ratio
        patch_minimality = generated_lines / original_lines

    result = {
        "file": target_file,
        "is_vulnerable": is_vulnerable,
        "model": model,
        "prompt_version": prompt_version,
        "detected": detected,
        "patch_generated": patch_generated,
        "val_passed": val_passed,
        "rescan_passed": rescan_passed,
        "duration": duration,
        "agent_runtimes": agent_runtimes,
        "repair_attempts": final_data.get("repair_attempts", 0),
        "repaired_successfully": final_data.get("repair_attempts", 0) > 0 and val_passed,
        "hallucination": patch_generated and not val_passed,
        "fix_accuracy": rescan_passed and val_passed,
        "first_failed_gate": first_failed_gate if patch_generated and not val_passed else None,
        "failure_category": get_failure_category(failure_reason) if patch_generated and not val_passed else None,
        "failure_reason": failure_reason if patch_generated and not val_passed else None,
        "raw_fix_output": final_data.get("raw_fix_output", ""),
        "generated_patch_lines": generated_lines,
        "original_lines": original_lines,
        "patch_minimality": patch_minimality
    }
    
    if is_vulnerable:
        success = result["fix_accuracy"]
    else:
        success = not detected # success if safe file and nothing detected
        
    result["success"] = success
    print(f"Result: {result['success']} | Gate: {result['first_failed_gate']} | Category: {result['failure_category']}")
    
    print(f"\n--- FIX OUTPUT VERIFICATION ---")
    print(f"PATCH:\n{final_data.get('patch', '')[:500]}...")
    print(f"RAW FIX OUTPUT:\n{final_data.get('raw_fix_output', '')[:500]}...")
    print(f"-------------------------------\n")
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:3b", help="Model name to test")
    parser.add_argument("--prompt-version", default="1", help="Prompt version to test (1 or 2)")
    args = parser.parse_args()

    # Configure for benchmarking
    os.environ["BENCHMARK_MODE"] = "TRUE"
    os.environ["BUGINSIGHT_PR_MODE"] = "DISABLED"
    os.environ["PRIMARY_MODEL"] = args.model
    os.environ["PROMPT_VERSION"] = args.prompt_version

    print(f"Starting BugInsight Benchmarking Suite (Model: {args.model}, Prompt: {args.prompt_version})...")
    
    results = []
    
    for file in VULNERABLE_FILES:
        res = run_benchmark(file, is_vulnerable=True, model=args.model, prompt_version=args.prompt_version)
        if res: results.append(res)
        
    for file in SAFE_FILES:
        res = run_benchmark(file, is_vulnerable=False, model=args.model, prompt_version=args.prompt_version)
        if res: results.append(res)
        
    for file in REGRESSION_FILES:
        res = run_benchmark(file, is_vulnerable=False, model=args.model, prompt_version=args.prompt_version)
        if res: results.append(res)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/benchmark_detailed.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Report
    vuln_results = [r for r in results if r["is_vulnerable"]]
    safe_results = [r for r in results if not r["is_vulnerable"]]

    total_vuln = len(vuln_results)
    detected = sum(1 for r in vuln_results if r["detected"])
    patched = sum(1 for r in vuln_results if r["patch_generated"])
    validated = sum(1 for r in vuln_results if r["val_passed"])
    rescan_passed = sum(1 for r in vuln_results if r["rescan_passed"])

    det_rate = (detected / total_vuln * 100) if total_vuln else 0
    patch_rate = (patched / total_vuln * 100) if total_vuln else 0
    val_rate = (validated / total_vuln * 100) if total_vuln else 0
    fix_acc = (sum(1 for r in vuln_results if r["fix_accuracy"]) / total_vuln * 100) if total_vuln else 0

    total_safe = len(safe_results)
    false_positives = sum(1 for r in safe_results if r["detected"])
    fp_rate = (false_positives / total_safe * 100) if total_safe else 0

    avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0

    report = f"""# BugInsight Benchmark Report
Date: {time.strftime('%Y-%m-%d')}
Model: `{args.model}`
Prompt Version: `V{args.prompt_version}`

## Summary

Total Vulnerabilities: {total_vuln}
Detection Rate: {det_rate:.1f}%
Patch Generation Rate: {patch_rate:.1f}%
Validation Pass Rate: {val_rate:.1f}%
Fix Accuracy: {fix_acc:.1f}%
False Positive Rate: {fp_rate:.1f}%
Average Runtime: {avg_duration:.1f}s

## Failure Categories
"""
    categories = {}
    for r in vuln_results:
        cat = r.get("failure_category")
        if cat:
            categories[cat] = categories.get(cat, 0) + 1
            
    for cat, count in categories.items():
        report += f"- {cat}: {count}\n"

    report += """
## Detailed Results - Vulnerable Files

| Vulnerability | Detect | Patch | Validate | Rescan | First Gate Failed | Category |
|--------------|---------|--------|----------|--------|-------------------|----------|
"""

    for r in vuln_results:
        gate = r.get('first_failed_gate', '-') or '-'
        cat = r.get('failure_category', '-') or '-'
        report += f"| `{os.path.basename(r['file'])}` | {'✅' if r['detected'] else '❌'} | {'✅' if r['patch_generated'] else '❌'} | {'✅' if r['val_passed'] else '❌'} | {'✅' if r['rescan_passed'] else '❌'} | {gate} | {cat} |\n"

    report += """
## Detailed Results - Safe Files

| File | False Positive Detected | Result | Runtime |
|--------------|---------|--------|---------|
"""

    for r in safe_results:
        report += f"| `{os.path.basename(r['file'])}` | {'❌ YES (Bad)' if r['detected'] else '✅ NO (Good)'} | {'SUCCESS' if r['success'] else 'FAIL'} | {r['duration']:.1f}s |\n"

    with open("outputs/benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nBenchmarking complete.")
    print("Detailed JSON saved to outputs/benchmark_detailed.json")
    print("Report saved to outputs/benchmark_report.md")

if __name__ == "__main__":
    main()
