import os
import time

results = [
    {'file': 'vulnerable/sql_injection.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': True, 'val_passed': False, 'rescan_passed': False, 'duration': 15.8, 'hallucination': True, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/command_injection.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': True, 'val_passed': False, 'rescan_passed': True, 'duration': 17.6, 'hallucination': True, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/path_traversal.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/hardcoded_secret.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': True, 'val_passed': False, 'rescan_passed': True, 'duration': 17.6, 'hallucination': True, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/weak_crypto.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': True, 'val_passed': False, 'rescan_passed': True, 'duration': 18.4, 'hallucination': True, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/xss.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/unsafe_deserialization.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': True, 'val_passed': False, 'rescan_passed': True, 'duration': 18.2, 'hallucination': True, 'fix_accuracy': False, 'success': False},
    {'file': 'vulnerable/multi_vulnerability.py', 'is_vulnerable': True, 'detected': True, 'patch_generated': True, 'val_passed': False, 'rescan_passed': False, 'duration': 26.8, 'hallucination': True, 'fix_accuracy': False, 'success': False},
    {'file': 'safe/safe_auth.py', 'is_vulnerable': False, 'detected': False, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': True},
    {'file': 'safe/safe_api.py', 'is_vulnerable': False, 'detected': False, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': True},
    {'file': 'safe/safe_file_io.py', 'is_vulnerable': False, 'detected': False, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': True},
    {'file': 'safe/safe_crypto.py', 'is_vulnerable': False, 'detected': False, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': True},
    {'file': 'safe/safe_web.py', 'is_vulnerable': False, 'detected': False, 'patch_generated': False, 'val_passed': False, 'rescan_passed': False, 'duration': 15.0, 'hallucination': False, 'fix_accuracy': False, 'success': True},
]

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
hallucination_rate = (sum(1 for r in vuln_results if r["hallucination"]) / patched * 100) if patched else 0

total_safe = len(safe_results)
false_positives = sum(1 for r in safe_results if r["detected"])
fp_rate = (false_positives / total_safe * 100) if total_safe else 0

avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0

report = f"""# BugInsight Benchmark Report

Date: {time.strftime('%Y-%m-%d')}

## Summary

Total Vulnerabilities: {total_vuln}
Detected: {detected}
Patched: {patched}
Validated: {validated}
Rescan Passed: {rescan_passed}

Detection Rate: {det_rate:.1f}%
Patch Generation Rate: {patch_rate:.1f}%
Validation Pass Rate: {val_rate:.1f}%
Fix Accuracy: {fix_acc:.1f}%
Hallucination Rate: {hallucination_rate:.1f}%
False Positive Rate: {fp_rate:.1f}%
Average Runtime: {avg_duration:.1f}s

## Detailed Results - Vulnerable Files

| Vulnerability | Detect | Patch | Validate | Rescan | Result | Runtime |
|--------------|---------|--------|----------|--------|--------|---------|
"""

for r in vuln_results:
    report += f"| `{os.path.basename(r['file'])}` | {'✅' if r['detected'] else '❌'} | {'✅' if r['patch_generated'] else '❌'} | {'✅' if r['val_passed'] else '❌'} | {'✅' if r['rescan_passed'] else '❌'} | {'SUCCESS' if r['success'] else 'FAIL'} | {r['duration']:.1f}s |\n"

report += f"""
## Detailed Results - Safe Files

| File | False Positive Detected | Result | Runtime |
|--------------|---------|--------|---------|
"""

for r in safe_results:
    report += f"| `{os.path.basename(r['file'])}` | {'❌ YES (Bad)' if r['detected'] else '✅ NO (Good)'} | {'SUCCESS' if r['success'] else 'FAIL'} | {r['duration']:.1f}s |\n"

os.makedirs("outputs", exist_ok=True)
with open("outputs/benchmark_report.md", "w", encoding="utf-8") as f:
    f.write(report)

print("Report saved to outputs/benchmark_report.md")
