# BugInsight Benchmark Report
Date: 2026-06-06
Model: `qwen2.5-coder:7b`
Prompt Version: `V1`

## Summary

Total Vulnerabilities: 8
Detection Rate: 75.0%
Patch Generation Rate: 75.0%
Validation Pass Rate: 25.0%
Fix Accuracy: 0.0%
False Positive Rate: 0.0%
Average Runtime: 143.6s

## Failure Categories
- Undefined Variable: 4

## Detailed Results - Vulnerable Files

| Vulnerability | Detect | Patch | Validate | Rescan | First Gate Failed | Category |
|--------------|---------|--------|----------|--------|-------------------|----------|
| `sql_injection.py` | ✅ | ✅ | ❌ | ❌ | functional_preservation | Undefined Variable |
| `command_injection.py` | ✅ | ✅ | ✅ | ❌ | - | - |
| `path_traversal.py` | ❌ | ❌ | ❌ | ❌ | - | - |
| `hardcoded_secret.py` | ✅ | ✅ | ❌ | ❌ | functional_preservation | Undefined Variable |
| `weak_crypto.py` | ✅ | ✅ | ❌ | ❌ | functional_preservation | Undefined Variable |
| `xss.py` | ❌ | ❌ | ❌ | ❌ | - | - |
| `unsafe_deserialization.py` | ✅ | ✅ | ✅ | ❌ | - | - |
| `multi_vulnerability.py` | ✅ | ✅ | ❌ | ❌ | functional_preservation | Undefined Variable |

## Detailed Results - Safe Files

| File | False Positive Detected | Result | Runtime |
|--------------|---------|--------|---------|
| `safe_auth.py` | ✅ NO (Good) | SUCCESS | 28.6s |
| `safe_api.py` | ✅ NO (Good) | SUCCESS | 27.6s |
| `safe_file_io.py` | ✅ NO (Good) | SUCCESS | 27.7s |
| `safe_crypto.py` | ✅ NO (Good) | SUCCESS | 27.5s |
| `safe_web.py` | ✅ NO (Good) | SUCCESS | 27.2s |
| `fixed_sqli.py` | ✅ NO (Good) | SUCCESS | 27.7s |
| `fixed_path_traversal.py` | ✅ NO (Good) | SUCCESS | 28.7s |
