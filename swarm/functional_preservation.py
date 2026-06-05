import ast
from typing import Set, Tuple, List, Dict

def extract_ast_metadata(code: str) -> Dict[str, Set[str]]:
    """Extracts top-level functions, classes, and imports from python code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"functions": set(), "classes": set(), "imports": set()}
        
    functions = set()
    classes = set()
    imports = set()
    
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
                
    return {
        "functions": functions,
        "classes": classes,
        "imports": imports
    }

def check_functional_preservation(original_code: str, patched_code: str, vulnerable_code: str = "") -> Tuple[bool, str]:
    """
    Checks if the patched code preserves the functional structure of the original code.
    Returns (is_valid, rejection_reason_or_warning)
    """
    # 1. Syntax Check
    try:
        ast.parse(patched_code)
    except SyntaxError as e:
        return False, f"REJECTED\n\nFunctional Preservation Failed: SyntaxError in generated patch.\n{str(e)}"
        
    # 2. Size Ratio Check
    if len(original_code) > 0:
        ratio = len(patched_code) / len(original_code)
        if ratio < 0.25 or ratio > 4.0:
            return False, f"REJECTED\n\nFunctional Preservation Failed: File size anomaly.\nOriginal size: {len(original_code)} bytes, Patched size: {len(patched_code)} bytes (Ratio: {ratio:.2f}). Patch likely hallucinated an entirely different file."
            
    # 3. AST Extraction
    orig_meta = extract_ast_metadata(original_code)
    patch_meta = extract_ast_metadata(patched_code)
    
    # 4. Function / Class Preservation Check
    missing_functions = orig_meta["functions"] - patch_meta["functions"]
    introduced_functions = patch_meta["functions"] - orig_meta["functions"]
    
    missing_classes = orig_meta["classes"] - patch_meta["classes"]
    introduced_classes = patch_meta["classes"] - orig_meta["classes"]
    
    if missing_functions or missing_classes:
        msg = "REJECTED\n\nFunctional Preservation Failed.\n\nThe generated patch removes or replaces existing application structures rather than remediating the identified vulnerability.\n"
        if missing_functions:
            msg += "\nMissing Functions:\n" + "\n".join(f"- {f}" for f in missing_functions)
        if introduced_functions:
            msg += "\n\nIntroduced Functions:\n" + "\n".join(f"- {f}" for f in introduced_functions)
        if missing_classes:
            msg += "\n\nMissing Classes:\n" + "\n".join(f"- {c}" for c in missing_classes)
        msg += "\n\nThis patch cannot be safely applied."
        return False, msg
        
    # 5. Import Analysis (Warning only, we don't reject but we can prepend to reason)
    new_imports = patch_meta["imports"] - orig_meta["imports"]
    warning_msg = ""
    if len(new_imports) > 3:
        warning_msg = f"Validation Warning: Patch introduces {len(new_imports)} new dependencies ({', '.join(new_imports)}).\n\n"
        
    # 6. Line-Level Vulnerability Check
    if vulnerable_code and len(vulnerable_code.strip()) > 10:
        # Strip all whitespace for a robust comparison
        vuln_stripped = "".join(vulnerable_code.split())
        patch_stripped = "".join(patched_code.split())
        if vuln_stripped in patch_stripped:
            return False, "REJECTED\n\nFunctional Preservation Failed.\n\nThe exact vulnerable code block is still present in the patched file without modification."
            
    return True, warning_msg
