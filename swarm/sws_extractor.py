import ast
import re
from typing import Dict, Any, Optional, Tuple

class TargetNodeVisitor(ast.NodeVisitor):
    def __init__(self, target_line: int):
        self.target_line = target_line
        self.best_match: Optional[ast.AST] = None
        # Priority: 1. FunctionDef/AsyncFunctionDef, 2. ClassDef
        # If we find a function/method containing the line, it is preferred.
        
    def visit_FunctionDef(self, node):
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if node.lineno <= self.target_line <= node.end_lineno:
                self.best_match = node
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if node.lineno <= self.target_line <= node.end_lineno:
                self.best_match = node
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        # We only set best_match to class if we haven't found a function inside it yet
        # Since we do generic_visit AFTER checking class, a nested function will overwrite this.
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if node.lineno <= self.target_line <= node.end_lineno:
                if not self.best_match:
                    self.best_match = node
        self.generic_visit(node)

def extract_surgical_window(file_content: str, target_line: int, window_size: int = 15) -> Dict[str, Any]:
    """
    Extracts the AST node containing the target line.
    Returns:
        {
            "function_name": str,
            "start_line": int, (0-indexed)
            "end_line": int, (0-indexed, exclusive)
            "window_code": str
        }
    """
    lines = file_content.splitlines()
    total_lines = len(lines)
    
    try:
        tree = ast.parse(file_content)
        visitor = TargetNodeVisitor(target_line)
        visitor.visit(tree)
        
        match = visitor.best_match
        if match and hasattr(match, 'lineno') and hasattr(match, 'end_lineno'):
            # Convert 1-based lineno to 0-based indices
            # Include decorators if present
            start_idx = match.lineno - 1
            if hasattr(match, 'decorator_list') and match.decorator_list:
                start_idx = min(start_idx, match.decorator_list[0].lineno - 1)
                
            end_idx = match.end_lineno
            
            window_lines = lines[start_idx:end_idx]
            name = getattr(match, 'name', 'Unknown')
            
            return {
                "function_name": name,
                "start_line": start_idx,
                "end_line": end_idx,
                "window_code": "\n".join(window_lines)
            }
    except SyntaxError:
        pass # Fallback if file is completely broken
        
    # Fallback to fixed window
    start_idx = max(0, target_line - 1 - window_size)
    end_idx = min(total_lines, target_line - 1 + window_size + 1)
    window_lines = lines[start_idx:end_idx]
    
    return {
        "function_name": "Fallback Window",
        "start_line": start_idx,
        "end_line": end_idx,
        "window_code": "\n".join(window_lines)
    }

def parse_and_splice(original_file_content: str, llm_response: str, start_idx: int, end_idx: int) -> Tuple[str, str, str]:
    """
    Returns: (spliced_file_content, fixed_window, new_imports)
    Raises SyntaxError if the fixed_window itself fails to parse.
    """
    # 1. Extract the fixed window
    window_match = re.search(r"<fixed_window>(.*?)</fixed_window>", llm_response, re.DOTALL)
    if not window_match:
        # Fallback to basic code block stripping
        clean_code = re.sub(r"```[a-zA-Z]*\n|```", "", llm_response).strip("\r\n")
        fixed_window = clean_code
    else:
        fixed_window = window_match.group(1).strip("\r\n")
        
    # AST Check immediately on the fixed window
    try:
        # We wrap in a try-except to ensure we fail fast if the LLM output is broken Python
        ast.parse(fixed_window)
    except SyntaxError as e:
        raise SyntaxError(f"AST parsing failed on the generated window: {e}")
        
    # 2. Extract new imports
    imports_match = re.search(r"<required_imports>(.*?)</required_imports>", llm_response, re.DOTALL)
    new_imports = []
    if imports_match:
        imports_text = imports_match.group(1).strip()
        new_imports = [line.strip() for line in imports_text.splitlines() if line.strip()]
        
    # 3. Splice the window back into the file
    lines = original_file_content.splitlines()
    lines[start_idx:end_idx] = fixed_window.splitlines()
    
    # 4. Safely inject new imports at the top of the file
    if new_imports:
        import_lines = []
        for imp in new_imports:
            if imp not in original_file_content:
                import_lines.append(imp)
        if import_lines:
            lines = import_lines + [""] + lines
            
    return "\n".join(lines), fixed_window, "\n".join(new_imports)
