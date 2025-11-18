import os
import ast
import json
import re
import difflib
import argparse
import subprocess
from typing import List, Dict, Optional, Set, Tuple, Any
from pydantic import BaseModel
from llm_provider import (
    responses_parse,
    responses_create,
    MODEL_GPT4O,
    MODEL_GPT4O_MINI,
)
from advanced_analysis import build_cfg_from_source, analyze_data_flow, analyze_file
from type_inference_engine import (
    TypeInferenceEngine,
    InferenceMode,
    TypeInference
)
import time

# ------------------------------
# Code Analysis & Static Inference
# ------------------------------

class CodeAnalyzer:
    """
    Loads a Python code base and infers variable types using AST and simple heuristics.
    """
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.modules: Dict[str, ast.AST] = {}  # file_path -> AST

    def load_codebase(self):
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            source = f.read()
                        tree = ast.parse(source, filename=file_path)
                        self.modules[file_path] = tree
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")

    def infer_types(self) -> Dict[str, Dict[str, str]]:
        """
        Infer types by traversing each AST and using heuristics.
        Returns a dict: file_path -> {variable_name: inferred_type}
        """
        inferred_types: Dict[str, Dict[str, str]] = {}
        for file_path, tree in self.modules.items():
            file_types: Dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        inferred = self._infer_type_from_value(node.value)
                        if inferred:
                            file_types[var_name] = inferred
            inferred_types[file_path] = file_types
        return inferred_types

    def _infer_type_from_value(self, value_node: ast.AST) -> Optional[str]:
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, int):
                return 'int'
            elif isinstance(value_node.value, float):
                return 'float'
            elif isinstance(value_node.value, str):
                return 'str'
            elif isinstance(value_node.value, bool):
                return 'bool'
            elif value_node.value is None:
                return 'None'
        elif isinstance(value_node, ast.List):
            return 'list'
        elif isinstance(value_node, ast.Dict):
            return 'dict'
        return None

# ------------------------------
# Patch Analysis
# ------------------------------

class PatchAnalyzer:
    """
    Processes a unified diff patch file.
    In this revised version, we preserve the full diff context (including headers, hunk markers, etc.)
    and avoid stripping out any context that might be necessary for the LLM to produce a valid patch.
    """
    def __init__(self, patch_file: str):
        self.patch_file = patch_file
        self.patch_content = self._load_patch() if patch_file else ""
        self.cleaned_patch = ""  
    def _load_patch(self) -> str:
        with open(self.patch_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def sanitize_patch(self, patch_content: str) -> str:
        """
        Sanitizes and normalizes a patch to ensure it's in valid unified diff format.
        Handles common issues like inconsistent line endings, improper headers, and corruption.
        """
        if not patch_content:
            return ""
            
        # Split into lines, preserving line endings for analysis
        lines = patch_content.splitlines(True)
        sanitized_lines = []
        
        # Track the state of the parser
        in_header = False
        in_hunk = False
        has_file_header = False
        filename_a = None
        filename_b = None
        
        # Process line by line
        i = 0
        while i < len(lines):
            line = lines[i]
            line = line.rstrip('\r\n') + '\n'  # Normalize line endings
            
            # Handle diff header
            if line.startswith('diff --git '):
                # Start of a new file diff
                in_header = True
                in_hunk = False
                has_file_header = True
                
                # Parse filenames from diff header
                parts = line.split()
                if len(parts) >= 4:
                    filename_a = parts[2].lstrip('a/')
                    filename_b = parts[3].lstrip('b/')
                
                sanitized_lines.append(line)
            
            # Handle --- and +++ lines
            elif line.startswith('--- ') or line.startswith('+++ '):
                if not has_file_header:
                    # If we don't have a diff header yet, create one
                    if line.startswith('--- '):
                        filename_a = line[4:].strip()
                        sanitized_lines.insert(0, f"diff --git a/{filename_a} b/{filename_a}\n")
                        has_file_header = True
                
                sanitized_lines.append(line)
            
            # Handle hunk headers
            elif line.startswith('@@'):
                in_header = False
                in_hunk = True
                
                # Validate hunk header format: @@ -start,count +start,count @@
                hunk_parts = line.split('@@')
                if len(hunk_parts) >= 2:
                    hunk_info = hunk_parts[1].strip()
                    if not (hunk_info.startswith('-') and '+' in hunk_info):
                        # Fix malformed hunk header
                        matches = re.findall(r'(-\d+(?:,\d+)?) (\+\d+(?:,\d+)?)', hunk_info)
                        if matches:
                            old_range, new_range = matches[0]
                            line = f"@@ {old_range} {new_range} @@\n"
                
                sanitized_lines.append(line)
            
            # Handle context, addition and deletion lines
            elif in_hunk and (line.startswith(' ') or line.startswith('+') or line.startswith('-')):
                sanitized_lines.append(line)
            
            # Handle binary files and other special headers
            elif any(line.startswith(prefix) for prefix in 
                    ['Binary files ', 'GIT binary patch', 'index ', 'new file mode', 
                     'deleted file mode', 'old mode', 'new mode']):
                sanitized_lines.append(line)
            
            # Skip any malformed or corrupt lines that would break the patch
            elif not line.strip() or line.strip().startswith('#'):
                # Keep empty lines and comments
                sanitized_lines.append(line)
            else:
                # If we're in a context where we should keep this line, do so
                if in_header or not has_file_header:
                    sanitized_lines.append(line)
            
            i += 1
        
        # Ensure proper ending
        if sanitized_lines and not sanitized_lines[-1].endswith('\n'):
            sanitized_lines[-1] += '\n'
            
        # Final verification - make sure we have at least one file header and hunk
        has_file_header = any(line.startswith('diff --git ') for line in sanitized_lines)
        has_hunk = any(line.startswith('@@') for line in sanitized_lines)
        
        if not has_file_header or not has_hunk:
            print("Warning: Sanitized patch may be incomplete - missing file header or hunk markers")
            
        return ''.join(sanitized_lines)

    def validate_patch_format(self) -> bool:
        """
        Ensures the patch contains the expected unified diff markers.
        Returns True if the patch format appears valid.
        """
        # Check basic markers
        content_to_check = self.cleaned_patch if self.cleaned_patch else self.patch_content
        basic_format = any(line.startswith(('diff --git', '+++', '---', '@@')) 
                          for line in content_to_check.splitlines())
        
        if not basic_format:
            return False
            
        # Additional validation for patch structure
        lines = content_to_check.splitlines()
        has_file_header = False
        has_file_paths = False
        has_hunk = False
        
        for line in lines:
            if line.startswith('diff --git'):
                has_file_header = True
            elif line.startswith('--- ') or line.startswith('+++ '):
                has_file_paths = True
            elif line.startswith('@@'):
                has_hunk = True
                
        return has_file_header and has_file_paths and has_hunk

    def validate_added_code_syntax(self) -> Dict[str, List[str]]:
        """
        Validates the syntax of the added code in each modified file block.
        Instead of extracting only added lines globally, it processes each diff block.
        """
        errors: Dict[str, List[str]] = {}
        current_file = None
        current_added = []
        
        content_to_check = self.cleaned_patch if self.cleaned_patch else self.patch_content
        for line in content_to_check.splitlines():
            if line.startswith('diff --git'):
                if current_file and current_added:
                    try:
                        ast.parse("\n".join(current_added))
                    except Exception as e:
                        errors.setdefault(current_file, []).append(str(e))
                parts = line.split()
                # Expect the file path in the last part after "b/"
                current_file = parts[-1][2:] if parts[-1].startswith("b/") else parts[-1]
                current_added = []
            elif line.startswith('@@'):
                continue  # hunk header
            elif line.startswith('+') and not line.startswith('+++'):
                current_added.append(line[1:])
        
        if current_file and current_added:
            try:
                ast.parse("\n".join(current_added))
            except Exception as e:
                errors.setdefault(current_file, []).append(str(e))
                
        return errors
        
    def extract_file_patches(self) -> Dict[str, str]:
        """
        Extracts individual file patches from a multi-file diff.
        Returns a dictionary mapping file paths to their individual patches.
        """
        file_patches = {}
        current_file = None
        current_patch_lines = []
        
        content_to_check = self.cleaned_patch if self.cleaned_patch else self.patch_content
        lines = content_to_check.splitlines(True)  # Keep line endings
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('diff --git'):
                # Start of a new file diff
                if current_file and current_patch_lines:
                    file_patches[current_file] = ''.join(current_patch_lines)
                
                current_patch_lines = [line]
                parts = line.split()
                if len(parts) >= 4:
                    current_file = parts[3].lstrip('b/')
                else:
                    current_file = f"file_{len(file_patches)}"
            elif current_file:
                current_patch_lines.append(line)
            
            i += 1
        
        # Add the last file
        if current_file and current_patch_lines:
            file_patches[current_file] = ''.join(current_patch_lines)
            
        return file_patches
        
    def get_most_important_patch(self) -> str:
        """
        Returns the most important/relevant patch from a multi-file diff.
        This is determined by the complexity of changes (number of chunks and lines modified).
        """
        content_to_use = self.cleaned_patch if self.cleaned_patch else self.patch_content
        
        file_patches = self.extract_file_patches()
        
        if not file_patches:
            return content_to_use
            
        if len(file_patches) == 1:
            return next(iter(file_patches.values()))
            
        # Score each patch by the amount of changes
        patch_scores = {}
        for file_path, patch in file_patches.items():
            lines = patch.splitlines()
            hunk_count = sum(1 for line in lines if line.startswith('@@'))
            added_lines = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
            deleted_lines = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
            is_python = file_path.endswith('.py')
            
            # Calculate score based on changes and file type
            score = (hunk_count * 10) + added_lines + deleted_lines
            if is_python:
                score *= 1.5  
                
            patch_scores[file_path] = score
        
        # Get file with highest score
        if patch_scores:
            most_important_file = max(patch_scores.items(), key=lambda x: x[1])[0]
            return file_patches[most_important_file]
            
        return content_to_use

# ------------------------------
# Variable Extraction from Patch
# ------------------------------

class VariablesResult(BaseModel):
    variables: List[str]

class PatchVariableExtractor:
    """
    Uses an LLM to identify variables in a patch that would benefit from type annotations.
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def extract_variables(self, patch_code: str, is_last_retry: bool = False) -> List[str]:
        """
        Analyzes a patch and returns a list of variable names that should have type annotations.
        """
        prompt = (
            "You are a Python expert. Given the following unified diff patch, identify "
            "variables that would benefit from explicit type annotations. "
            "Focus on function parameters, return values, and important variables "
            "that might be ambiguous without type hints.\n\n"
            f"Patch to analyze:\n{patch_code}\n\n"
            "Return only a list of variable names that need type annotations."
        )
        messages = [
            {"role": "system", "content": "You are a helpful coding expert who analyzes Python code."},
            {"role": "user", "content": prompt}
        ]
        try:
            model = MODEL_GPT4O if is_last_retry else MODEL_GPT4O_MINI
            print(f"Using model {model} for variable extraction")
            input_chars = sum(len(m.get("content", "")) for m in messages)
            t0 = time.time()
            completion = responses_parse(
                model=model,
                input=messages,
                text_format=VariablesResult,
                max_output_tokens=300,
            )
            seconds = time.time() - t0
            try:
                output_chars = len(completion.output_parsed.model_dump_json())  # type: ignore[attr-defined]
            except Exception:
                output_chars = len(json.dumps(getattr(completion, "output_parsed", {})))
            self.calls.append({
                "stage": "var_extraction",
                "model": model,
                "input_chars": input_chars,
                "output_chars": output_chars,
                "seconds": seconds,
            })
            return completion.output_parsed.variables
        except Exception as e:
            print(f"LLM variable extraction error: {e}")
            return []

# ------------------------------
# Advanced Code Retrieval and Context Flattening
# ------------------------------

class EnhancedCodeRetriever:
    """
    Retrieves and flattens related code context using advanced AST/CFG/data-flow analysis.
    """
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.file_cfgs = {}  
        
    def _analyze_file_if_needed(self, file_path: str):
        """Analyze a file if not already in the cache."""
        if file_path not in self.file_cfgs:
            try:
                self.file_cfgs[file_path] = analyze_file(file_path)
            except Exception as e:
                print(f"Error analyzing file {file_path}: {e}")
                self.file_cfgs[file_path] = {}
    
    def _find_files_with_variable(self, variable: str) -> List[str]:
        """Find files in the codebase that contain the target variable."""
        relevant_files = []
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if re.search(r'\b' + re.escape(variable) + r'\b', content):
                            relevant_files.append(file_path)
                    except Exception:
                        continue
        return relevant_files
    
    def _extract_function_with_variable(self, file_path: str, variable: str) -> List[Tuple[str, str]]:
        """Extract function definitions that contain the target variable."""
        functions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if variable is used in this function
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Name) and subnode.id == variable:
                            # Found the variable in this function
                            func_source = ast.get_source_segment(source, node)
                            if func_source:
                                functions.append((node.name, func_source))
                            break
        except Exception as e:
            print(f"Error extracting functions from {file_path}: {e}")
        return functions
    
    def _extract_data_flow_info(self, file_path: str, variable: str) -> str:
        """Extract data flow information for the variable."""
        self._analyze_file_if_needed(file_path)
        if file_path not in self.file_cfgs:
            return ""
        
        file_analysis = self.file_cfgs[file_path]
        results = []
        
        for func_name, data in file_analysis.items():
            # Check if variable is in any of the in_sets or out_sets
            var_in_function = False
            for node, var_set in data["in_sets"].items():
                if variable in var_set:
                    var_in_function = True
                    break
            if not var_in_function:
                for node, var_set in data["out_sets"].items():
                    if variable in var_set:
                        var_in_function = True
                        break
            
            if var_in_function:
                results.append(f"Data flow for variable '{variable}' in function '{func_name}':")
                
                # Find nodes where the variable is defined or used
                for node, attrs in data["cfg"].nodes(data=True):
                    if variable in data["in_sets"].get(node, set()) or variable in data["out_sets"].get(node, set()):
                        stmts = attrs.get("stmts", [])
                        if any(variable in stmt for stmt in stmts):
                            results.append(f"  Node {node} ({attrs.get('label', 'unknown')}): {stmts}")
        
        return "\n".join(results)
    
    def retrieve_context_for_variables(self, variables: List[str]) -> Dict[str, str]:
        """
        Retrieve rich context for multiple variables using advanced analysis.
        Returns a dict mapping variable name to its context.
        """
        contexts = {}
        
        for variable in variables:
            relevant_files = self._find_files_with_variable(variable)
            var_context = []
            
            for file_path in relevant_files:
                # Extract functions containing the variable
                functions = self._extract_function_with_variable(file_path, variable)
                for func_name, func_source in functions:
                    var_context.append(f"Function '{func_name}' in {file_path}:\n{func_source}")
                
                # Extract data flow information
                data_flow_info = self._extract_data_flow_info(file_path, variable)
                if data_flow_info:
                    var_context.append(data_flow_info)
            
            # Intelligently distill context
            distilled_context = self._distill_context(variable, var_context)
            contexts[variable] = distilled_context
        
        return contexts
        
    def _distill_context(self, variable: str, context_items: List[str]) -> str:
        """
        Intelligently distill context for type inference by extracting and prioritizing 
        type-relevant information for the target variable.
        """
        if not context_items:
            return ""
        
        # Regular expressions for identifying type-relevant patterns
        var_pattern = re.compile(r'\b' + re.escape(variable) + r'\b')
        
        # Collect relevant lines with priority levels
        high_priority = []  # Type hints, assignments, function signatures
        medium_priority = []  # Returns, method calls, comparisons
        low_priority = []  # Other variable usages
        
        for item in context_items:
            # Preserve function header information for context
            header = ""
            if item.startswith("Function '"):
                lines = item.splitlines()
                if len(lines) > 0:
                    header = lines[0]
                    high_priority.append(header)
                if len(lines) > 1 and lines[1].strip().startswith("def "):
                    high_priority.append(lines[1])
            
            for line in item.splitlines():
                # Skip lines without the variable
                if not var_pattern.search(line):
                    continue
                
                line_stripped = line.strip()
                
                # High priority: Type hints, assignments, function parameters
                if (f"{variable}:" in line or 
                    f"{variable} =" in line or 
                    (line_stripped.startswith("def ") and variable in line)):
                    high_priority.append(line)
                
                # Medium priority: Return statements, method calls, operators
                elif (line_stripped.startswith("return ") or 
                      f"{variable}." in line or
                      any(op in line for op in [" + ", " - ", " * ", " / ", " == ", " != "])):
                    medium_priority.append(line)
                
                # Low priority: Other usages
                else:
                    low_priority.append(line)
        
        # Remove duplicates while preserving order
        def deduplicate(lines):
            seen = set()
            result = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    result.append(line)
            return result
        
        high_priority = deduplicate(high_priority)
        medium_priority = deduplicate(medium_priority)
        low_priority = deduplicate(low_priority)
        
        # Build final context with priority
        all_lines = high_priority + medium_priority + low_priority
        final_context = "\n".join(all_lines)
        
        # If still too long, keep only high and medium priority
        if len(final_context) > 5000:
            final_context = "\n".join(high_priority + medium_priority)
            
            # If still too long, keep only high priority
            if len(final_context) > 5000:
                final_context = "\n".join(high_priority)
                
            # Add a note about truncation
            if len(all_lines) > len(high_priority) + len(medium_priority):
                final_context += f"\n\n... ({len(low_priority)} additional usages omitted)"
        
        return final_context

# ------------------------------
# Type Inference with LLM
# ------------------------------

class TypeInferenceResult(BaseModel):
    suggested_type: str

class VariableTypeInfo(BaseModel):
    variable: str
    type: str

class StructuredTypeResult(BaseModel):
    variable_types: List[VariableTypeInfo]

class EnhancedLLMTypeInferencer:
    """
    Uses an LLM to infer types for multiple variables based on their contexts.
    """
    def infer_types(self, variable_contexts: Dict[str, str], is_last_retry: bool = False) -> List[Dict[str, str]]:
        """
        Infer types for multiple variables.
        Args:
            variable_contexts: Dict mapping variable names to their contexts
            
        Returns:
            List of dicts with 'variable' and 'type' keys
        """
        # If no variables to process, return empty list
        if not variable_contexts:
            return []
            
        # Format the context for each variable
        formatted_contexts = []
        for var, context in variable_contexts.items():
            formatted_contexts.append(f"Variable: {var}\nContext:\n{context}\n")
        
        context_str = "\n===\n".join(formatted_contexts)
        
        # Try the structured approach first
        try:
            result = self._infer_types_structured(context_str, is_last_retry)
            if result:
                return result
        except Exception as e:
            print(f"Structured type inference failed: {e}")
            # Continue to fallback methods
        
        # Fallback to individual inference
        return self._infer_types_individually(variable_contexts, is_last_retry)
    
    def _infer_types_structured(self, context_str: str, is_last_retry: bool) -> List[Dict[str, str]]:
        """
        Attempt to infer types using structured output format.
        """
        prompt = (
            "You are a Python type inference expert. For each variable below, analyze its context "
            "and suggest the most appropriate Python type annotation. Consider both usage patterns "
            "and the surrounding code.\n\n"
            f"{context_str}\n\n"
            "For each variable, determine the most appropriate Python type annotation. "
            "Return your results in JSON format with the following structure:\n"
            "{\n"
            "  \"variable_types\": [\n"
            "    { \"variable\": \"variable_name\", \"type\": \"inferred_type\" },\n"
            "    { \"variable\": \"another_var\", \"type\": \"another_type\" }\n"
            "  ]\n"
            "}"
        )
        
        messages = [
            {"role": "system", "content": "You are a Python expert specialized in type inference."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            model = MODEL_GPT4O if is_last_retry else MODEL_GPT4O_MINI
            print(f"Using model {model} for structured type inference")
            parsed = responses_parse(
                model=model,
                input=messages,
                text_format=StructuredTypeResult,
                max_output_tokens=1200,
            )
            result = []
            for item in parsed.output_parsed.variable_types:
                result.append({"variable": item.variable, "type": item.type})
            return result
        except Exception as e:
            print(f"JSON structured type inference error: {e}")
            return []
    
    def _infer_types_individually(self, variable_contexts: Dict[str, str], is_last_retry: bool) -> List[Dict[str, str]]:
        """
        Fallback method to infer types for each variable individually.
        """
        results = []
        for var, context in variable_contexts.items():
            var_type = self._infer_single_type(var, context, is_last_retry)
            results.append({"variable": var, "type": var_type})
        return results
    
    def _infer_single_type(self, variable: str, context: str, is_last_retry: bool) -> str:
        """
        Infer type for a single variable.
        """
        prompt = (
            f"Based on the following context, what is the most appropriate Python type for the variable '{variable}'?\n\n"
            f"Context:\n{context}\n\n"
            f"Respond with only the Python type (e.g., 'int', 'str', 'List[str]', etc.)."
        )
        
        messages = [
            {"role": "system", "content": "You are a Python type inference expert."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            model = MODEL_GPT4O if is_last_retry else MODEL_GPT4O_MINI
            response = responses_create(
                model=model,
                input=messages,
                max_output_tokens=50,
            )
            inferred_type = response.output_text.strip()
            
            # Clean up the response to extract just the type
            inferred_type = re.sub(r'^[\'"`]|[\'"`]$', '', inferred_type)  # Remove quotes
            inferred_type = re.sub(r'^The type .+ is ', '', inferred_type)  # Remove prefix
            inferred_type = inferred_type.split('\n')[0].strip()  # Take first line only
            
            return inferred_type if inferred_type else "Any"
            
        except Exception as e:
            print(f"Single type inference error for {variable}: {e}")
            return "Any"

# ------------------------------
# Patch Rewriting with LLM
# ------------------------------

class PatchFileResult(BaseModel):
    explanation: str
    patch_in_unified_diff_format: str

class LLMRewriter:
    """
    Uses an LLM to rewrite a patch with enhanced type information.
    """
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def rewrite_patch(self, patch_code: str, context_info: str, 
                      type_info: Optional[str] = None, is_last_retry: bool = False) -> str:
        # First, try to use the original patch directly with the LLM
        try:
            # Use original patch without sanitization
            result = self._attempt_rewrite(patch_code, context_info, is_last_retry)
            
            # Basic validation to check if it's a valid patch
            if any(line.startswith(('diff --git', '+++', '---', '@@')) for line in result.splitlines()):
                return result
                
            print("LLM result doesn't appear to be a valid patch format. Will try sanitization...")
        except Exception as e:
            print(f"Initial patch rewriting with original format failed: {e}")
            
        # Sanitization fallback - only used if first attempt failed
        try:
            patch_analyzer = PatchAnalyzer("")  # Empty file path since we're providing content directly
            sanitized_patch = patch_analyzer.sanitize_patch(patch_code)
            
            # Extract the most important file patch if it's a multi-file patch
            important_patch = patch_analyzer.get_most_important_patch()
            
            # Try again with the sanitized patch
            result = self._attempt_rewrite(important_patch, context_info, is_last_retry)
            
            # Verify the basic structure is intact
            if any(line.startswith(('diff --git', '+++', '---', '@@')) for line in result.splitlines()):
                return result
                
            # If we still don't have a valid result, fall back to original
            print("All patch rewriting attempts failed. Returning original patch.")
            return patch_code
            
        except Exception as e:
            print(f"Sanitized patch rewriting also failed: {e}")
            return patch_code
        
    def _attempt_rewrite(self, patch_content: str, context_info: str, is_last_retry: bool) -> str:
        """Helper method to attempt a patch rewrite with the LLM."""
        prompt = (
            "You are a world-class Python developer. You will be given an original unified diff patch "
            "and additional context. Your task is to produce a revised patch that is syntactically correct, "
            "applies cleanly, and improves type clarity and correctness.\n\n"
            "Constraints:\n"
            "- Output ONLY a unified diff patch. No prose, no backticks.\n"
            "- Preserve the original intent of the patch, but improve types and fix issues if needed.\n"
            "- Ensure the patch uses correct unified diff headers and hunk markers.\n"
            "- Only include diffs for files that already exist in the repository. Do NOT add new files like reproduce.py, manage.py, generated build artifacts, or unrelated directories.\n"
            "- Do NOT include test run logs, HTML, or any non-diff content.\n"
            "- Ensure every line in the patch ends with a newline.\n"
            "- Ensure each hunk header (e.g., @@ -a,b +c,d @@) is consistent with the following hunk lines.\n"
            "- If there are mypy or apply errors referenced below, fix the root cause.\n\n"
            f"Additional Context:\n{context_info}\n\n"
            "Please Return only the revised patch in valid unified diff format."
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful coding expert who writes code based solely on instructions."},
            {"role": "user", "content": prompt}
        ]
        
        model = MODEL_GPT4O if is_last_retry else MODEL_GPT4O_MINI
        print(f"Using model {model} for patch rewriting")
        input_chars = sum(len(m.get("content", "")) for m in messages)
        t0 = time.time()
        parsed = responses_parse(
            model=model,
            input=messages,
            text_format=PatchFileResult,
            max_output_tokens=4096,
        )
        seconds = time.time() - t0
        try:
            output_chars = len(parsed.output_parsed.model_dump_json())  # type: ignore[attr-defined]
        except Exception:
            output_chars = len(json.dumps(getattr(parsed, "output_parsed", {})))
        self.calls.append({
            "stage": "rewriter",
            "model": model,
            "input_chars": input_chars,
            "output_chars": output_chars,
            "seconds": seconds,
        })
        
        raw_patch = parsed.output_parsed.patch_in_unified_diff_format
        return self._postprocess_llm_patch(raw_patch)

    def _postprocess_llm_patch(self, text: str) -> str:
        """Extract unified diff blocks conservatively without over-stripping.
        - Remove surrounding code fences if present.
        - If multiple diff blocks exist, keep all starting from each 'diff --git'.
        - Preserve all diff lines (index/new file mode/binary/etc.).
        - Normalize newlines and ensure trailing newline.
        """
        if not text:
            return text
        # Strip common code fences while keeping inner content intact
        code_fence = re.search(r"```(?:diff)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if code_fence:
            text = code_fence.group(1)
        # If we have explicit diff headers, keep everything from the first one onward, preserving multiple blocks
        if "diff --git " in text:
            segments = []
            # Find all block starts
            starts = [m.start() for m in re.finditer(r"^diff --git ", text, flags=re.MULTILINE)]
            for i, s in enumerate(starts):
                e = starts[i+1] if i + 1 < len(starts) else len(text)
                segments.append(text[s:e])
            text = "".join(segments)
        # Normalize newlines and ensure trailing newline
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.endswith("\n"):
            text += "\n"
        return text

# ------------------------------
# Syntax Validation
# ------------------------------

def validate_syntax_of_revised_patch(patch_code: str) -> bool:
    """
    Validates that the revised patch's added code (ignoring diff markers) is syntactically correct.
    """
    if not any(line.startswith(('diff --git', '+++', '---', '@@')) for line in patch_code.splitlines()):
        print("Warning: Patch does not appear to be in proper diff format")
        return False

    code_lines = []
    in_hunk = False

    for line in patch_code.splitlines():
        if line.startswith('@@'):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith('+') and not line.startswith('+++'):
            code_line = line[1:]
            if not code_line.strip() or code_line.strip().startswith('#'):
                continue
            code_lines.append(code_line)

    if not code_lines:
        return True

    try:
        ast.parse("\n".join(code_lines))
        return True
    except Exception as e:
        print(f"Syntax validation error: {e}")
        return False

# ------------------------------
# Patch Application and Validation
# ------------------------------

def apply_and_validate_patch(patch_file: str, codebase_dir: str, max_retries: int = 3) -> Tuple[bool, str]:
    """
    Attempts to apply a patch, with intelligent fallback and repair strategies if it fails.
    Returns (success_status, message).
    """
    if not os.path.exists(patch_file):
        error_msg = f"Error: Patch file does not exist: {patch_file}"
        print(error_msg)
        return False, error_msg

    if os.path.getsize(patch_file) == 0:
        error_msg = f"Error: Patch file is empty: {patch_file}"
        print(error_msg)
        return False, error_msg

    patch_file_abs = os.path.abspath(patch_file)
    
    # First attempt: Try to apply the patch as-is
    print(f"Applying patch with command: git -C {codebase_dir} apply {patch_file_abs}")
    original_result = subprocess.run(
        ["git", "-C", codebase_dir, "apply", patch_file_abs],
        capture_output=True, text=True, check=False
    )
    
    # If original patch applies successfully, we're done
    if original_result.returncode == 0:
        return True, "Patch applied successfully"
        
    # If it fails, log the error and try repair strategies
    print(f"Failed to apply original patch: {original_result.stderr}")
    
    # Read the original patch content
    with open(patch_file, 'r', encoding='utf-8') as f:
        original_patch = f.read()
    
    # Clean up any partial changes
    try:
        subprocess.run(["git", "-C", codebase_dir, "reset", "--hard"],
                       capture_output=True, text=True, check=False)
        print("Cleaned up partially applied changes")
    except Exception as e:
        print(f"Error cleaning up: {e}")
    
    # Try various repair strategies
    strategies = [
        lambda: sanitize_and_apply_patch(patch_file, original_patch, codebase_dir),
        
        lambda: apply_with_options(patch_file_abs, codebase_dir, ["--ignore-whitespace"]),
        
        lambda: apply_with_options(patch_file_abs, codebase_dir, ["--3way"]),
        
        lambda: sanitize_and_apply_patch(patch_file, original_patch, codebase_dir, ["--ignore-whitespace"]),
        
        lambda: sanitize_and_apply_patch(patch_file, original_patch, codebase_dir, ["--3way"])
    ]
    
    # Try each strategy until one succeeds
    results = []
    for i, strategy in enumerate(strategies):
        try:
            retry_suffix = f" (Repair strategy {i+1}/{len(strategies)})"
            print(f"Attempting patch repair{retry_suffix}")
            success, message = strategy()
            if success:
                return True, f"Patch applied successfully after repair{retry_suffix}"
            results.append(message)
        except Exception as e:
            error_msg = f"Error in repair strategy {i+1}: {str(e)}"
            results.append(error_msg)
            print(error_msg)
    
    # All strategies failed
    return False, "Failed to apply patch after all repair attempts:\n" + "\n".join(results)

def sanitize_and_apply_patch(patch_file: str, patch_content: str, codebase_dir: str, extra_options: List[str] = None) -> Tuple[bool, str]:
    """
    Sanitizes a patch and attempts to apply it.
    Returns (success_status, message).
    """
    # Create a PatchAnalyzer instance with the original content
    analyzer = PatchAnalyzer("")  # Empty path, we'll use the content directly
    analyzer.cleaned_patch = analyzer.sanitize_patch(patch_content)
    
    # If sanitizing produced no valid patch, fail
    if not analyzer.validate_patch_format():
        return False, "Failed to create a valid patch after sanitization"
    
    # Write the sanitized patch to a temporary file
    temp_patch_file = patch_file + ".sanitized"
    with open(temp_patch_file, 'w', encoding='utf-8') as f:
        f.write(analyzer.cleaned_patch)
    
    # Try to apply the sanitized patch
    cmd = ["git", "-C", codebase_dir, "apply"]
    if extra_options:
        cmd.extend(extra_options)
    cmd.append(os.path.abspath(temp_patch_file))
    
    print(f"Applying sanitized patch with: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode == 0:
        return True, "Sanitized patch applied successfully"
    else:
        return False, f"Failed to apply sanitized patch: {result.stderr}"

def apply_with_options(patch_file: str, codebase_dir: str, options: List[str]) -> Tuple[bool, str]:
    """
    Applies a patch with the specified git apply options.
    Returns (success_status, message).
    """
    cmd = ["git", "-C", codebase_dir, "apply"] + options + [patch_file]
    print(f"Trying with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode == 0:
        return True, f"Patch applied successfully with options: {' '.join(options)}"
    else:
        return False, f"Failed to apply patch with options {' '.join(options)}: {result.stderr}"

def _filter_patch_to_existing_files(patch_content: str, codebase_dir: str) -> str:
    """Keep only file diffs whose target file exists under codebase_dir."""
    analyzer = PatchAnalyzer("")
    analyzer.cleaned_patch = analyzer.sanitize_patch(patch_content)
    file_patches = analyzer.extract_file_patches()
    kept: List[str] = []
    for fp, content in file_patches.items():
        target = os.path.join(codebase_dir, fp)
        if os.path.exists(target):
            kept.append(content)
    return "".join(kept)

def _filter_patch_to_applicable_files(patch_content: str, codebase_dir: str) -> str:
    """Keep only file diffs that pass git apply --check individually."""
    analyzer = PatchAnalyzer("")
    analyzer.cleaned_patch = analyzer.sanitize_patch(patch_content)
    file_patches = analyzer.extract_file_patches()
    kept: List[str] = []
    for _, content in file_patches.items():
        tmp = None
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".patch")
            os.close(fd)
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(content if content.endswith('\n') else content + '\n')
            res = subprocess.run(["git", "-C", codebase_dir, "apply", "--check", os.path.abspath(tmp)], capture_output=True, text=True, check=False)
            if res.returncode == 0:
                kept.append(content)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    return "".join(kept)

def filter_and_apply_patch(patch_file: str, original_patch: str, codebase_dir: str, *, existing_only: bool = False, applicable_only: bool = False) -> Tuple[bool, str]:
    """Filter the patch content and try to apply it."""
    if existing_only:
        filtered = _filter_patch_to_existing_files(original_patch, codebase_dir)
    elif applicable_only:
        filtered = _filter_patch_to_applicable_files(original_patch, codebase_dir)
    else:
        filtered = original_patch
    if not filtered.strip():
        return False, "Filtered patch became empty"
    tmp = patch_file + ".filtered"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(filtered if filtered.endswith('\n') else filtered + '\n')
    cmd = ["git", "-C", codebase_dir, "apply", os.path.abspath(tmp)]
    print(f"Applying filtered patch with: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode == 0:
        return True, "Filtered patch applied successfully"
    return False, f"Failed to apply filtered patch: {res.stderr}"

# ------------------------------
# Mypy Validator
# ------------------------------

class Validator:
    """
    Runs mypy on a file.
    """
    def run_mypy(self, file_path: str) -> str:
        try:
            result = subprocess.run(["mypy", file_path],
                                    capture_output=True,
                                    text=True,
                                    check=False)
            return result.stdout + "\n" + result.stderr
        except Exception as e:
            return f"Error running mypy: {e}"

# ------------------------------
# Main Pipeline
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sophisticated Type-Aware Patch Rewriter Framework")
    parser.add_argument("--codebase", required=True, help="Path to the Python code base directory")
    parser.add_argument("--patch", required=True, help="Path to the patch file (unified diff format)")
    parser.add_argument("--variable", help="Target variable name to retrieve related code context (will be used in addition to automatically detected variables)")
    parser.add_argument("--output", required=True, help="Path to output the revised patch file")
    parser.add_argument("--inference-mode", choices=["static", "llm", "hybrid"], default="hybrid",
                        help="Type inference mode: static (static analysis only), llm (LLM with local context only), hybrid (static + LLM validation)")
    parser.add_argument(
        "--extra-context-file",
        help="Optional sideband artifact for hybrid runs; contents are streamed verbatim for whatever auxiliary reasoning the runner wires up"
    )
    args = parser.parse_args()
    
    # Parse inference mode
    mode_map = {"static": InferenceMode.STATIC, "llm": InferenceMode.LLM, "hybrid": InferenceMode.HYBRID}
    inference_mode = mode_map[args.inference_mode]
    print(f"Type Inference Mode: {args.inference_mode.upper()}")

    total_t0 = time.time()
    print("=== Step 1: Initializing Type Inference Engine ===")
    type_engine = TypeInferenceEngine(args.codebase, mode=inference_mode)
    print(f"Engine initialized in {inference_mode.value.upper()} mode")
    
    print("\n=== Step 2: Analyzing Patch ===")
    with open(args.patch, 'r', encoding='utf-8') as f:
        original_patch = f.read()
    
    patch_analyzer = PatchAnalyzer(args.patch)
    if not patch_analyzer.validate_patch_format():
        print("Error: Patch file is not in valid unified diff format.")
        return
    
    syntax_errors = patch_analyzer.validate_added_code_syntax()
    if syntax_errors:
        print("Syntax errors in added code (pre-rewrite):")
        print(json.dumps(syntax_errors, indent=2))
    else:
        print("No syntax errors detected in added code (pre-rewrite).")

    print("\n=== Step 2.1: Validating Patch Applicability ===")
    patch_applicable, apply_message = apply_and_validate_patch(args.patch, args.codebase)
    print(apply_message)
    if not patch_applicable:
        print("Warning: The patch could not be applied. Continuing with analysis but results may be affected.")
    
    print("\n=== Step 2.2: Identifying Variables Needing Type Annotations ===")
    extractor = PatchVariableExtractor()
    ve_t0 = time.time()
    extracted_variables = extractor.extract_variables(original_patch)
    ve_seconds = time.time() - ve_t0
    print(f"Variables identified for type annotation: {extracted_variables}")
    
    # Add the CLI-specified variable if provided
    all_variables = list(extracted_variables)
    if args.variable and args.variable not in all_variables:
        all_variables.append(args.variable)
    
    # Determine target file from patch
    target_file = None
    for line in original_patch.splitlines():
        if line.startswith('+++'):
            # Extract file path from +++ b/path/to/file.py
            match = re.match(r'\+\+\+ b/(.+)', line)
            if match:
                target_file = os.path.join(args.codebase, match.group(1))
                break
    
    print("\n=== Step 3: Comprehensive Type Inference ===")
    if all_variables:
        print(f"Inferring types for {len(all_variables)} variables using {inference_mode.value.upper()} mode...")
        type_inferences = type_engine.infer_types_for_patch_variables(all_variables, target_file)
        
        print("\nInferred Types:")
        for var_name, inference in type_inferences.items():
            print(f"  {var_name}: {inference.inferred_type} ")
            print(f"    (confidence: {inference.confidence:.2f}, source: {inference.source})")
    else:
        type_inferences = {}
        print("No variables to analyze; skipping type inference.")

    # Prepare a serializable payload for type inferences
    try:
        types_payload = {
            "mode": inference_mode.value,
            "variables_analyzed": all_variables,
            "inferences": [
                {
                    "name": name,
                    "inferred_type": inf.inferred_type,
                    "confidence": inf.confidence,
                    "source": inf.source,
                    "location": list(inf.location) if getattr(inf, "location", None) else None,
                }
                for name, inf in type_inferences.items()
            ],
        }
        print("PAGENT_TYPES_JSON:" + json.dumps(types_payload))
    except Exception:
        pass

    max_retries = 3
    retry_count = 0
    best_patch = None
    least_errors_count = float('inf')
    least_errors_patch = None
    mypy_error_history = []
    last_apply_error_message = ""
    
    # Load extra context if provided (runner passes this only for hybrid)
    extra_context_str = ""
    if args.extra_context_file:
        try:
            with open(args.extra_context_file, 'r', encoding='utf-8') as _ecf:
                extra_context_str = _ecf.read()
        except Exception as _e:
            print(f"Warning: failed to read extra context file: {args.extra_context_file}: {_e}")

    while retry_count <= max_retries:
        retry_suffix = f" (Retry {retry_count}/{max_retries})" if retry_count > 0 else ""
        additional_context = ""
        ctx_parts = []
        if retry_count > 0 and mypy_error_history:
            ctx_parts.append(f"Previous attempt had mypy errors. Please fix the following issues if necessary:\n{mypy_error_history[-1]}")
        if last_apply_error_message:
            ctx_parts.append(f"Previous attempt had patch apply errors. Please fix the issues so the patch applies cleanly:\n{last_apply_error_message}")
        if ctx_parts:
            additional_context = "\n\n".join(ctx_parts)
        
        print(f"\n=== Step 4: Preparing Type Information for Patch Rewriter{retry_suffix} ===")
        
        # Format type inferences for patch rewriter
        type_info_formatted = type_engine.format_for_patch_rewriter(type_inferences)
        print("Type information prepared:")
        print(type_info_formatted[:500] + "..." if len(type_info_formatted) > 500 else type_info_formatted)
        
        print(f"\n=== Step 5: Rewriting Patch with LLM{retry_suffix} ===")
        llm_rewriter = LLMRewriter()
        is_last_retry = (retry_count >= max_retries-2)
        # rewriter context only: type info + internal diagnostics + optional extra context
        rewriter_context_parts = [type_info_formatted]
        if additional_context:
            rewriter_context_parts.append(additional_context)
        if extra_context_str:
            rewriter_context_parts.append(extra_context_str)
        rewriter_context = "\n\n".join([p for p in rewriter_context_parts if p])
        rp_t0 = time.time()
        revised_patch = llm_rewriter.rewrite_patch(
            original_patch, 
            rewriter_context, 
            None,
            is_last_retry
        )
        rp_seconds = time.time() - rp_t0
        
        syntax_validation = validate_syntax_of_revised_patch(revised_patch)
        if syntax_validation:
            print("Revised patch syntax validated successfully.")
        else:
            print("Warning: Revised patch has syntax errors.")

        temp_output = f"{args.output}.attempt{retry_count}"
        os.makedirs(os.path.dirname(os.path.abspath(temp_output)), exist_ok=True)
        with open(temp_output, 'w', encoding='utf-8') as f:
            f.write(revised_patch)

        print(f"\n=== Step 5.1: Validating Revised Patch Applicability{retry_suffix} ===")
        revised_patch_applicable, revised_apply_message = apply_and_validate_patch(temp_output, args.codebase)
        print(revised_apply_message)
        
        if not revised_patch_applicable:
            print("Warning: The revised patch could not be applied even after repairs. Using original patch for this iteration.")
            last_apply_error_message = revised_apply_message
            revised_patch = original_patch
            with open(temp_output, 'w', encoding='utf-8') as f:
                f.write(original_patch)

        print(f"\n=== Step 6: Running Mypy on the Revised Patch{retry_suffix} ===")
        validator = Validator()
        mypy_report = validator.run_mypy(temp_output)
        print("Mypy output:")
        print(mypy_report)
        error_count = mypy_report.count("error:")
        mypy_error_history.append(mypy_report)
        
        if error_count < least_errors_count:
            least_errors_count = error_count
            least_errors_patch = revised_patch
        
        if error_count == 0:
            print(f"Success! No mypy errors detected after {retry_count + 1} attempts.")
            best_patch = revised_patch
            break
        
        retry_count += 1

    if best_patch:
        final_patch = best_patch
        print("Using error-free patch.")
    elif least_errors_patch:
        final_patch = least_errors_patch
        print(f"Falling back to patch with the least errors ({least_errors_count} errors).")
    else:
        final_patch = original_patch
        print("All attempts failed. Falling back to the original patch.")

    # Final conformance step: ensure the patch applies or, if not, keep original content (never emit empty)
    try:
        # Always ensure newline at end
        if final_patch and not final_patch.endswith("\n"):
            final_patch = final_patch + "\n"
        # Try apply --check
        import tempfile
        fd, tmpf = tempfile.mkstemp(suffix=".patch")
        try:
            import os as _os
            _os.close(fd)
            with open(tmpf, 'w', encoding='utf-8') as _tf:
                _tf.write(final_patch)
            res = subprocess.run(["git", "-C", args.codebase, "apply", "--check", _os.path.abspath(tmpf)], capture_output=True, text=True, check=False)
            if res.returncode != 0:
                print("Final patch failed --check, filtering to applicable file diffs...")
                filtered = _filter_patch_to_applicable_files(final_patch, args.codebase)
                if filtered and not filtered.endswith("\n"):
                    filtered += "\n"
                if filtered:
                    with open(tmpf, 'w', encoding='utf-8') as _tf:
                        _tf.write(filtered)
                    res2 = subprocess.run(["git", "-C", args.codebase, "apply", "--check", _os.path.abspath(tmpf)], capture_output=True, text=True, check=False)
                    if res2.returncode == 0:
                        print("Using filtered patch that passes --check.")
                        final_patch = filtered
                    else:
                        print("Filtered patch still fails --check; keeping original LLM patch content.")
                else:
                    print("Filtering removed all file diffs; keeping original LLM patch content.")
        finally:
            try:
                _os.remove(tmpf)
            except Exception:
                pass
    except Exception as e:
        print(f"Final conformance step error: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(final_patch)
    print(f"Final patch saved to: {args.output}")

    print("\n=== Final Mypy Validation ===")
    final_mypy_report = validator.run_mypy(args.output)
    print("Final mypy output:")
    print(final_mypy_report)

    # ---- Emit metrics for runner ----
    total_seconds = time.time() - total_t0
    metrics: Dict[str, Any] = {
        "total_seconds": total_seconds,
        "static_seconds": type_engine.metrics.get("static_seconds", 0.0),
        "llm_inference_seconds": type_engine.metrics.get("llm_inference_seconds", 0.0),
        "llm_inference_calls": type_engine.metrics.get("llm_calls", []),
        "var_extraction_seconds": ve_seconds,
        "var_extraction_calls": getattr(extractor, "calls", []),
        "rewriter_seconds": sum(c.get("seconds", 0.0) for c in getattr(llm_rewriter, "calls", [])) if 'llm_rewriter' in locals() else 0.0,
        "rewriter_calls": getattr(llm_rewriter, "calls", []) if 'llm_rewriter' in locals() else [],
    }
    try:
        print("PAGENT_METRICS_JSON:" + json.dumps(metrics))
    except Exception:
        pass

if __name__ == "__main__":
    main()