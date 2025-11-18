"""
Comprehensive Type Inference Engine for PAGENT

Provides three modes for ablation studies:
1. STATIC: Mypy + AST + CFG-based constraint inference
2. LLM: LLM-based inference with local code context (fair comparison)
3. HYBRID: Static inference validated by LLM (default, most effective)

Design principles:
- Static analysis is primary, using mypy for robust inference
- LLM acts as validator/cross-checker, not fallback
- Fair ablation: LLM-only mode gets reasonable local context, not full static results
- Efficient: cache mypy results, reuse AST/CFG analysis
"""

import os
import ast
import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel

from llm_provider import responses_parse, responses_create, MODEL_GPT4O_MINI, MODEL_GPT4O
from advanced_analysis import analyze_file
import time


class InferenceMode(Enum):
    """Type inference mode for ablation studies."""
    STATIC = "static"  # Only static analysis
    LLM = "llm"  # Only LLM with local context
    HYBRID = "hybrid"  # Static + LLM validation (default)


@dataclass
class TypeInference:
    """Result of type inference for a variable/parameter."""
    name: str
    inferred_type: str
    confidence: float  # 0.0-1.0
    source: str  # "mypy", "ast", "cfg", "llm", "hybrid"
    location: Optional[Tuple[str, int]] = None  # (file, line)
    
    
class VariableTypeInfo(BaseModel):
    """Pydantic model for LLM structured output."""
    variable: str
    type: str
    confidence: str  # "high", "medium", "low"


class TypeInferenceListResult(BaseModel):
    """Pydantic model for multiple type inferences."""
    inferences: List[VariableTypeInfo]


# ============================================================================
# Mypy Integration
# ============================================================================

class MypyIntegration:
    """
    Integrates with mypy to extract inferred types.
    Uses mypy's JSON output to get precise type information.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self._cache: Dict[str, Dict[str, str]] = {}
        
    def run_mypy_on_file(self, file_path: str) -> Dict[str, str]:
        """
        Run mypy on a specific file and extract inferred types.
        Returns dict: {variable_location: inferred_type}
        """
        if file_path in self._cache:
            return self._cache[file_path]
            
        try:
            # Run mypy with machine-readable output
            result = subprocess.run(
                [
                    "mypy",
                    "--show-column-numbers",
                    "--no-error-summary",
                    "--no-pretty",
                    "--show-absolute-path",
                    file_path
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.project_root)
            )
            
            types = self._parse_mypy_output(result.stdout + result.stderr, file_path)
            self._cache[file_path] = types
            return types
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Mypy execution failed: {e}")
            return {}
    
    def _parse_mypy_output(self, output: str, file_path: str) -> Dict[str, str]:
        """
        Parse mypy output to extract type information.
        Look for patterns like: "Revealed type is 'int'" or inferred parameter types.
        """
        types = {}
        
        # Pattern for revealed types
        revealed_pattern = re.compile(r'(.+?):(\d+):(\d+):\s*note:\s*Revealed type is ["\']([^"\']+)["\']')
        
        for match in revealed_pattern.finditer(output):
            filepath, line, col, revealed_type = match.groups()
            location = f"{line}:{col}"
            types[location] = revealed_type
            
        return types


# ============================================================================
# Enhanced Static Analyzer
# ============================================================================

class EnhancedStaticAnalyzer:
    """
    Comprehensive static type inference using:
    1. Explicit annotations (PEP 484)
    2. Assignment inference (literals, constructors, comprehensions)
    3. Control-flow narrowing (isinstance, None checks)
    4. Attribute and class member tracking
    5. CFG/DFG-based constraint propagation
    """
    
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.mypy = MypyIntegration(codebase_path)
        self._cfg_cache: Dict[str, Any] = {}
        
    def infer_types_in_file(self, file_path: str) -> Dict[str, TypeInference]:
        """
        Perform comprehensive type inference on a file.
        Returns dict: {variable_name: TypeInference}
        """
        inferences: Dict[str, TypeInference] = {}
        
        # Step 1: Get mypy inferred types (highest confidence)
        mypy_types = self.mypy.run_mypy_on_file(file_path)
        
        # Step 2: Parse AST for annotations and assignments
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            
            # Extract annotations
            for inference in self._extract_annotations(tree, file_path, source):
                inferences[inference.name] = inference
                
            # Extract from assignments
            for inference in self._extract_from_assignments(tree, file_path, source):
                if inference.name not in inferences:
                    inferences[inference.name] = inference
                    
            # Extract from class attributes
            for inference in self._extract_class_attributes(tree, file_path, source):
                if inference.name not in inferences:
                    inferences[inference.name] = inference
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            
        # Step 3: CFG-based constraint propagation
        cfg_inferences = self._infer_from_cfg(file_path)
        for name, inference in cfg_inferences.items():
            if name not in inferences:
                inferences[name] = inference
                
        return inferences
    
    def _extract_annotations(self, tree: ast.AST, file_path: str, source: str) -> List[TypeInference]:
        """Extract explicit type annotations from AST."""
        inferences = []
        
        for node in ast.walk(tree):
            # Function parameter annotations
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    if arg.annotation:
                        type_str = ast.unparse(arg.annotation)
                        inferences.append(TypeInference(
                            name=arg.arg,
                            inferred_type=type_str,
                            confidence=1.0,
                            source="ast_annotation",
                            location=(file_path, node.lineno)
                        ))
                        
                # Return type annotation
                if node.returns:
                    return_type = ast.unparse(node.returns)
                    inferences.append(TypeInference(
                        name=f"{node.name}__return",
                        inferred_type=return_type,
                        confidence=1.0,
                        source="ast_annotation",
                        location=(file_path, node.lineno)
                    ))
                    
            # Variable annotations (PEP 526)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    type_str = ast.unparse(node.annotation)
                    inferences.append(TypeInference(
                        name=node.target.id,
                        inferred_type=type_str,
                        confidence=1.0,
                        source="ast_annotation",
                        location=(file_path, node.lineno)
                    ))
                    
        return inferences
    
    def _extract_from_assignments(self, tree: ast.AST, file_path: str, source: str) -> List[TypeInference]:
        """Infer types from assignment values."""
        inferences = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    inferred_type = self._infer_from_value(node.value)
                    
                    if inferred_type:
                        inferences.append(TypeInference(
                            name=var_name,
                            inferred_type=inferred_type,
                            confidence=0.8,
                            source="ast_assignment",
                            location=(file_path, node.lineno)
                        ))
                        
        return inferences
    
    def _infer_from_value(self, value_node: ast.AST) -> Optional[str]:
        """Infer type from value node with comprehensive rules."""
        # Constants
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return "bool"
            elif isinstance(value_node.value, int):
                return "int"
            elif isinstance(value_node.value, float):
                return "float"
            elif isinstance(value_node.value, str):
                return "str"
            elif value_node.value is None:
                return "None"
                
        # Collections
        elif isinstance(value_node, ast.List):
            if value_node.elts:
                elem_types = set()
                for elt in value_node.elts[:5]:  # Sample first 5
                    elem_type = self._infer_from_value(elt)
                    if elem_type:
                        elem_types.add(elem_type)
                if len(elem_types) == 1:
                    return f"List[{elem_types.pop()}]"
            return "List[Any]"
            
        elif isinstance(value_node, ast.Dict):
            return "Dict[Any, Any]"
            
        elif isinstance(value_node, ast.Set):
            return "Set[Any]"
            
        elif isinstance(value_node, ast.Tuple):
            return "Tuple[Any, ...]"
            
        # Constructor calls
        elif isinstance(value_node, ast.Call):
            if isinstance(value_node.func, ast.Name):
                func_name = value_node.func.id
                constructors = {
                    "list": "List[Any]",
                    "dict": "Dict[Any, Any]",
                    "set": "Set[Any]",
                    "tuple": "Tuple[Any, ...]",
                    "int": "int",
                    "float": "float",
                    "str": "str",
                    "bool": "bool",
                }
                if func_name in constructors:
                    return constructors[func_name]
                    
        # Comprehensions
        elif isinstance(value_node, ast.ListComp):
            return "List[Any]"
        elif isinstance(value_node, ast.DictComp):
            return "Dict[Any, Any]"
        elif isinstance(value_node, ast.SetComp):
            return "Set[Any]"
            
        return None
    
    def _extract_class_attributes(self, tree: ast.AST, file_path: str, source: str) -> List[TypeInference]:
        """Extract class attribute types."""
        inferences = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    # Annotated attributes
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        type_str = ast.unparse(item.annotation)
                        inferences.append(TypeInference(
                            name=f"{node.name}.{item.target.id}",
                            inferred_type=type_str,
                            confidence=1.0,
                            source="ast_class_attr",
                            location=(file_path, item.lineno)
                        ))
                        
        return inferences
    
    def _infer_from_cfg(self, file_path: str) -> Dict[str, TypeInference]:
        """Use CFG/DFG for constraint-based inference."""
        inferences = {}
        
        if file_path in self._cfg_cache:
            file_analysis = self._cfg_cache[file_path]
        else:
            try:
                file_analysis = analyze_file(file_path)
                self._cfg_cache[file_path] = file_analysis
            except Exception as e:
                print(f"CFG analysis failed for {file_path}: {e}")
                return {}
                
        # TODO: Implement constraint propagation using in_sets/out_sets
        # For now, return empty; this can be expanded with lattice-based inference
        
        return inferences


# ============================================================================
# LLM-Based Inference (for LLM-only and validation modes)
# ============================================================================

class LLMTypeInferencer:
    """
    LLM-based type inference with proper context extraction.
    Used in two modes:
    1. LLM-only: Primary inference with local code context
    2. Hybrid: Validation of static inference results
    """
    
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []  # per-call LLM stats

    def infer_types_from_context(self, 
                                  variables: List[str],
                                  code_contexts: Dict[str, str],
                                  mode: InferenceMode = InferenceMode.LLM) -> Dict[str, TypeInference]:
        """
        Infer types using LLM with local code context.
        
        Args:
            variables: List of variable names to infer
            code_contexts: Dict mapping variable -> relevant code snippet
            mode: Inference mode
            
        Returns:
            Dict mapping variable name to TypeInference
        """
        if not variables:
            return {}
            
        # Format context for LLM
        context_str = self._format_context(variables, code_contexts)
        
        # Get LLM inference
        try:
            result = self._call_llm_for_types(context_str, mode)
            inferences = {}
            
            for item in result:
                conf_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                confidence = conf_map.get(item.get("confidence", "medium"), 0.7)
                
                inferences[item["variable"]] = TypeInference(
                    name=item["variable"],
                    inferred_type=item["type"],
                    confidence=confidence,
                    source="llm"
                )
                
            return inferences
            
        except Exception as e:
            print(f"LLM type inference failed: {e}")
            return {}
    
    def validate_static_inference(self,
                                   static_inferences: Dict[str, TypeInference],
                                   code_contexts: Dict[str, str]) -> Dict[str, TypeInference]:
        """
        Use LLM to validate static inference results.
        Returns updated inferences with arbitration.
        """
        if not static_inferences:
            return {}
            
        # Prepare validation prompt
        static_str = self._format_static_for_validation(static_inferences, code_contexts)
        
        try:
            llm_review = self._call_llm_for_validation(static_str)
            
            # Arbitrate: prefer static unless LLM has high confidence disagreement
            final_inferences = {}
            for var_name, static_inf in static_inferences.items():
                if var_name in llm_review:
                    llm_inf = llm_review[var_name]
                    
                    # If LLM strongly disagrees and has good reason, flag for review
                    if llm_inf.confidence > 0.8 and llm_inf.inferred_type != static_inf.inferred_type:
                        # Hybrid: take union or more general type
                        final_inferences[var_name] = TypeInference(
                            name=var_name,
                            inferred_type=f"Union[{static_inf.inferred_type}, {llm_inf.inferred_type}]",
                            confidence=0.7,
                            source="hybrid"
                        )
                    else:
                        # Static wins
                        final_inferences[var_name] = static_inf
                else:
                    final_inferences[var_name] = static_inf
                    
            return final_inferences
            
        except Exception as e:
            print(f"LLM validation failed: {e}")
            return static_inferences
    
    def _format_context(self, variables: List[str], code_contexts: Dict[str, str]) -> str:
        """Format code context for LLM inference."""
        formatted = []
        for var in variables:
            context = code_contexts.get(var, "")
            if context:
                formatted.append(f"Variable: {var}\nCode Context:\n{context}\n")
        return "\n===\n".join(formatted)
    
    def _format_static_for_validation(self, 
                                      static_inferences: Dict[str, TypeInference],
                                      code_contexts: Dict[str, str]) -> str:
        """Format static inferences for LLM validation."""
        formatted = []
        for var_name, inference in static_inferences.items():
            context = code_contexts.get(var_name, "")
            formatted.append(
                f"Variable: {var_name}\n"
                f"Static Inference: {inference.inferred_type} (confidence: {inference.confidence:.2f})\n"
                f"Code Context:\n{context}\n"
            )
        return "\n===\n".join(formatted)
    
    def _call_llm_for_types(self, context_str: str, mode: InferenceMode) -> List[Dict[str, str]]:
        """Call LLM to infer types from context."""
        prompt = (
            "You are a Python type inference expert. Analyze the code context for each variable "
            "and infer the most appropriate Python type annotation.\n\n"
            f"{context_str}\n\n"
            "Return your inferences in JSON format with the following structure:\n"
            "{\n"
            '  "inferences": [\n'
            '    {"variable": "var_name", "type": "inferred_type", "confidence": "high|medium|low"},\n'
            '    ...\n'
            "  ]\n"
            "}\n\n"
            "Use standard Python typing conventions (int, str, List[T], Dict[K,V], Optional[T], etc.)."
        )
        
        messages = [
            {"role": "system", "content": "You are a Python type inference specialist."},
            {"role": "user", "content": prompt}
        ]
        # Char counts
        input_chars = sum(len(m.get("content", "")) for m in messages)
        
        parsed = responses_parse(
            model=MODEL_GPT4O_MINI,
            input=messages,
            text_format=TypeInferenceListResult,
            max_output_tokens=1500
        )
        try:
            output_chars = len(parsed.output_parsed.model_dump_json())  # type: ignore[attr-defined]
        except Exception:
            try:
                output_chars = len(json.dumps(parsed.output_parsed))  # type: ignore
            except Exception:
                output_chars = 0
        self.calls.append({"stage": "llm_types", "input_chars": input_chars, "output_chars": output_chars})

        return [{"variable": item.variable, "type": item.type, "confidence": item.confidence} 
                for item in parsed.output_parsed.inferences]
    
    def _call_llm_for_validation(self, static_str: str) -> Dict[str, TypeInference]:
        """Call LLM to validate static inferences."""
        prompt = (
            "You are validating static type inference results. For each variable, review the "
            "static inference and code context. If you agree, respond with the same type. "
            "If you disagree strongly (with high confidence), suggest an alternative.\n\n"
            f"{static_str}\n\n"
            "Return your review in JSON format:\n"
            "{\n"
            '  "inferences": [\n'
            '    {"variable": "var_name", "type": "your_inference", "confidence": "high|medium|low"},\n'
            '    ...\n'
            "  ]\n"
            "}"
        )
        
        messages = [
            {"role": "system", "content": "You are a Python type validation expert."},
            {"role": "user", "content": prompt}
        ]
        input_chars = sum(len(m.get("content", "")) for m in messages)
        parsed = responses_parse(
            model=MODEL_GPT4O_MINI,
            input=messages,
            text_format=TypeInferenceListResult,
            max_output_tokens=1500
        )
        try:
            output_chars = len(parsed.output_parsed.model_dump_json())  # type: ignore[attr-defined]
        except Exception:
            try:
                output_chars = len(json.dumps(parsed.output_parsed))  # type: ignore
            except Exception:
                output_chars = 0
        self.calls.append({"stage": "llm_validation", "input_chars": input_chars, "output_chars": output_chars})
        
        conf_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
        result = {}
        for item in parsed.output_parsed.inferences:
            result[item.variable] = TypeInference(
                name=item.variable,
                inferred_type=item.type,
                confidence=conf_map.get(item.confidence, 0.7),
                source="llm_validation"
            )
            
        return result


# ============================================================================
# Context Extraction for LLM
# ============================================================================

class CodeContextExtractor:
    """
    Extracts minimal, relevant code context for LLM inference.
    Ensures fair comparison in ablation: LLM gets usage patterns, not static results.
    """
    
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        
    def extract_context_for_variable(self, variable: str, file_path: Optional[str] = None) -> str:
        """
        Extract relevant code context for a variable.
        Includes: assignments, function signatures, usage patterns, but NOT type inferences.
        """
        if file_path:
            files_to_search = [file_path]
        else:
            files_to_search = self._find_files_with_variable(variable)
            
        contexts = []
        for fpath in files_to_search[:3]:  # Limit to 3 files
            ctx = self._extract_from_file(variable, fpath)
            if ctx:
                contexts.append(ctx)
                
        return "\n".join(contexts[:500])  # Limit total context length
    
    def _find_files_with_variable(self, variable: str) -> List[str]:
        """Find files containing the variable."""
        relevant_files = []
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            if re.search(r'\b' + re.escape(variable) + r'\b', f.read()):
                                relevant_files.append(file_path)
                    except:
                        continue
        return relevant_files
    
    def _extract_from_file(self, variable: str, file_path: str) -> str:
        """Extract relevant lines from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            relevant_lines = []
            var_pattern = re.compile(r'\b' + re.escape(variable) + r'\b')
            
            for i, line in enumerate(lines):
                if var_pattern.search(line):
                    # Include context: 2 lines before, current, 2 lines after
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    relevant_lines.extend(lines[start:end])
                    
            # Deduplicate and limit
            seen = set()
            unique_lines = []
            for line in relevant_lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
                    
            return "".join(unique_lines[:20])  # Max 20 lines per file
            
        except Exception as e:
            return ""


# ============================================================================
# Main Type Inference Engine
# ============================================================================

class TypeInferenceEngine:
    """
    Main orchestrator for type inference.
    Supports three modes for ablation studies.
    """
    
    def __init__(self, codebase_path: str, mode: InferenceMode = InferenceMode.HYBRID):
        self.codebase_path = codebase_path
        self.mode = mode
        self.static_analyzer = EnhancedStaticAnalyzer(codebase_path)
        self.llm_inferencer = LLMTypeInferencer()
        self.context_extractor = CodeContextExtractor(codebase_path)
        self.metrics: Dict[str, Any] = {
            "static_seconds": 0.0,
            "llm_inference_seconds": 0.0,
            "llm_calls": [],
        }
        
    def infer_types_for_patch_variables(self, 
                                         variables: List[str],
                                         file_path: Optional[str] = None) -> Dict[str, TypeInference]:
        """
        Main entry point: infer types for variables in a patch.
        
        Args:
            variables: List of variable names to type
            file_path: Optional specific file to analyze
            
        Returns:
            Dict mapping variable name to TypeInference
        """
        if self.mode == InferenceMode.STATIC:
            return self._static_only_inference(variables, file_path)
        elif self.mode == InferenceMode.LLM:
            return self._llm_only_inference(variables, file_path)
        else:  # HYBRID
            return self._hybrid_inference(variables, file_path)
    
    def _static_only_inference(self, variables: List[str], file_path: Optional[str]) -> Dict[str, TypeInference]:
        """Static-only mode: use only static analysis."""
        if not file_path:
            # Find most relevant file
            file_path = self._find_primary_file(variables)
            
        if file_path:
            t0 = time.time()
            all_inferences = self.static_analyzer.infer_types_in_file(file_path)
            self.metrics["static_seconds"] += time.time() - t0
            return {v: all_inferences[v] for v in variables if v in all_inferences}
        return {}
    
    def _llm_only_inference(self, variables: List[str], file_path: Optional[str]) -> Dict[str, TypeInference]:
        """LLM-only mode: use LLM with fair local context (no static results)."""
        # Extract code contexts for each variable
        code_contexts = {}
        for var in variables:
            context = self.context_extractor.extract_context_for_variable(var, file_path)
            code_contexts[var] = context
        t0 = time.time()
        out = self.llm_inferencer.infer_types_from_context(variables, code_contexts, InferenceMode.LLM)
        self.metrics["llm_inference_seconds"] += time.time() - t0
        # collect per-call stats
        if self.llm_inferencer.calls:
            self.metrics["llm_calls"].extend(self.llm_inferencer.calls)
            self.llm_inferencer.calls = []
        return out
    
    def _hybrid_inference(self, variables: List[str], file_path: Optional[str]) -> Dict[str, TypeInference]:
        """Hybrid mode: static analysis validated by LLM."""
        # Step 1: Get static inferences
        static_inferences = self._static_only_inference(variables, file_path)
        
        # Step 2: Extract code contexts
        code_contexts = {}
        for var in variables:
            context = self.context_extractor.extract_context_for_variable(var, file_path)
            code_contexts[var] = context
            
        # Step 3: LLM validates static results
        t0 = time.time()
        validated_inferences = self.llm_inferencer.validate_static_inference(static_inferences, code_contexts)
        self.metrics["llm_inference_seconds"] += time.time() - t0
        if self.llm_inferencer.calls:
            self.metrics["llm_calls"].extend(self.llm_inferencer.calls)
            self.llm_inferencer.calls = []
        
        # Step 4: For variables not in static, use LLM
        for var in variables:
            if var not in validated_inferences and var in code_contexts:
                t1 = time.time()
                llm_only = self.llm_inferencer.infer_types_from_context([var], {var: code_contexts[var]})
                self.metrics["llm_inference_seconds"] += time.time() - t1
                if self.llm_inferencer.calls:
                    self.metrics["llm_calls"].extend(self.llm_inferencer.calls)
                    self.llm_inferencer.calls = []
                if var in llm_only:
                    validated_inferences[var] = llm_only[var]
                    
        return validated_inferences
    
    def _find_primary_file(self, variables: List[str]) -> Optional[str]:
        """Find the most relevant file for the given variables."""
        # Simple heuristic: find file with most variables
        file_counts: Dict[str, int] = {}
        
        for var in variables:
            files = self.context_extractor._find_files_with_variable(var)
            for f in files:
                file_counts[f] = file_counts.get(f, 0) + 1
                
        if file_counts:
            return max(file_counts.items(), key=lambda x: x[1])[0]
        return None
    
    def format_for_patch_rewriter(self, inferences: Dict[str, TypeInference]) -> str:
        """Format type inferences for patch rewriter consumption."""
        lines = []
        for var_name, inference in sorted(inferences.items(), key=lambda x: x[1].confidence, reverse=True):
            lines.append(
                f"{var_name}: {inference.inferred_type} "
                f"(confidence: {inference.confidence:.2f}, source: {inference.source})"
            )
        return "\n".join(lines)
