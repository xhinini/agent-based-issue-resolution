# PAGENT: Patch Agent for Type-Aware Code Repair

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-orange" alt="OpenAI GPT-4o">
</p>

## 🚀 Overview

PAGENT (Patch Agent) is a specialized framework designed to address type-related errors in patches generated by LLM-based code agents. It combines program analysis techniques (CFG creation and exploration) with LLM-based inference to accurately determine variable types in code patches. By applying repository-level static code analysis and refining the results with LLM-based type inference, PAGENT successfully fixes patches that fail due to incorrect variable typing, improving the overall success rate of automated issue resolution.

## ✨ Key Features

- **Static Code Analysis**: Utilizes AST and CFG to analyze code structure
- **Type Inference**: Combines static analysis with LLM-based inference
- **Patch Rewriting**: Utilizing LLMs to automatically enhances patches with correct type information
- **Validation Pipeline**: Tests patches with syntax validation and mypy

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see below)

```bash
pip install openai networkx pydantic
```

## 🛠️ Core Components

### 1. Code Analysis & Static Inference

The `CodeAnalyzer` class loads a Python codebase and infers variable types using AST and simple heuristics.

### 2. Patch Analysis

The `PatchAnalyzer` processes unified diff patch files, preserving the full diff context and ensuring valid unified diff format.

### 3. Variable Extraction from Patch

The `PatchVariableExtractor` uses an LLM to identify variables in a patch that would benefit from type annotations.

### 4. Advanced Code Retrieval

The `EnhancedCodeRetriever` retrieves and flattens related code context using advanced AST/CFG/data-flow analysis.

### 5. Type Inference with LLM

The `EnhancedLLMTypeInferencer` uses an LLM to infer types for multiple variables based on their contexts.

### 6. Patch Rewriting with LLM

The `LLMRewriter` rewrites patches with enhanced type information and prompt.

### 7. Patch Analysis Script

The `patch_ana.py` script analyzes failed patches using the OpenAI API (GPT-4o) with a custom prompt, extracting sections from case files, analyzing patches, and saving results.

## 🔍 Usage

### Command Line Interface

```bash
# Basic CLI usage
python core.py --codebase /path/to/codebase --patch /path/to/patch.diff --output /path/to/output/revised_patch.diff

# Specifying a target variable for additional context retrieval
python core.py --codebase /path/to/codebase --patch /path/to/patch.diff --output /path/to/output/revised_patch.diff --variable target_variable_name
```

### Python API

```python
# Example: Analyzing a patch file
from core import PatchAnalyzer, LLMRewriter

# Initialize the patch analyzer
analyzer = PatchAnalyzer("path/to/patch.diff")

# Sanitize and validate the patch
analyzer.sanitize_patch(analyzer.patch_content)
is_valid = analyzer.validate_patch_format()

if is_valid:
    # Extract variables from the patch
    from core import PatchVariableExtractor
    extractor = PatchVariableExtractor()
    variables = extractor.extract_variables(analyzer.cleaned_patch)
    
    # Retrieve context for the variables
    from core import EnhancedCodeRetriever
    retriever = EnhancedCodeRetriever("path/to/codebase")
    contexts = retriever.retrieve_context_for_variables(variables)
    
    # Infer types for the variables
    from core import EnhancedLLMTypeInferencer
    inferencer = EnhancedLLMTypeInferencer()
    types = inferencer.infer_types(contexts)
    
    # Rewrite the patch with enhanced type information
    rewriter = LLMRewriter()
    result = rewriter.rewrite_patch(analyzer.cleaned_patch, str(contexts), str(types))
    
    print(f"Explanation: {result.explanation}")
    print(f"Rewritten patch: {result.patch_in_unified_diff_format}")
```

### Analyzing Patch Failures

```python
# Example: Analyzing patch failures
from patch_ana import analyze_case, save_analysis

# Analyze a single case
analysis = analyze_case("path/to/case_file.txt")

# Save the analysis results
save_analysis(analysis, "analysis_results")
```

## 📁 Project Structure

```
.
├── core.py                      # Core framework components
├── advanced_analysis.py         # Advanced code analysis utilities
├── patch_ana.py                 # Patch analysis script
├── Examples.pdf                 # Examples we listed for each category
├── model_failed_cases/          # Directory containing case files
└── patch_analysis_results/      # Directory for analysis results
```

