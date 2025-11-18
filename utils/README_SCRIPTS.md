# Analysis Scripts Documentation

This document describes the three utility scripts created for analyzing SWE-bench evaluation results.

## Script 1: Extract Resolved Instances

**File:** `extract_resolved_instances.py`

### Purpose
Extracts unique resolved instance IDs from all JSON evaluation files in the `Ablations/Pagent` folder.

### Usage
```bash
python utils/extract_resolved_instances.py
```

### Output
- Creates `resolved_instances_unique.txt` in the project root
- Contains:
  - Total count of unique resolved instances
  - Complete list of resolved instance IDs (sorted)

### Example Output Format
```
Unique Resolved Instance IDs
============================================================
Total Count: 42
============================================================

astropy__astropy-6938
django__django-11910
matplotlib__matplotlib-25311
...
```

---

## Script 2: Calculate Timing Statistics

**File:** `calculate_timing_stats.py`

### Purpose
Calculates aggregate timing statistics from `pagent_runs/pagent/processing_results.jsonl` for:
- Static analysis time
- Type inference (LLM) time
- Patch rewriting (LLM) time

Reports: median, mean, P90, min, max, and standard deviation for each component.

### Usage
```bash
python utils/calculate_timing_stats.py
```

### Output
- Creates `timing_statistics.csv` in the project root
- CSV with columns: component, median, mean, p90, min, max, std_dev, count
- Also prints detailed statistics to console

### Example Output
```csv
component,median,mean,p90,min,max,std_dev,count
Static Analysis,0.850,0.912,1.234,0.450,2.100,0.345,215
Type Inference (LLM),8.456,9.123,15.678,2.123,45.234,6.789,215
Patch Rewriting (LLM),5.234,6.789,12.345,1.234,34.567,4.567,215
```

---

## Script 3: Classify Subcategories

**File:** `classify_subcategories.py`

### Purpose
Takes the voted taxonomy results and classifies each patch into a subcategory using LLM (GPT-5-mini).

### Subcategory Mapping

**Class 1: Type & Data-Shape/Schema Mismanagement**
- Basic type conversions (string to integer, etc.)
- Complex data structure manipulations (numpy array to pandas dataframe, etc.)
- Missing type validation (dtype/nullable dtypes, shape/axis/index alignment, numeric precision/range)

**Class 2: Contract/Architecture Violation**
- Inheritance/Dispatch Misunderstanding (override/abstract method contracts)
- Protocol & Return-Contract Issues (call order, expected outputs)
- Lifecycle/Config Semantics (state, init/finalize, feature flags)

**Class 3: Fault & Edge-Condition Handling**
- Exception Handling & Propagation (catch/raise, preserving cause)
- Boundary & Edge Conditions (off-by-one, empty/rectangular inputs)
- Incomplete Fix Scope (guards added in one place but missed siblings/callers)

**Class 4: Framework/Abstraction Bypass**
- Reimplementing Existing Helpers (reinvented wheels)/Creating Redundant methods
- Ignoring Framework-Specific Patterns

**Class 5: Version/Compatibility Drift**
- Misunderstanding API Changes Across Versions
- Missing Version Gates / Deprecated Features (no compatibility path)

**Note:** Class 6 (Algorithmic Inefficiency) is skipped as per requirements.

### Usage

Basic usage:
```bash
python utils/classify_subcategories.py
```

With custom options:
```bash
python utils/classify_subcategories.py \
    --voted-csv taxonomy_results_voted.csv \
    --cases-dir model_failed_cases \
    --output subcategory_classification_results.csv \
    --model openai/gpt-5-mini-2025-08-07 \
    --verbose
```

### Arguments
- `--voted-csv`: Path to taxonomy_results_voted.csv (default: taxonomy_results_voted.csv)
- `--cases-dir`: Directory with patch case files (default: model_failed_cases)
- `--output`: Output CSV file (default: subcategory_classification_results.csv)
- `--limit`: Process only N entries (default: 0 = all)
- `--model`: LLM model to use (default: openai/gpt-5-mini-2025-08-07)
- `--retry`: Number of retries per classification (default: 2)
- `--verbose`: Print detailed progress

### Output
- Creates `subcategory_classification_results.csv` in the project root
- CSV with columns: instance_id, model_name, class, subcategory, explanation

### Example Output
```csv
instance_id,model_name,class,subcategory,explanation
astropy__astropy-14182,Aider,1,Basic type conversions (string to integer etc.),The patch converts string values to appropriate numeric types for array indexing
django__django-11910,Agentless GPT 4o,1,Missing type validation (dtype/nullable dtypes shape/axis/index alignment numeric precision/range),Added validation to ensure correct dtype before processing
...
```

### Important Notes
1. **LLM does not alter the class**: The script explicitly instructs the LLM to accept the provided class and only choose a subcategory
2. **Structured output**: Uses Pydantic models to ensure consistent JSON responses
3. **Retry logic**: Automatically retries failed classifications with truncated content
4. **Error handling**: Captures errors in the output CSV for manual review

---

## Dependencies

All scripts require:
- Python 3.10+
- `pydantic` (for Script 3)
- Access to `llm_provider` module (for Script 3)

Install dependencies:
```bash
pip install pydantic
```

---

## Quick Start: Run All Scripts

```bash
# 1. Extract resolved instances
python utils/extract_resolved_instances.py

# 2. Calculate timing statistics
python utils/calculate_timing_stats.py

# 3. Classify subcategories (this will take longer due to LLM calls)
python utils/classify_subcategories.py --verbose
```

---

## Troubleshooting

### Script 1: No JSON files found
- Ensure `Ablations/Pagent` directory exists and contains JSON files

### Script 2: Source file not found
- Verify `pagent_runs/pagent/processing_results.jsonl` exists
- Check the file path in the script if data is stored elsewhere

### Script 3: Patch file not found
- Ensure `model_failed_cases` directory contains the expected `.txt` files
- Files should be named: `{instance_id}_{model_name}.txt`
- Model name in filename should have underscores instead of spaces

### Script 3: LLM errors
- Check API credentials for GPT-5-mini
- Reduce `--limit` for testing
- Increase `--retry` for flaky connections
- Check token limits if patches are very large

---

## Output Files Summary

After running all scripts, you will have:

1. `resolved_instances_unique.txt` - List of unique resolved instance IDs
2. `timing_statistics.csv` - Aggregate timing metrics
3. `subcategory_classification_results.csv` - Subcategory classifications for each patch

These files can be used for further analysis, paper writing, or visualization.
