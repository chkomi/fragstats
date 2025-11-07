# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains FRAGSTATS landscape metrics analysis tools for agricultural land use planning, specifically focused on establishing criteria for agricultural promotion area designation/de-designation in Naju (나주) and Hwasun (화순) regions.

## Data Structure

### Input Files
FRAGSTATS output files are tab-separated text files organized by:
- **Categories**: `infra`, `toyang`, `nongeup`, `pibok`
- **Analysis Levels**:
  - `*_class.txt` - Class-level metrics (31 indicators)
  - `*_land.txt` - Landscape-level metrics (19 indicators)
  - `*_patch.txt` - Patch-level metrics (8 indicators)
- **Regions**: `naju`, `hwasun` (embedded in LOC field or filename)

Special cases:
- `pibok` patch data is split into separate files: `pibok_patch_naju.txt` and `pibok_patch_hwasun.txt`
- Patch files are very large (1-2MB) and should be processed with chunking or streaming

### Agricultural Suitability Types
- **infra**: `giban_benefited` (suitable) vs `giban_not_benefited` (unsuitable)
- **toyang/nongeup/pibok**: `cls_1` (suitable) vs `cls_9` (unsuitable)

## Running Analysis

### Primary Command
```bash
python3 analyze_fragstats.py
```

This generates `FRAGSTATS_분석결과_종합보고서.txt` containing:
1. Class-level analysis with suitable vs unsuitable type comparisons
2. Landscape-level analysis with regional (Naju vs Hwasun) comparisons
3. Patch-level descriptive statistics (mean, std, min, max, median)
4. Comprehensive summary with recommendations for agricultural land designation criteria

### Dependencies
The script uses **only Python standard library** (no pandas/numpy). This is intentional to avoid dependency issues.

## Entropy Weight Model

A new, objective evaluation model has been introduced to calculate the importance of different landscape metrics.

### Running the Model
```bash
python3 entropy_weight_model.py
```
This script reads the class-level metrics from `FRAGSTATS_분석결과_종합보고서.txt`, applies the entropy weight method, and generates two key files:
1.  `엔트로피가중치_평가모델_결과.txt`: Contains the calculated weights for each layer (pibok, nongeup, infra, toyang) and each of the 31 class-level metrics. It also includes the comprehensive scores and 5-tier ratings for each data point (Naju/Hwasun).
2.  `엔트로피가중치법_방법론_설명서.txt`: A detailed document explaining the theory, calculation steps, and policy application of the entropy weight method.

### Code Architecture

**`entropy_weight_model.py`**
- **`load_class_metrics_from_report(report_path)`**: Parses the main analysis report to extract the raw data for the 8 data points (Naju/Hwasun x 4 layers).
- **`calculate_entropy_weights(data)`**: Implements the 6-step entropy weight calculation process:
    1.  Normalization of the data matrix.
    2.  Calculation of entropy for each indicator.
    3.  Calculation of the degree of dispersion.
    4.  Calculation of the weight for each indicator.
    5.  Calculation of comprehensive scores for each data point.
    6.  Classification into a 5-tier rating system (Absolute Preservation to Priority De-designation).
- **`save_results(weights, scores, ratings)`**: Saves the final weights, scores, and ratings into `엔트로피가중치_평가모델_결과.txt`.

This model provides an objective, data-driven method for evaluating agricultural promotion areas, removing subjectivity from the weighting process.

## Main Analysis Functions

**`analyze_class_metrics()`**
- Reads 4 category files (infra, toyang, nongeup, pibok)
- Outputs metrics for each region/type combination
- Compares suitable vs unsuitable agricultural types
- Key metrics: CA, PLAND, NP, PD, LPI, ED, AREA_MN, TCA, CPLAND, CLUMPY, PLADJ, COHESION, AI

**`analyze_land_metrics()`**
- Landscape-level analysis for each category
- Regional comparisons (Hwasun vs Naju)
- Key metrics: TA, NP, PD, LPI, ED, TCA, CONTAG, COHESION, DIVISION, MESH, SPLIT, SHDI, SIDI, SHEI, AI

**`analyze_patch_metrics()`**
- Processes large patch files with streaming (line-by-line reading)
- Computes descriptive statistics per region/type
- Handles pibok's split files separately
- Key metrics: AREA, PERIM, GYRATE, SHAPE, FRAC, CORE, NCORE, CAI

**`generate_summary()`**
- Provides interpretation guidelines for FRAGSTATS indicators
- Proposes criteria for agricultural land preservation vs de-designation
- Suggests weighted scoring model for comprehensive evaluation

### Utility Functions

**`read_fragstats_file(file_path)`**
- Parses tab-separated FRAGSTATS output
- Returns dict with 'header' and 'data' (list of dicts)
- Use for small files (class, land levels)

**`safe_float(value)`**
- Handles 'N/A' values and conversion errors
- Returns None for non-numeric values

**`calculate_stats(values)`**
- Computes mean, std, min, max, median
- Filters out None values automatically

## Important Implementation Notes

### Working Directory
The script has a hardcoded path in line 13:
```python
work_dir = Path("/Users/yunhyungchang/Documents/FRAGSTATS")
```
When adapting this script for other systems, update this path accordingly.

### Large File Handling
Patch files (especially `infra_patch.txt`, `pibok_patch_*.txt`) are too large to load into memory. The code uses:
```python
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:  # Stream line-by-line
        # process each patch
```

### Text Encoding
All files use UTF-8 encoding. Korean text is extensively used in variable names, comments, and output.

## Key FRAGSTATS Metrics

High-level interpretation for agricultural land evaluation:

**Preservation Priority Indicators:**
- High PLAND (landscape percentage)
- High LPI (large connected patches)
- High CLUMPY (>0.85), AI (>90), COHESION (>98) - concentrated distribution
- Low NP, PD, ED - minimal fragmentation

**De-designation Consideration Indicators:**
- Low PLAND (small landscape share)
- High NP, PD, ED - high fragmentation
- Low CLUMPY (<0.8), AI (<85) - dispersed distribution
- Small AREA_MN - tiny average patch size

## Output Format

The analysis report uses Korean text with structured sections:
- Fixed-width column formatting for numeric tables
- 80-character section dividers
- Difference values and percentage changes for comparisons
- Summary recommendations at the end
