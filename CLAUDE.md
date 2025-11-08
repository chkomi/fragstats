# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains tools for a multi-level landscape metric analysis of agricultural land in Naju (나주) and Hwasun (화순).

The primary goal is to establish objective, data-driven criteria for evaluating Agricultural Promotion Areas. The core of the project is an **advanced entropy weight model** that integrates metrics from three different spatial scales (Class, Land, and Patch) to derive a comprehensive "Agricultural Value" score.

## Data Structure

### Input Files
FRAGSTATS output files are tab-separated text files organized by:
- **Categories**: `infra`, `toyang`, `nongeup`, `pibok`
- **Analysis Levels**:
  - `*_class.txt` - Class-level metrics
  - `*_land.txt` - Landscape-level metrics
  - `*_patch.txt` - Patch-level metrics
- **Regions**: `naju`, `hwasun`

### Agricultural Suitability Types
The analysis focuses on the "suitable" land types within each category:
- **infra**: `giban_benefited`
- **toyang/nongeup/pibok**: `cls_1`

---

## Current Analysis Workflow

The entire analysis, from data aggregation to final report generation, is now handled by a single, comprehensive script.

### **Main Script: `advanced_entropy_model.py`**

This script performs a complete, end-to-end analysis.

#### **Process:**

1.  **Data Aggregation:** The script reads data from all three levels (`class`, `land`, `patch`) for the 8 evaluation targets (2 regions × 4 categories). It unifies 58+ indicators into a single data matrix.
2.  **Entropy Analysis:** It applies the entropy weight method to the entire data matrix to calculate an objective, data-driven weight for every single indicator.
3.  **Scoring & Reporting:** Using these weights, it calculates a final comprehensive "Agricultural Value" score for each of the 8 targets and generates a series of detailed reports documenting the process and results.

#### **Running the Analysis:**

To execute the full analysis and regenerate all reports, run:
```bash
python3 advanced_entropy_model.py
```

---

## Key Outputs & Reports

The analysis generates the following key documents, which are the main results of this project:

-   **`1. ... .md`**: Initial verification reports that identify and explain the critical bug found in the original, now-deprecated analysis script.
-   **`2. ... .md`**: A summary of the final, corrected analysis results, formatted for easy inclusion in a research paper.
-   **`3. ... .md`**: The detailed final analysis report. This includes the complete calculation process and the full list of all 58+ indicator weights and their rationales, ensuring full verifiability.
-   **`4. ... .md`**: Documents outlining the next steps for policy application. This includes a proposal for using the results in a GIS-based weighted overlay analysis and the final data-driven layer weights derived from the model's output.

---

## Key Results

The main quantitative outcomes of the analysis are:

#### Final "Agricultural Value" Scores:

| Rank | Target ID | Score |
| :--: | :--- | :---: |
| 1 | `naju_nongeup` | **74.08** |
| 2 | `hwasun_nongeup` | **62.41** |
| 3 | `hwasun_toyang` | **59.32** |
| 4 | `naju_toyang` | **56.53** |
| 5 | `hwasun_infra` | **44.64** |
| 6 | `naju_infra` | **38.81** |
| 7 | `hwasun_pibok` | **28.03** |
| 8 | `naju_pibok` | **23.54** |

#### Data-Driven Layer Weights:

These weights, calculated from the analysis results, represent the relative importance of each layer in determining the final agricultural value. They are intended for use in GIS weighted overlay analysis.

| Rank | Layer | Weight |
| :--: | :--- | :---: |
| 1 | **Nongeup (농업용도)** | **35.2%** |
| 2 | **Toyang (토양)** | **29.9%** |
| 3 | **Infra (기반시설)** | **21.5%** |
| 4 | **Pibok (토지피복)** | **13.3%** |

---

## Deprecated Scripts

The following scripts are now superseded by `advanced_entropy_model.py` and should not be used for analysis. They are retained in the repository for historical context regarding the project's evolution and the bug discovery process.

-   `analyze_fragstats.py`
-   `entropy_weight_model.py`
-   `verify_data.py`, `recalculate_entropy.py`
