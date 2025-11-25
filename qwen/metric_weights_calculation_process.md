# Process of Calculating Metric Weights: A Step-by-Step Documentation

## Overview

This document provides a detailed, step-by-step documentation of the process used to calculate metric weights in the FRAGSTATS agricultural land evaluation study. The methodology employs a hierarchical entropy weighting approach across four stages.

## Stage 1: Data Preparation and Normalization

### Step 1.1: Data Collection
- **Input**: 30 FRAGSTATS class-level metrics for 4 spatial information layers (Infra, Pibok, Nongeup, Toyang3)
- **Evaluation objects**: 16 combinations of (2 regions × 4 layers × 2 classes) = 16 objects
  - Regions: Hwasun, Naju
  - Layers: Infrastructure, Land Use Complexity, Rice Suitability, Soil Conditions
  - Classes: cls_1 (suitable), cls_9 (unsuitable)

### Step 1.2: Min-Max Normalization
- **Purpose**: Convert metrics with different units and scales to a common 0-1 range
- **Formula**: 
  $$x_{ij}' = \\frac{x_{ij} - x_{j\\_min}}{x_{j\\_max} - x_{j\\_min}}$$
- **Where**: 
  - $x_{ij}$ = original value of i-th evaluation object on j-th metric
  - $x_{ij}'$ = normalized value
  - $x_{j\\_min}$ = minimum value across all objects for metric j
  - $x_{j\\_max}$ = maximum value across all objects for metric j

### Step 1.3: Handling Special Cases
- **Zero denominators**: If $x_{j\\_max} - x_{j\\_min} = 0$, apply small constant to avoid division by zero
- **Negative values**: Min-max normalization handles negative values naturally
- **Outliers**: The method is sensitive to outliers, but this is desired to capture variation

## Stage 2: Intragroup Weight Calculation

### Step 2.1: Proportion Matrix Calculation
- **Formula**: 
  $$p_{ij} = \\frac{x_{ij}'}{\\sum_{i=1}^{m} x_{ij}'}$$
- **Where**:
  - $p_{ij}$ = proportion of i-th object in the total for j-th metric
  - $m$ = total number of evaluation objects (m=16 in this study)
- **Purpose**: Convert normalized values to probability-like proportions

### Step 2.2: Information Entropy Calculation for Each Metric
- **Formula**:
  $$E_j = -k \\sum_{i=1}^{m} p_{ij} \\ln(p_{ij})$$
- **Where**:
  - $k = \\frac{1}{\\ln(m)}$ = normalization constant
  - $E_j$ = information entropy of j-th metric
- **Special case**: If $p_{ij} = 0$, then $p_{ij} \\ln(p_{ij}) = 0$
- **Range**: $0 \\leq E_j \\leq 1$, where $E_j = 1$ indicates maximum entropy (no information value)

### Step 2.3: Information Utility Value Calculation
- **Formula**:
  $$d_j = 1 - E_j$$
- **Where**:
  - $d_j$ = information utility value (diversity) of j-th metric
- **Range**: $0 \\leq d_j \\leq 1$, where higher values indicate more useful information
- **Interpretation**: Metrics with greater variation across objects have higher $d_j$

### Step 2.4: Weight Calculation Within Each Group
- **Formula**:
  $$W_j = \\frac{d_j}{\\sum_{j=1}^{n} d_j}$$
- **Where**:
  - $W_j$ = weight of j-th metric within its group
  - $n$ = number of metrics in the group
- **Constraint**: $\\sum_{j=1}^{n} W_j = 1$ for each group
- **Application**: Calculate weights separately for each of the 4 groups:
  - Area-Density group: 9 metrics
  - Shape-Edge group: 7 metrics  
  - Core group: 6 metrics
  - Aggregation group: 8 metrics

## Stage 3: Group Score Calculation

### Step 3.1: Weighted Group Score Computation
- **Formula**:
  $$Score_{layer,group} = \\sum_{j \\in group} (x_{ij}' \\times W_j)$$
- **Where**:
  - $x_{ij}'$ = normalized value of i-th evaluation object (16 total objects)
  - $W_j$ = weight of j-th metric within the group
  - $Score_{layer,group}$ = composite score for a layer-group combination
- **Interpretation**: Each score represents the overall performance of a land class on a specific structural dimension

### Step 3.2: Group Score Interpretation
- **Range**: $0 \\leq Score_{layer,group} \\leq 1$
- **Higher scores**: Better performance on that structural dimension
- **Applications**: Used for within-group analysis and cross-group comparison

## Stage 4: Discriminability Calculation

### Step 4.1: Class Difference Calculation
- **Formula**:
  $$D_{layer,group} = |Score_{cls1,group} - Score_{cls9,group}|$$
- **Where**:
  - $D_{layer,group}$ = discriminability of layer for group
  - $Score_{cls1,group}$ = group score for suitable class (cls_1)
  - $Score_{cls9,group}$ = group score for unsuitable class (cls_9)
- **Purpose**: Measures how well a layer can distinguish between suitable and unsuitable land

### Step 4.2: Total Discriminability Calculation
- **Formula**:
  $$D_{layer,total} = \\sum_{group=1}^{4} D_{layer,group}$$
- **Where**:
  - $D_{layer,total}$ = total discriminability of the layer
- **Interpretation**: Higher values indicate better ability to differentiate land classes

### Step 4.3: Discriminability Analysis
- **Analysis**: Compare discriminability values across the 4 layers (Infra, Pibok, Nongeup, Toyang3)
- **Purpose**: Determine which spatial information layer best differentiates suitable from unsuitable agricultural land

## Stage 5: Final Layer Weight Calculation

### Step 5.1: Layer Entropy Calculation
- **Formula** (for each layer):
  $$E_{layer} = -k \\sum_{layer=1}^{4} p_{layer} \\ln(p_{layer})$$
- **Where**:
  - $k = \\frac{1}{\\ln(4)}$ (since there are 4 layers)
  - $p_{layer} = \\frac{D_{layer,total}}{\\sum_{layer=1}^{4} D_{layer,total}}$ (proportion of discriminability)
  - $E_{layer}$ = entropy of the layer based on its discriminability

### Step 5.2: Layer Information Utility
- **Formula**:
  $$d_{layer} = 1 - E_{layer}$$
- **Where**:
  - $d_{layer}$ = information utility of the layer
- **Interpretation**: Higher values indicate greater importance of the layer

### Step 5.3: Final Layer Weight Calculation
- **Formula**:
  $$W_{layer} = \\frac{d_{layer}}{\\sum_{layer=1}^{4} d_{layer}}$$
- **Where**:
  - $W_{layer}$ = final weight of the layer
- **Constraint**: $\\sum_{layer=1}^{4} W_{layer} = 1$
- **Interpretation**: Percentage of importance for each spatial information layer

## Quality Assurance and Validation Steps

### Step 6.1: Consistency Checks
- **Weight Sum Validation**: Verify that weights within each group sum to 1.0
- **Range Validation**: Confirm that all weights are between 0 and 1
- **Cross-Stage Validation**: Ensure results make logical sense when compared between stages

### Step 6.2: Sensitivity Analysis
- **Outlier Testing**: Verify that extreme values don't unduly influence results
- **Stability Testing**: Test whether small changes in input data significantly affect weights
- **Robustness Testing**: Confirm that the approach yields meaningful results across different regions

## Implementation Considerations

### Step 7.1: Software Implementation
- **Tools**: Calculations can be implemented using Python, R, or MATLAB
- **Verification**: Use multiple software tools to verify consistency of results
- **Automation**: The process can be automated for different datasets or regions

### Step 7.2: Data Quality Requirements
- **Input Requirements**: Complete data for all evaluation objects and metrics
- **Missing Data Handling**: Strategies for handling missing values if present
- **Data Preprocessing**: Standard procedures for data cleaning and validation

### Step 7.3: Interpretation Guidelines
- **Threshold Setting**: Establish thresholds for meaningful weight differences
- **Contextual Consideration**: Interpret weights within the specific agricultural and regional context
- **Policy Translation**: Convert metric weights into actionable policy recommendations

## Example Calculation Walkthrough

For the Area-Density group in the study:

1. **Normalization**: All 9 metrics normalized to 0-1 range across 16 evaluation objects
2. **Proportion Calculation**: Each normalized value converted to proportion of total
3. **Entropy Calculation**: Information entropy calculated for each metric
4. **Utility Values**: $d_j = 1 - E_j$ calculated for each metric
5. **Weights**: $W_j = d_j / \\sum d_j$ yielding weights that sum to 1.0
6. **Results**: AREA_MN achieved 38.09% weight due to its high discriminatory power

## Conclusion

The hierarchical entropy weighting process systematically transforms 30 FRAGSTATS metrics into interpretable weights that reflect the relative importance of different landscape structural properties for agricultural land evaluation. This objective approach avoids subjective bias while providing clear insights into which aspects of landscape structure are most relevant for distinguishing land quality. The step-by-step process ensures transparency, reproducibility, and robustness of the final weighting scheme.