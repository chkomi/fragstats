# Why 30 Metrics Were Selected in the FRAGSTATS Analysis

## Overview

This analysis addresses why 30 specific metrics from FRAGSTATS were selected for the agricultural land evaluation study in the Naju and Hwasun regions. The choice was driven by the need to comprehensively evaluate agricultural landscape structures from multiple dimensions while maintaining analytical efficiency.

## Key Reasons for Selecting 30 Metrics

### 1. Comprehensive Coverage of Landscape Characteristics

The 30 metrics were selected to thoroughly evaluate different aspects of agricultural landscape structure:

- **Size and Density Metrics (9 metrics)**: Including AREA_MN (mean patch area), PD (patch density), and CA (total area), these metrics assess the spatial extent and distribution of agricultural areas.
- **Shape and Edge Metrics (7 metrics)**: Including PARA_MN (perimeter/area ratio), ED (edge density), and SHAPE_MN (shape index), these metrics evaluate the geometric complexity and edge effects of agricultural patches.
- **Core Area Metrics (6 metrics)**: Including CORE_MN (mean core area), TCA (total core area), and CPLAND (core area percentage), these metrics measure the interior habitat that is not affected by edge effects.
- **Aggregation Metrics (8 metrics)**: Including AI (aggregation index), CLUMPY (clumpiness), and GYRATE_MN (mean gyration radius), these metrics assess how patches are spatially arranged and connected.

### 2. Multi-Level Structural Analysis

The selected 30 metrics provide a comprehensive view of agricultural landscape structure at the **Class Level** specifically. This level of analysis was chosen over the Fragment (Patch) or Landscape levels because:

- The study aimed to focus on policy-relevant land classifications (suitable vs unsuitable agricultural land)
- Class-level metrics allow for meaningful comparisons between different land categories
- It maintains analytical focus while avoiding excessive complexity from patch-level analysis

### 3. Differentiation Among Spatial Information Layers

The 30 metrics were specifically selected to differentiate among four key spatial information layers:

- **Infra (Agricultural Infrastructure)**: Assessing the impact of agricultural support facilities
- **Pibok (Land Use Complexity)**: Evaluating the structural complexity of actual land use patterns
- **Nongeup (Rice Suitability Grade)**: Assessing the value of official rice suitability classification
- **Toyang3 (Soil Conditions)**: Evaluating the importance of soil quality for agricultural land

Each of the 30 metrics contributes unique information to differentiate these layers and their spatial characteristics.

### 4. Data Quality and Analytical Reliability

The final 30 metrics were chosen after screening for:

- **Low multicollinearity**: Metrics that provide independent information rather than redundant data
- **Effective discrimination**: Metrics that show measurable differences between land classes
- **Robust statistical properties**: Metrics that remain stable across different landscape configurations

### 5. Theoretical and Practical Justification

Based on FRAGSTATS theoretical framework and the specific application to agricultural evaluation:

- The metrics follow established landscape ecology principles from McGarigal & Marks (1995)
- They capture the essential structural characteristics that affect agricultural efficiency
- The selection balances comprehensiveness with analytical tractability
- The metrics have shown to be effective in agricultural landscape assessment applications

### 6. Comparison with Original Analysis

The study notes that an initial analysis used 31 metrics, but the final selection was reduced to 30. This reduction reflects a refinement process to optimize the analytical framework by eliminating any metrics that did not contribute meaningfully to the discrimination between land classes or had high correlation with other metrics.

## Conclusion

The selection of 30 metrics represents a strategic balance between comprehensive landscape assessment and analytical efficiency. These metrics provide multi-dimensional evaluation capabilities while maintaining focus on the key research question: how different spatial information layers (infrastructure, land use, rice suitability, soil) contribute to agricultural land quality from a landscape structure perspective.

This metric selection enables the study to apply entropy-based weighting methodology effectively, allowing for objective determination of which spatial information layer is most important for agricultural land evaluation in different regional contexts.