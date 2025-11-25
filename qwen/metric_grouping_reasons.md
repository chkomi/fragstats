# Grouping of 30 FRAGSTATS Metrics into 4 Categories: Rationale and Reasoning

## Overview

The 30 FRAGSTATS Class-level metrics were systematically grouped into 4 meaningful categories based on their conceptual properties and the aspects of landscape structure they measure. This grouping serves multiple analytical purposes and reflects established landscape ecology principles.

## The 4 Metric Categories and Their Rationale

### 1. Area-Density Group (9 metrics)

**Rationale**: This group focuses on the size, number, and distribution of patches within each land class. These metrics provide fundamental information about landscape composition and fragmentation intensity.

**Metrics included:**
- AREA_MN (Mean patch area)
- AREA_AM (Area-weighted mean)
- AREA_MD (Median area)
- LPI (Largest patch index)
- CA (Total class area)
- PLAND (Percentage of landscape)
- AREA_CV (Coefficient of variation of area)
- NP (Number of patches)
- PD (Patch density)

**Conceptual basis**: These metrics all relate to the dimensional properties of patches - their absolute and relative sizes, their abundance, and their proportional representation in the landscape. They collectively measure the extent of spatial fragmentation and patch aggregation at the most basic level.

### 2. Shape-Edge Group (7 metrics)

**Rationale**: This group evaluates the geometric complexity and edge characteristics of patches. These metrics assess how irregular patch shapes affect ecological and agricultural functions.

**Metrics included:**
- PARA_MN (Mean perimeter/area ratio)
- TE (Total edge length)
- ED (Edge density)
- FRAC_MN (Mean fractal dimension)
- FRAC_AM (Area-weighted fractal dimension)
- SHAPE_MN (Mean shape index)
- SHAPE_AM (Area-weighted shape index)

**Conceptual basis**: These metrics quantify the complexity of patch boundaries and their geometric properties. They measure how much edge exists relative to patch area and the complexity of patch shapes, which affects edge effects, management efficiency, and resource access in agricultural contexts.

### 3. Core Group (6 metrics)

**Rationale**: This group assesses the interior habitat that is not affected by edge effects. In agricultural terms, this represents the "core" productive area of fields that is not influenced by boundary conditions.

**Metrics included:**
- CORE_MN (Mean core area)
- TCA (Total core area)
- CPLAND (Core area percentage of landscape)
- CAI_MN (Mean core area index)
- NDCA (Number of disjunct core areas)
- DCAD (Core area density)

**Conceptual basis**: These metrics focus on habitat/core area that is sufficiently interior to patches to be unaffected by edge influences. In agricultural contexts, this relates to the productive area that is not affected by edge effects such as shading, wind exposure, or field border management practices.

### 4. Aggregation Group (8 metrics)

**Rationale**: This group measures the spatial arrangement and connectivity of patches within each class. These metrics indicate how patches are clustered together versus being dispersed across the landscape.

**Metrics included:**
- GYRATE_MN (Mean gyration radius)
- GYRATE_MD (Median gyration radius)
- GYRATE_AM (Area-weighted gyration radius)
- GYRATE_CV (Coefficient of variation of gyration radius)
- AI (Aggregation index)
- CLUMPY (Clumpiness index)
- PLADJ (Percentage of like adjacencies)
- COHESION (Cohesion index)

**Conceptual basis**: These metrics quantify spatial clustering and connectivity patterns. They measure how patches of the same class are positioned relative to each other, indicating the degree of spatial aggregation versus dispersion, which affects ecological connectivity and agricultural operational efficiency.

## Theoretical Foundation for Grouping

### 1. FRAGSTATS Conceptual Framework

The grouping follows the established FRAGSTATS framework, which organizes landscape metrics according to the aspects of spatial pattern they measure:
- **Composition metrics**: How much of each patch type exists (represented by Area-Density group)
- **Configuration metrics**: How patches are shaped and arranged (represented by Shape-Edge and Aggregation groups)
- **Connectivity metrics**: How patches are connected spatially (represented by Core and Aggregation groups)

### 2. Analytical Efficiency and Interpretability

The four-group structure provides analytical benefits:
- **Reduces complexity**: Instead of analyzing 30 individual metrics, the analysis can focus on 4 key dimensions
- **Maintains ecological meaning**: Each group represents a coherent aspect of landscape structure
- **Facilitates comparison**: Results can be compared across groups to understand which landscape properties are most important
- **Enables hierarchical analysis**: Group-level weights can be calculated before overall metric weights

### 3. Application Relevance for Agricultural Evaluation

The grouping is particularly relevant for agricultural land evaluation:
- **Area-Density**: Measures the scale and extent of agricultural land, affecting economies of scale
- **Shape-Edge**: Measures field complexity, affecting management efficiency and machinery operations
- **Core**: Measures productive land area that is not affected by field boundaries
- **Aggregation**: Measures spatial clustering, affecting operational efficiency and infrastructure access

## Methodological Advantages

### 1. Reduced Multicollinearity

Grouping metrics with similar conceptual properties helps manage multicollinearity issues by:

- Identifying metrics that measure similar landscape properties
- Allowing internal group comparisons to determine relative importance
- Reducing the risk of double-counting similar information in the overall analysis

### 2. Hierarchical Weighting Approach

The group structure enables a hierarchical entropy analysis:

1. **Intragroup weights**: Calculate weights for metrics within each group
2. **Group scores**: Calculate overall scores for each group based on the weighted metrics
3. **Group weights**: Determine the relative importance of each group dimension
4. **Final layer weights**: Calculate the final weights for the spatial information layers

### 3. Interpretation and Policy Implications

The four-group structure provides clear interpretation pathways:

- Understanding which dimension of landscape structure is most important for agricultural land quality
- Identifying the specific properties that make land suitable or unsuitable
- Providing targeted recommendations based on which group shows the strongest discrimination between suitable and unsuitable land

## Conclusion

The four-category grouping of 30 FRAGSTATS metrics reflects both theoretical landscape ecology principles and practical analytical needs. This grouping strategy enables comprehensive evaluation of agricultural land quality while maintaining analytical tractability and interpretability. The structure allows for nuanced understanding of how different aspects of landscape spatial configuration contribute to overall agricultural land value, supporting evidence-based agricultural policy decisions.