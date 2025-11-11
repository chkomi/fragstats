
import json
import math

def define_indicators():
    """
    Defines the direction for each indicator.
    This is copied from entropy_weight_model.py for consistency.
    """
    return {
        'CA': {'direction': 'positive'}, 'PLAND': {'direction': 'positive'}, 'LPI': {'direction': 'positive'},
        'NP': {'direction': 'negative'}, 'PD': {'direction': 'negative'}, 'ED': {'direction': 'negative'},
        'AREA_MN': {'direction': 'positive'}, 'AREA_MD': {'direction': 'positive'}, 'GYRATE_MN': {'direction': 'positive'},
        'TCA': {'direction': 'positive'}, 'CPLAND': {'direction': 'positive'}, 'CORE_MN': {'direction': 'positive'},
        'CLUMPY': {'direction': 'positive'}, 'PLADJ': {'direction': 'positive'}, 'AI': {'direction': 'positive'},
        'COHESION': {'direction': 'positive'},
        'SHAPE_MN': {'direction': 'negative'}, 'FRAC_MN': {'direction': 'negative'},
        'NDCA': {'direction': 'negative'}, 'DCAD': {'direction': 'negative'},
        # Indicators from the report that are not in the model's define_indicators function
        'TE': {'direction': 'negative'},
        'AREA_AM': {'direction': 'positive'},
        'AREA_CV': {'direction': 'negative'},
        'GYRATE_AM': {'direction': 'positive'},
        'GYRATE_MD': {'direction': 'positive'},
        'GYRATE_CV': {'direction': 'negative'},
        'SHAPE_AM': {'direction': 'negative'},
        'FRAC_AM': {'direction': 'negative'},
        'PARA_MN': {'direction': 'negative'},
        'CAI_MN': {'direction': 'positive'},
    }

def normalize_data(matrix, directions):
    """
    Normalizes the entire data matrix.
    Matrix is a dict of lists {indicator: [values...]}.
    Directions is a dict {indicator: 'positive'/'negative'}.
    """
    normalized_matrix = {}
    for indicator, values in matrix.items():
        direction = directions.get(indicator, {'direction': 'positive'})['direction']
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            normalized_matrix[indicator] = [None] * len(values)
            continue

        min_val = min(valid_values)
        max_val = max(valid_values)

        if max_val == min_val:
            normalized_matrix[indicator] = [0.5 if v is not None else None for v in values]
            continue

        normalized_values = []
        for v in values:
            if v is None:
                normalized_values.append(None)
            elif direction == 'positive':
                normalized_values.append((v - min_val) / (max_val - min_val))
            else: # negative
                normalized_values.append((max_val - v) / (max_val - min_val))
        normalized_matrix[indicator] = normalized_values
    return normalized_matrix

def calculate_entropy_and_weights(normalized_matrix):
    """Calculates entropy and weights for each indicator."""
    entropies = {}
    num_datapoints = len(next(iter(normalized_matrix.values())))
    k = 1.0 / math.log(num_datapoints)
    
    for indicator, norm_values in normalized_matrix.items():
        valid_values = [v for v in norm_values if v is not None]
        if not valid_values:
            entropies[indicator] = 1 # Max entropy
            continue

        # Add epsilon for stability if a value is 0
        epsilon = 1e-10
        sum_norm_values = sum(valid_values)
        
        if sum_norm_values == 0:
             proportions = [1.0/len(valid_values) for v in valid_values]
        else:
            proportions = [(v + epsilon) / (sum_norm_values + len(valid_values) * epsilon) for v in valid_values]

        entropy = -k * sum(p * math.log(p) for p in proportions if p > 0)
        entropies[indicator] = entropy

    diversities = {indicator: 1 - e for indicator, e in entropies.items()}
    total_diversity = sum(diversities.values())

    if total_diversity == 0:
        num_indicators = len(entropies)
        weights = {indicator: 1.0 / num_indicators for indicator in entropies}
    else:
        weights = {indicator: d / total_diversity for indicator, d in diversities.items()}
        
    return entropies, weights

def main():
    # Load the data extracted from the report
    with open('/Users/yunhyungchang/Documents/FRAGSTATS/extracted_metrics.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_points_order = list(data.keys())
    
    # Get all unique indicators from the data
    all_indicators = set()
    for dp_data in data.values():
        all_indicators.update(dp_data.keys())
    
    indicator_definitions = define_indicators()
    # Filter to only indicators defined in the original model
    model_indicators = list(indicator_definitions.keys())

    # Create the data matrix
    matrix = {indicator: [] for indicator in model_indicators}
    for dp_name in data_points_order:
        for indicator in model_indicators:
            matrix[indicator].append(data[dp_name].get(indicator))

    # Perform calculations
    normalized_matrix = normalize_data(matrix, indicator_definitions)
    entropies, weights = calculate_entropy_and_weights(normalized_matrix)

    # Calculate comprehensive scores
    scores = []
    for i, dp_name in enumerate(data_points_order):
        score = 0
        for indicator in model_indicators:
            norm_value = normalized_matrix[indicator][i]
            if norm_value is not None:
                score += norm_value * weights[indicator]
        scores.append({"data_point": dp_name, "score": score * 100}) # Scale to 100

    # --- Output Results ---
    print("--- Verification Results ---")
    
    print("\n1. Calculated Indicator Weights (Verification):")
    print(f"{'Indicator':<12} | {'Weight':<10}")
    print("-" * 25)
    sorted_weights = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    for indicator, weight in sorted_weights:
        print(f"{indicator:<12} | {weight:<10.4f}")

    print("\n2. Calculated Comprehensive Scores (Verification):")
    print(f"{'Data Point':<20} | {'Score':<10}")
    print("-" * 33)
    sorted_scores = sorted(scores, key=lambda item: item['score'], reverse=True)
    for item in sorted_scores:
        print(f"{item['data_point']:<20} | {item['score']:<10.2f}")
        
    # You can manually compare these results with '엔트로피가중치_평가모델_결과.txt'
    # The bug in the original script's calculate_comprehensive_score function
    # means the scores will be very different. The weights should be similar though.

if __name__ == "__main__":
    main()
