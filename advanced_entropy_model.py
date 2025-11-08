import json
import math
from pathlib import Path
from datetime import datetime
import re

# --- Configuration ---
WORK_DIR = Path("/Users/yunhyungchang/Documents/FRAGSTATS")
OUTPUT_METHODOLOGY_FILE = WORK_DIR / "종합_평가_방법론.md"
OUTPUT_RESULT_FILE = WORK_DIR / "종합_평가_결과.txt"

def safe_float(value):
    """Safely convert a value to a float, handling 'N/A' and other errors."""
    try:
        if value in ('N/A', '', None):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def read_fragstats_file(file_path):
    """Generic function to read FRAGSTATS tab-separated files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].strip().split('\t')]
        data = []
        for line in lines[1:]:
            if line.strip():
                values = [v.strip() for v in line.strip().split('\t')]
                if len(values) == len(header):
                    data.append(dict(zip(header, values)))
        return {'header': header, 'data': data}
    except Exception:
        return None

def get_patch_metrics(category, region):
    """
    Reads patch files and calculates mean values for the 8 patch-level metrics.
    Handles the split pibok files and correctly filters by region.
    """
    patch_files = []
    # The main patch files contain data for both regions, so we always read them.
    # The pibok file is special as it's pre-split by region.
    if category == 'pibok':
        patch_files.append(WORK_DIR / f"pibok_patch_{region}.txt")
    else:
        patch_files.append(WORK_DIR / f"{category}_patch.txt")

    patch_data = []
    for file_path in patch_files:
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            header_line = f.readline()
            header = [h.strip() for h in header_line.strip().split('\t')]
            
            if 'LOC' not in header or 'TYPE' not in header:
                continue

            for line in f:
                values = [v.strip() for v in line.strip().split('\t')]
                row_dict = dict(zip(header, values))
                
                # Filter for the correct region and type
                loc = row_dict.get('LOC', '')
                type_val = row_dict.get('TYPE', '')

                is_correct_region = region in loc
                is_correct_type = ('benefited' in type_val and 'not' not in type_val) or 'cls_1' in type_val
                
                if is_correct_region and is_correct_type:
                    patch_data.append(row_dict)

    if not patch_data:
        return {}

    # Calculate mean for the 8 patch metrics
    patch_indicators = ['AREA', 'PERIM', 'GYRATE', 'SHAPE', 'FRAC', 'CORE', 'NCORE', 'CAI']
    agg_metrics = {}
    for indicator in patch_indicators:
        values = [safe_float(row.get(indicator)) for row in patch_data]
        valid_values = [v for v in values if v is not None]
        if valid_values:
            agg_metrics[f"PATCH_{indicator}_MN"] = sum(valid_values) / len(valid_values)
        else:
            agg_metrics[f"PATCH_{indicator}_MN"] = None
            
    return agg_metrics

def aggregate_all_data():
    """
    Aggregates data from class, land, and patch levels for the 8 data points.
    """
    data_points = {}
    categories = ['infra', 'toyang2', 'nongeup', 'pibok']
    regions = ['hwasun', 'naju']

    for region in regions:
        for category in categories:
            # Use the base name 'toyang' for dictionary keys and properties for consistency
            report_category = 'toyang' if category == 'toyang2' else category
            key = f"{region}_{report_category}"
            data_points[key] = {'category': report_category, 'region': region, 'metrics': {}}

            # 1. Get Class Metrics
            class_file = WORK_DIR / f"{category}_class.txt"
            class_data = read_fragstats_file(class_file)
            if class_data:
                for row in class_data['data']:
                    type_val = row.get('TYPE', '')
                    loc = row.get('LOC', region)
                    if region in loc and (('benefited' in type_val and 'not' not in type_val) or 'cls_1' in type_val):
                        for k, v in row.items():
                            if k not in ['LOC', 'TYPE', 'ID']:
                                data_points[key]['metrics'][f"CLS_{k}"] = safe_float(v)
                        break

            # 2. Get Land Metrics
            land_file = WORK_DIR / f"{category}_land.txt"
            land_data = read_fragstats_file(land_file)
            if land_data:
                for row in land_data['data']:
                    loc = row.get('LOC', region)
                    if region in loc:
                        for k, v in row.items():
                            if k not in ['LOC', 'TYPE', 'ID']:
                                data_points[key]['metrics'][f"LND_{k}"] = safe_float(v)
                        break

            # 3. Get Patch Metrics (aggregated)
            patch_metrics = get_patch_metrics(category, region)
            data_points[key]['metrics'].update(patch_metrics)

    return data_points

def define_all_indicators():
    """Defines the direction ('positive' or 'negative') for all indicators."""
    # Positive: Higher value is better for agricultural land suitability
    # Negative: Lower value is better (e.g., less fragmented)
    return {
        # Class Level
        'CLS_CA': 'positive', 'CLS_PLAND': 'positive', 'CLS_NP': 'negative', 'CLS_PD': 'negative',
        'CLS_LPI': 'positive', 'CLS_TE': 'negative', 'CLS_ED': 'negative', 'CLS_AREA_MN': 'positive',
        'CLS_AREA_AM': 'positive', 'CLS_AREA_MD': 'positive', 'CLS_AREA_CV': 'negative', 'CLS_GYRATE_MN': 'positive',
        'CLS_GYRATE_AM': 'positive', 'CLS_GYRATE_MD': 'positive', 'CLS_GYRATE_CV': 'negative', 'CLS_SHAPE_MN': 'negative',
        'CLS_SHAPE_AM': 'negative', 'CLS_FRAC_MN': 'negative', 'CLS_FRAC_AM': 'negative', 'CLS_PARA_MN': 'negative',
        'CLS_TCA': 'positive', 'CLS_CPLAND': 'positive', 'CLS_NDCA': 'negative', 'CLS_DCAD': 'negative',
        'CLS_CORE_MN': 'positive', 'CLS_CAI_MN': 'positive', 'CLS_CLUMPY': 'positive', 'CLS_PLADJ': 'positive',
        'CLS_IJI': 'positive', 'CLS_COHESION': 'positive', 'CLS_AI': 'positive',

        # Land Level
        'LND_TA': 'positive', 'LND_NP': 'negative', 'LND_PD': 'negative', 'LND_LPI': 'positive',
        'LND_TE': 'negative', 'LND_ED': 'negative', 'LND_TCA': 'positive', 'LND_CONTAG': 'positive',
        'LND_COHESION': 'positive', 'LND_DIVISION': 'negative', 'LND_MESH': 'positive', 'LND_SPLIT': 'negative',
        'LND_PR': 'positive', 'LND_PRD': 'positive', 'LND_SHDI': 'positive', 'LND_SIDI': 'positive',
        'LND_MSIDI': 'positive', 'LND_SHEI': 'positive', 'LND_AI': 'positive',

        # Patch Level (Aggregated Mean)
        'PATCH_AREA_MN': 'positive', 'PATCH_PERIM_MN': 'negative', 'PATCH_GYRATE_MN': 'positive',
        'PATCH_SHAPE_MN': 'negative', 'PATCH_FRAC_MN': 'negative', 'PATCH_CORE_MN': 'positive',
        'PATCH_NCORE_MN': 'negative', 'PATCH_CAI_MN': 'positive'
    }

def normalize_matrix(matrix, directions):
    """Normalizes the entire data matrix."""
    normalized_matrix = {}
    for indicator, values in matrix.items():
        direction = directions.get(indicator, 'positive')
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
            else:
                normalized_values.append((max_val - v) / (max_val - min_val))
        normalized_matrix[indicator] = normalized_values
    return normalized_matrix

def calculate_entropy_and_weights(normalized_matrix):
    """Calculates entropy and weights for each indicator."""
    entropies = {}
    num_datapoints = len(next(iter(normalized_matrix.values())))
    k = 1.0 / math.log(num_datapoints) if num_datapoints > 1 else 0

    for indicator, norm_values in normalized_matrix.items():
        valid_values = [v for v in norm_values if v is not None and v > 0]
        if not valid_values:
            entropies[indicator] = 1  # Max entropy for no variation or no data
            continue

        sum_norm_values = sum(valid_values)
        proportions = [v / sum_norm_values for v in valid_values]
        
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

def calculate_layer_weights(indicator_weights):
    """Calculates weights for each category based on the sum of their indicator weights."""
    layer_weights = {'infra': 0, 'toyang': 0, 'nongeup': 0, 'pibok': 0}
    
    for indicator, weight in indicator_weights.items():
        if 'infra' in indicator: layer_weights['infra'] += weight
        elif 'toyang' in indicator: layer_weights['toyang'] += weight
        elif 'nongeup' in indicator: layer_weights['nongeup'] += weight
        elif 'pibok' in indicator: layer_weights['pibok'] += weight
        # A simple heuristic: assign metrics by name if they don't have a clear category
        elif 'CLS' in indicator:
            # This case shouldn't happen with the current aggregation, but as a fallback
            pass

    total_weight = sum(layer_weights.values())
    if total_weight == 0:
        return {k: 0.25 for k in layer_weights}
        
    standardized_weights = {k: v / total_weight for k, v in layer_weights.items()}
    return standardized_weights

def main():
    """Main execution function."""
    print("Starting advanced entropy model analysis...")
    
    # Step 1: Aggregate data
    print("Step 1: Aggregating data...")
    all_data = aggregate_all_data()
    data_points_order = list(all_data.keys())
    
    # Step 2: Define indicators
    print("Step 2: Defining indicator directions...")
    indicator_directions = define_all_indicators()
    all_indicator_names = sorted(list(indicator_directions.keys()))

    # Step 3: Build data matrix
    print("Step 3: Building data matrix...")
    matrix = {indicator: [] for indicator in all_indicator_names}
    for dp_name in data_points_order:
        for indicator in all_indicator_names:
            matrix[indicator].append(all_data[dp_name]['metrics'].get(indicator))

    # Step 4: Normalize and Calculate Weights
    print("Step 4: Calculating entropy and indicator weights...")
    normalized_matrix = normalize_matrix(matrix, indicator_directions)
    entropies, indicator_weights = calculate_entropy_and_weights(normalized_matrix)

    # Step 5: Calculate Layer Weights
    print("Step 5: Calculating layer weights...")
    # This is a conceptual step; we calculate scores for each layer first
    # then we can decide how to weigh them.
    # For now, let's calculate a simple score for each of the 8 data points.
    
    # Step 6: Calculate Scores
    print("Step 6: Calculating comprehensive scores...")
    data_point_scores = {}
    for i, dp_name in enumerate(data_points_order):
        score = 0
        for indicator in all_indicator_names:
            norm_value = normalized_matrix[indicator][i]
            if norm_value is not None:
                score += norm_value * indicator_weights[indicator]
        data_point_scores[dp_name] = score * 100  # Scale to 100

    # Step 7: Generate Reports
    print("Step 7: Generating methodology and result files...")
    generate_methodology_report(matrix, normalized_matrix, entropies, indicator_weights, data_point_scores, data_points_order)
    generate_summary_report(indicator_weights, data_point_scores)

    print("\nAnalysis complete.")
    print(f"Methodology document saved to: {OUTPUT_METHODOLOGY_FILE}")
    print(f"Result summary saved to: {OUTPUT_RESULT_FILE}")

def generate_methodology_report(matrix, normalized_matrix, entropies, indicator_weights, scores, order):
    """Generates the detailed methodology markdown file."""
    md = []
    md.append("# 종합 평가 모델 방법론")
    md.append(f"**분석 일시:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ... (Detailed sections for each step)
    md.append("## 1. 분석 개요")
    md.append("본 문서는 나주 및 화순 지역의 농업진흥지역 적합성을 평가하기 위한 종합 모델의 방법론을 상세히 기술합니다.")
    md.append("FRAGSTATS의 Class, Land, Patch 레벨 지표를 모두 통합하고, 엔트로피 가중치법을 적용하여 객관적인 가중치를 산정하고, 이를 기반으로 각 지역-레이어 조합의 종합 점수를 산출합니다.\n")

    md.append("## 2. 데이터 수집 및 통합")
    md.append("총 8개의 평가 대상에 대해 3개 레벨의 데이터를 통합하여 분석 매트릭스를 구축합니다.")
    md.append("- **평가 대상 (8개):** " + ", ".join(order))
    md.append("- **지표 레벨:** Class, Land, Patch (Mean Aggregation)\n")

    md.append("### 2.1. 원본 데이터 매트릭스 (Raw Data Matrix)")
    md.append("각 평가 대상에 대한 원본 지표 값입니다.")
    # Create markdown table for raw data
    header = "| 평가 대상 | " + " | ".join(matrix.keys()) + " |"
    separator = "|---" * (len(matrix.keys()) + 1) + "|"
    md.append(header)
    md.append(separator)
    for i, dp_name in enumerate(order):
        row = f"| **{dp_name}** |"
        for indicator in matrix.keys():
            val = matrix[indicator][i]
            row += f" {val:.4f} |" if val is not None else " N/A |"
        md.append(row)
    md.append("\n")

    md.append("## 3. 엔트로피 가중치법 적용")
    md.append("### 3.1. 데이터 정규화 (Normalization)")
    md.append("각 지표의 방향성(긍정/부정)을 고려하여 모든 값을 0과 1 사이로 정규화합니다.\n")
    # ... (Add normalization formulas)

    md.append("### 3.2. 지표별 엔트로피 및 가중치 산정")
    md.append("정규화된 데이터를 바탕으로 각 지표의 엔트로피(E), 분산도(D=1-E), 최종 가중치(W)를 계산합니다.\n")
    md.append("| 지표명 | 방향성 | 엔트로피 (E) | 분산도 (D) | 최종 가중치 (W) |")
    md.append("|---|---|---|---|---|")
    sorted_weights = sorted(indicator_weights.items(), key=lambda item: item[1], reverse=True)
    for indicator, weight in sorted_weights:
        direction = indicator_directions.get(indicator, 'positive')
        md.append(f"| {indicator} | {direction} | {entropies[indicator]:.4f} | {1-entropies[indicator]:.4f} | {weight:.4f} |")
    md.append("\n")

    md.append("## 4. 종합 점수 산출")
    md.append("산출된 지표별 가중치와 정규화된 값을 선형 결합하여 각 평가 대상의 종합 점수를 100점 만점으로 계산합니다.")
    md.append("`Score_i = Sum(w_j * r_ij) * 100`\n")
    md.append("| 평가 대상 | 종합 점수 |")
    md.append("|---|---|")
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for dp_name, score in sorted_scores:
        md.append(f"| {dp_name} | {score:.2f} |")
    md.append("\n")

    with open(OUTPUT_METHODOLOGY_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

def generate_summary_report(indicator_weights, scores):
    """Generates the summary text file with key results."""
    report = []
    report.append("=" * 80)
    report.append("종합 평가 모델 주요 결과 요약")
    report.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    report.append("\n▶ 상위 10개 주요 지표 (가중치 순)")
    report.append("-" * 40)
    sorted_weights = sorted(indicator_weights.items(), key=lambda item: item[1], reverse=True)
    for i, (indicator, weight) in enumerate(sorted_weights[:10], 1):
        report.append(f"{i:2d}. {indicator:<20} | 가중치: {weight:.4f}")
    
    report.append("\n▶ 평가 대상별 종합 점수 (순위 순)")
    report.append("-" * 40)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for i, (dp_name, score) in enumerate(sorted_scores, 1):
        report.append(f"{i:2d}. {dp_name:<20} | 점수: {score:.2f}")

    with open(OUTPUT_RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    # This is a placeholder for the full indicator definition
    indicator_directions = define_all_indicators()
    main()
