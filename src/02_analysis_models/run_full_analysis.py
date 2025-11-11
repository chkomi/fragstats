
import numpy as np
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# --- 1. 설정 (Configuration) ---
# 프로젝트 루트 디렉토리 설정 (src/02_analysis_models/ 에서 3단계 상위)
WORK_DIR = Path(__file__).parent.parent.parent.resolve()
# 분석 결과를 저장할 새 폴더 이름 정의
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = WORK_DIR / "results" / "03_regional_weights_analysis" / f"analysis_{timestamp}"

# 분석할 카테고리 및 지역 목록
CATEGORIES = ['infra', 'nongeup', 'pibok', 'toyang']
REGIONS = ['naju', 'hwasun']

def get_indicator_definitions():
    """
    58개 지표의 방향성을 정의합니다.
    """
    return {
        'CLS_AREA_AM': {'direction': 'positive'}, 'CLS_LPI': {'direction': 'positive'},
        'LND_PRD': {'direction': 'positive'}, 'LND_TA': {'direction': 'positive'},
        'PATCH_CORE_MN': {'direction': 'positive'}, 'CLS_CORE_MN': {'direction': 'positive'},
        'CLS_GYRATE_AM': {'direction': 'positive'}, 'CLS_AREA_MN': {'direction': 'positive'},
        'PATCH_AREA_MN': {'direction': 'positive'}, 'CLS_AREA_MD': {'direction': 'positive'},
        'PATCH_GYRATE_MN': {'direction': 'positive'}, 'CLS_GYRATE_MN': {'direction': 'positive'},
        'CLS_GYRATE_MD': {'direction': 'positive'}, 'LND_MESH': {'direction': 'positive'},
        'CLS_FRAC_AM': {'direction': 'negative'}, 'CLS_SHAPE_AM': {'direction': 'negative'},
        'CLS_PARA_MN': {'direction': 'negative'}, 'PATCH_CAI_MN': {'direction': 'positive'},
        'CLS_CAI_MN': {'direction': 'positive'}, 'LND_DIVISION': {'direction': 'negative'},
        'CLS_CPLAND': {'direction': 'positive'}, 'CLS_TCA': {'direction': 'positive'},
        'CLS_PD': {'direction': 'negative'}, 'LND_TCA': {'direction': 'positive'},
        'CLS_CLUMPY': {'direction': 'positive'}, 'LND_CONTAG': {'direction': 'positive'},
        'LND_LPI': {'direction': 'positive'}, 'CLS_PLAND': {'direction': 'positive'},
        'CLS_GYRATE_CV': {'direction': 'negative'}, 'LND_COHESION': {'direction': 'positive'},
        'LND_MSIDI': {'direction': 'positive'}, 'CLS_NDCA': {'direction': 'negative'},
        'CLS_CA': {'direction': 'positive'}, 'CLS_NP': {'direction': 'negative'},
        'PATCH_FRAC_MN': {'direction': 'negative'}, 'CLS_FRAC_MN': {'direction': 'negative'},
        'LND_SIDI': {'direction': 'positive'}, 'LND_SPLIT': {'direction': 'negative'},
        'PATCH_PERIM_MN': {'direction': 'negative'}, 'LND_NP': {'direction': 'negative'},
        'LND_SHEI': {'direction': 'positive'}, 'LND_SHDI': {'direction': 'positive'},
        'CLS_AREA_CV': {'direction': 'negative'}, 'CLS_PLADJ': {'direction': 'positive'},
        'CLS_AI': {'direction': 'positive'}, 'CLS_DCAD': {'direction': 'negative'},
        'CLS_COHESION': {'direction': 'positive'}, 'LND_TE': {'direction': 'negative'},
        'CLS_SHAPE_MN': {'direction': 'negative'}, 'PATCH_SHAPE_MN': {'direction': 'negative'},
        'CLS_TE': {'direction': 'negative'}, 'LND_PD': {'direction': 'negative'},
        'LND_ED': {'direction': 'negative'}, 'LND_AI': {'direction': 'positive'},
        'CLS_ED': {'direction': 'negative'}, 'PATCH_NCORE_MN': {'direction': 'negative'},
        'CLS_IJI': {'direction': 'positive'}, 'LND_PR': {'direction': 'positive'}
    }

def read_fragstats_file(file_path):
    """
    FRAGSTATS 결과 파일을 읽어 Pandas DataFrame으로 반환합니다.
    """
    if not file_path.exists():
        print(f"  - File not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"  - Error reading {file_path}: {e}")
        return None

# --- 2. 데이터 집계 (Data Aggregation) ---
def aggregate_all_data():
    """
    모든 FRAGSTATS 파일을 읽고 58개 지표에 대한 데이터 행렬을 구축합니다.
    """
    print("Step 1: Aggregating data from all FRAGSTATS files...")
    data_rows = []
    indicator_defs = get_indicator_definitions()
    
    data_dir = WORK_DIR / "data" / "01_raw_fragstats"
    category_file_map = {cat: cat for cat in CATEGORIES}
    category_file_map['toyang'] = 'toyang3'

    for category in CATEGORIES:
        for region in REGIONS:
            target_id = f"{region}_{category}"
            print(f"  - Processing {target_id}...")
            metrics = {}
            file_prefix = category_file_map[category]

            # Class Metrics
            class_df = read_fragstats_file(data_dir / f"{file_prefix}_class.txt")
            if class_df is not None:
                class_row = class_df[
                    (class_df['LOC'].str.contains(region, case=False)) &
                    (
                        (class_df['TYPE'].str.contains('benefited', case=False) & ~class_df['TYPE'].str.contains('not', case=False)) |
                        (class_df['TYPE'].str.contains('cls_1', case=False))
                    )
                ]
                if not class_row.empty:
                    for indicator in indicator_defs:
                        if indicator.startswith('CLS_'):
                            metric_name = indicator.replace('CLS_', '')
                            if metric_name in class_row.columns:
                                metrics[indicator] = pd.to_numeric(class_row.iloc[0][metric_name], errors='coerce')

            # Land Metrics
            land_df = read_fragstats_file(data_dir / f"{file_prefix}_land.txt")
            if land_df is not None:
                land_row = land_df[land_df['LOC'].str.contains(region, case=False)]
                if not land_row.empty:
                    for indicator in indicator_defs:
                        if indicator.startswith('LND_'):
                            metric_name = indicator.replace('LND_', '')
                            if metric_name in land_row.columns:
                                metrics[indicator] = pd.to_numeric(land_row.iloc[0][metric_name], errors='coerce')

            # Patch Metrics
            patch_file_path = data_dir / f"{file_prefix}_patch_{region}.txt"
            if not patch_file_path.exists():
                patch_file_path = data_dir / f"{file_prefix}_patch.txt"
            patch_df = read_fragstats_file(patch_file_path)
            if patch_df is not None:
                patch_rows = patch_df[
                    (
                        (patch_df['TYPE'].str.contains('benefited', case=False) & ~patch_df['TYPE'].str.contains('not', case=False)) |
                        (patch_df['TYPE'].str.contains('cls_1', case=False))
                    )
                ]
                if not patch_rows.empty:
                    for indicator in indicator_defs:
                        if indicator.startswith('PATCH_'):
                            metric_name = indicator.replace('PATCH_', '').replace('_MN', '')
                            if metric_name in patch_rows.columns:
                                metrics[indicator] = pd.to_numeric(patch_rows[metric_name], errors='coerce').mean()
            
            if metrics:
                metrics['target_id'] = target_id
                data_rows.append(metrics)

    if not data_rows:
        print("Warning: No data was aggregated. Check file paths and names.")
        return pd.DataFrame()

    data_matrix = pd.DataFrame(data_rows).set_index('target_id')
    final_columns = list(indicator_defs.keys())
    for col in final_columns:
        if col not in data_matrix.columns:
            data_matrix[col] = np.nan
    data_matrix = data_matrix[final_columns]
    
    print("Step 1 finished: Data aggregation complete.")
    return data_matrix

# --- 3. 엔트로피 가중치 계산 (Entropy Weight Calculation) ---
def calculate_weights(df):
    """
    데이터프레임을 입력받아 엔트로피 가중치를 계산합니다.
    """
    print("Step 2: Calculating entropy weights for 58 indicators...")
    indicator_defs = get_indicator_definitions()
    n_datapoints = len(df)
    
    if n_datapoints == 0: return {}, pd.DataFrame()

    normalized_df = pd.DataFrame(index=df.index)
    for indicator, col_data in df.items():
        direction = indicator_defs.get(indicator, {}).get('direction', 'positive')
        valid_values = col_data.dropna()
        if valid_values.empty:
            normalized_df[indicator] = np.nan
            continue
        min_val, max_val = valid_values.min(), valid_values.max()
        if max_val == min_val:
            normalized_df[indicator] = 0.5
        else:
            if direction == 'positive':
                normalized_df[indicator] = (col_data - min_val) / (max_val - min_val)
            else:
                normalized_df[indicator] = (max_val - col_data) / (max_val - min_val)
    
    entropies = {}
    k = 1.0 / math.log(n_datapoints) if n_datapoints > 1 else 0
    for indicator, norm_values in normalized_df.items():
        valid_values = norm_values.dropna()
        if valid_values.empty or k == 0:
            entropies[indicator] = 1
            continue
        epsilon = 1e-10
        proportions = (valid_values + epsilon) / (valid_values.sum() + len(valid_values) * epsilon)
        entropy = -k * (np.log(proportions) * proportions).sum()
        entropies[indicator] = entropy

    diversities = {indicator: 1 - e for indicator, e in entropies.items()}
    total_diversity = sum(diversities.values())
    weights = {indicator: d / total_diversity for indicator, d in diversities.items()} if total_diversity > 0 else {ind: 1.0 / len(diversities) for ind in diversities}

    indicator_weights = {ind: {'weight': w, 'entropy': entropies[ind], 'diversity': diversities[ind]} for ind, w in weights.items()}
    
    print("Step 2 finished: Entropy weights calculated.")
    return indicator_weights, normalized_df

# --- 4. 종합 점수 및 레이어 가중치 산정 (Scoring and Layer Weights) ---
def calculate_scores_and_layer_weights(normalized_df, indicator_weights):
    """
    종합 점수와 지역별 최종 레이어 가중치를 계산합니다.
    """
    print("Step 3: Calculating comprehensive scores and region-specific layer weights...")
    if normalized_df.empty or not indicator_weights:
        return pd.Series(dtype='float64'), {}, {}

    # 1. 종합 점수 계산
    weights_series = pd.Series({ind: w_info['weight'] for ind, w_info in indicator_weights.items()})
    scores = (normalized_df * weights_series).sum(axis=1) * 100
    
    # 2. 지역별 최종 레이어 가중치 계산
    naju_scores = scores[scores.index.str.startswith('naju')]
    hwasun_scores = scores[scores.index.str.startswith('hwasun')]

    def get_regional_weights(regional_scores):
        if regional_scores.empty:
            return {}
        total_score = regional_scores.sum()
        if total_score == 0:
            return {idx.split('_')[1]: 1.0 / len(regional_scores) for idx in regional_scores.index}
        
        regional_weights = (regional_scores / total_score).to_dict()
        # Clean up keys from 'naju_infra' to 'infra'
        return {k.split('_')[1]: v for k, v in regional_weights.items()}

    naju_layer_weights = get_regional_weights(naju_scores)
    hwasun_layer_weights = get_regional_weights(hwasun_scores)

    print("Step 3 finished: Scores and layer weights calculated.")
    return scores, naju_layer_weights, hwasun_layer_weights

# --- 5. 결과 생성 및 저장 (Report Generation) ---
def generate_report(indicator_weights, scores, naju_layer_weights, hwasun_layer_weights):
    """
    분석 결과를 종합하여 보고서 파일을 생성합니다.
    """
    print(f"Step 4: Generating report in {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    report_md_path = OUTPUT_DIR / "종합_분석_보고서_지역별_가중치.md"
    naju_layer_weights_path = OUTPUT_DIR / "레이어별_가중치_나주.csv"
    hwasun_layer_weights_path = OUTPUT_DIR / "레이어별_가중치_화순.csv"
    scores_path = OUTPUT_DIR / "평가대상별_종합점수.csv"
    indicator_weights_path = OUTPUT_DIR / "지표별_가중치.csv"

    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 종합 분석 보고서 (지역별 가중치 산정)\n")
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("이 보고서는 나주와 화순 지역의 레이어별 가중치를 별도로 산정한 결과입니다.\n\n")

        # 1. 나주 최종 레이어 가중치
        f.write("## 1. 나주(Naju) 지역 최종 레이어 가중치\n\n| 순위 | 레이어 | 최종 가중치 | 백분율 (%) |\n|:--:|:---|:---:|:---:|")
        sorted_layers_naju = sorted(naju_layer_weights.items(), key=lambda item: item[1], reverse=True)
        for i, (layer, weight) in enumerate(sorted_layers_naju, 1):
            f.write(f"| {i} | {layer.capitalize()} | {weight:.4f} | {weight*100:.2f} |\n")
        f.write("\n")
        
        # 2. 화순 최종 레이어 가중치
        f.write("## 2. 화순(Hwasun) 지역 최종 레이어 가중치\n\n| 순위 | 레이어 | 최종 가중치 | 백분율 (%) |\n|:--:|:---|:---:|:---:|")
        sorted_layers_hwasun = sorted(hwasun_layer_weights.items(), key=lambda item: item[1], reverse=True)
        for i, (layer, weight) in enumerate(sorted_layers_hwasun, 1):
            f.write(f"| {i} | {layer.capitalize()} | {weight:.4f} | {weight*100:.2f} |\n")
        f.write("\n")

        f.write("---\n\n")

        f.write("## 3. 평가 대상별 종합 점수 (공통)\n\n| 순위 | 평가 대상 ID | 종합 점수 (100점 만점) |\n|:--:|:---|:---:|")
        sorted_scores = scores.sort_values(ascending=False)
        for i, (target_id, score) in enumerate(sorted_scores.items(), 1):
            f.write(f"| {i} | `{target_id}` | {score:.2f} |\n")
        f.write("\n")

        f.write("## 4. 지표별 가중치 (58개, 공통)\n\n| 순위 | 지표명 | 최종 가중치 (W) | 엔트로피 (E) | 분산도 (D) |\n|:--:|:---|:---:|:---:|:---:|")
        sorted_indicators = sorted(indicator_weights.items(), key=lambda item: item[1]['weight'], reverse=True)
        for i, (name, w_info) in enumerate(sorted_indicators, 1):
            f.write(f"| {i} | `{name}` | {w_info['weight']:.4f} | {w_info['entropy']:.4f} | {w_info['diversity']:.4f} |\n")
        f.write("\n")

    # CSV 파일 저장
    pd.DataFrame.from_dict(naju_layer_weights, orient='index', columns=['weight']).sort_values('weight', ascending=False).to_csv(naju_layer_weights_path, encoding='utf-8-sig')
    pd.DataFrame.from_dict(hwasun_layer_weights, orient='index', columns=['weight']).sort_values('weight', ascending=False).to_csv(hwasun_layer_weights_path, encoding='utf-8-sig')
    scores.sort_values(ascending=False).to_csv(scores_path, encoding='utf-8-sig')
    pd.DataFrame.from_dict(indicator_weights, orient='index').sort_values('weight', ascending=False).to_csv(indicator_weights_path, encoding='utf-8-sig')

    print(f"Report and CSV files successfully generated in: {OUTPUT_DIR}")

# --- 6. 메인 실행 함수 (Main Execution) ---
def main():
    """
    전체 분석 파이프라인을 실행합니다.
    """
    print("Starting full analysis pipeline...")
    data_matrix = aggregate_all_data()
    if data_matrix.empty:
        print("Execution stopped because no data could be aggregated.")
        return
    indicator_weights, normalized_matrix = calculate_weights(data_matrix)
    scores, naju_layer_weights, hwasun_layer_weights = calculate_scores_and_layer_weights(normalized_matrix, indicator_weights)
    generate_report(indicator_weights, scores, naju_layer_weights, hwasun_layer_weights)
    print(f"\nAnalysis pipeline finished successfully. Results are in:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
