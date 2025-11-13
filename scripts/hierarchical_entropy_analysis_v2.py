#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
계층적 엔트로피 분석 v2 (Hierarchical Entropy Weighting Analysis)
농지 잠재력 평가를 위한 레이어별 가중치 산정

연구 프레임워크:
1단계: 그룹 내 지표 가중치 산정 (엔트로피 분석)
2단계: 그룹 대표 점수 산출 (cls_1과 cls_9 모두)
3단계: 변별력 계산 (|점수_cls1 - 점수_cls9|)
4단계: 지역별 레이어 가중치 산정
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "01_raw_fragstats"
RESULTS_DIR = BASE_DIR / "results" / "06_hierarchical_entropy_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 분석 시작 시간
analysis_time = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n{'='*80}")
print(f"계층적 엔트로피 분석 v2 시작: {analysis_time}")
print(f"{'='*80}\n")


# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

print("1단계: 데이터 로드 및 전처리")
print("-" * 80)

# 4개 레이어의 class.txt 파일 읽기
layers = ['infra', 'pibok', 'nongeup', 'toyang3']
all_data = []

for layer in layers:
    file_path = DATA_DIR / f"{layer}_class.txt"
    print(f"  - {layer}_class.txt 읽는 중...")

    df = pd.read_csv(file_path, sep='\t', skipinitialspace=True)

    # 컬럼명 정리 (앞뒤 공백 제거)
    df.columns = df.columns.str.strip()

    # LOC 컬럼 파싱: "지역_레이어" 형식
    df['region'] = df['LOC'].str.split('_').str[0]
    df['layer'] = layer

    # TYPE 컬럼 정리 및 class 구분
    # infra: giban_benefited=cls_1, giban_not_benefited=cls_9
    # 나머지: cls_1, cls_9
    def get_class_type(x):
        x_str = str(x).strip()
        if x_str == 'giban_benefited' or x_str == 'cls_1':
            return 'cls_1'
        else:
            return 'cls_9'

    df['class_type'] = df['TYPE'].apply(get_class_type)

    all_data.append(df)

# 전체 데이터 병합
full_df = pd.concat(all_data, ignore_index=True)

print(f"\n총 데이터 행 수: {len(full_df)}")
print(f"지역: {sorted(full_df['region'].unique())}")
print(f"레이어: {sorted(full_df['layer'].unique())}")
print(f"클래스: {sorted(full_df['class_type'].unique())}")

# ============================================================================
# 2. 지표 그룹 정의
# ============================================================================

print(f"\n2단계: 지표 그룹 정의")
print("-" * 80)

# FRAGSTATS 이론에 따른 4개 그룹
indicator_groups = {
    'area_density': ['CA', 'PLAND', 'NP', 'PD', 'LPI', 'AREA_MN', 'AREA_AM', 'AREA_MD', 'AREA_CV'],
    'shape': ['TE', 'ED', 'SHAPE_MN', 'SHAPE_AM', 'FRAC_MN', 'FRAC_AM', 'PARA_MN'],
    'core': ['TCA', 'CPLAND', 'NDCA', 'DCAD', 'CORE_MN', 'CAI_MN'],
    'aggregation': ['GYRATE_MN', 'GYRATE_AM', 'GYRATE_MD', 'GYRATE_CV', 'CLUMPY', 'PLADJ', 'COHESION', 'AI']
}

# 모든 분석 대상 지표
all_indicators = []
for group_indicators in indicator_groups.values():
    all_indicators.extend(group_indicators)

print(f"\n지표 그룹 구성:")
for group_name, indicators in indicator_groups.items():
    print(f"  - {group_name}: {len(indicators)}개 지표")

# ============================================================================
# 3. 지표 방향성 정의 (농업적 가치 관점)
# ============================================================================

print(f"\n3단계: 지표 방향성 정의")
print("-" * 80)

# 높을수록 좋은 지표 (positive): 1
# 낮을수록 좋은 지표 (negative): -1
indicator_direction = {
    # 면적/밀도 그룹
    'CA': 1, 'PLAND': 1, 'NP': -1, 'PD': -1, 'LPI': 1,
    'AREA_MN': 1, 'AREA_AM': 1, 'AREA_MD': 1, 'AREA_CV': -1,
    # 형태 그룹
    'TE': -1, 'ED': -1, 'SHAPE_MN': -1, 'SHAPE_AM': -1,
    'FRAC_MN': -1, 'FRAC_AM': -1, 'PARA_MN': -1,
    # 코어 그룹
    'TCA': 1, 'CPLAND': 1, 'NDCA': -1, 'DCAD': -1, 'CORE_MN': 1, 'CAI_MN': 1,
    # 응집 그룹
    'GYRATE_MN': 1, 'GYRATE_AM': 1, 'GYRATE_MD': 1, 'GYRATE_CV': -1,
    'CLUMPY': 1, 'PLADJ': 1, 'COHESION': 1, 'AI': 1
}

positive = [k for k, v in indicator_direction.items() if v == 1]
negative = [k for k, v in indicator_direction.items() if v == -1]
print(f"  - Positive (높을수록 좋음): {len(positive)}개")
print(f"  - Negative (낮을수록 좋음): {len(negative)}개")


# ============================================================================
# 4. 엔트로피 가중치 계산 함수
# ============================================================================

def calculate_entropy_weights(data_matrix, indicator_names, directions):
    """엔트로피 가중치 계산"""
    n_samples, n_indicators = data_matrix.shape
    epsilon = 1e-10  # 0 방지

    # 1단계: 정규화 (Min-Max with Direction)
    normalized = np.zeros_like(data_matrix, dtype=float)

    for j, indicator in enumerate(indicator_names):
        col = data_matrix[:, j]
        min_val = np.min(col)
        max_val = np.max(col)

        if max_val - min_val < epsilon:
            normalized[:, j] = 0.5
        else:
            norm_col = (col - min_val) / (max_val - min_val)
            if directions[indicator] == -1:
                norm_col = 1 - norm_col
            normalized[:, j] = norm_col

    # 2단계: 비율 행렬 (P_ij)
    col_sums = np.sum(normalized, axis=0)
    col_sums = np.where(col_sums < epsilon, 1, col_sums)
    P = normalized / col_sums

    # 3단계: 엔트로피 (E_j)
    k = 1 / np.log(n_samples) if n_samples > 1 else 1
    P_log = np.where(P > epsilon, P * np.log(P), 0)
    E = -k * np.sum(P_log, axis=0)

    # 4단계: 정보 효용도 (D_j)
    D = 1 - E

    # 5단계: 가중치 (W_j)
    D_sum = np.sum(D)
    W = D / D_sum if D_sum > epsilon else np.ones(n_indicators) / n_indicators

    weights = {indicator_names[i]: W[i] for i in range(n_indicators)}

    details = {
        'entropy': {indicator_names[i]: E[i] for i in range(n_indicators)},
        'utility': {indicator_names[i]: D[i] for i in range(n_indicators)},
        'normalized_data': normalized
    }

    return weights, normalized, details


# ============================================================================
# 5. 1단계: 그룹 내 지표 가중치 산정
# ============================================================================

print(f"\n{'='*80}")
print("5단계: 1단계 분석 - 그룹 내 지표 가중치 산정")
print(f"{'='*80}\n")

# cls_1과 cls_9 모두 포함하여 분석 (변별력 계산을 위해)
analysis_units = []
for region in ['hwasun', 'naju']:
    for layer in layers:
        for class_type in ['cls_1', 'cls_9']:
            unit_data = full_df[
                (full_df['region'] == region) &
                (full_df['layer'] == layer) &
                (full_df['class_type'] == class_type)
            ]
            if len(unit_data) > 0:
                analysis_units.append({
                    'region': region,
                    'layer': layer,
                    'class_type': class_type,
                    'data': unit_data
                })

print(f"분석 단위 수: {len(analysis_units)}개")
print(f"  (지역 2개 x 레이어 4개 x 클래스 2개 = 16개 예상)\n")

# 그룹별 분석
group_weights_results = {}
group_scores_all = {}  # 모든 유닛의 그룹 점수 저장

for group_name, group_indicators in indicator_groups.items():
    print(f"\n[{group_name.upper()} 그룹] - {len(group_indicators)}개 지표")
    print("-" * 80)

    # 데이터 행렬 구성
    data_matrix = []
    unit_labels = []

    for unit in analysis_units:
        region = unit['region']
        layer = unit['layer']
        class_type = unit['class_type']
        unit_df = unit['data']

        row = []
        for indicator in group_indicators:
            if indicator in unit_df.columns:
                value = unit_df[indicator].values[0]
                row.append(value)
            else:
                row.append(0)

        data_matrix.append(row)
        unit_labels.append(f"{region}_{layer}_{class_type}")

    data_matrix = np.array(data_matrix, dtype=float)

    # 엔트로피 가중치 계산
    weights, normalized, details = calculate_entropy_weights(
        data_matrix, group_indicators, indicator_direction
    )

    group_weights_results[group_name] = {
        'weights': weights,
        'details': details,
        'indicators': group_indicators
    }

    # 그룹 대표 점수 계산
    group_scores = {}
    for i, unit_label in enumerate(unit_labels):
        score = np.sum(normalized[i, :] * np.array(list(weights.values())))
        group_scores[unit_label] = score

    group_scores_all[group_name] = group_scores

    # 결과 출력 (상위 5개만)
    print(f"\n[가중치 Top 5]")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for indicator, weight in sorted_weights:
        print(f"  {indicator:12s}: {weight:.4f}")


# ============================================================================
# 6. 2단계: 그룹 대표 점수 DataFrame 구성
# ============================================================================

print(f"\n{'='*80}")
print("6단계: 2단계 분석 - 그룹 대표 점수 정리")
print(f"{'='*80}\n")

# 그룹 점수를 DataFrame으로 변환
group_score_df = pd.DataFrame(group_scores_all)
group_score_df.index.name = 'unit'

# 분리: region, layer, class_type
# 인덱스 형식: "region_layer_cls_1" 또는 "region_layer_cls_9"
def parse_unit_index(idx):
    parts = idx.split('_')
    region = parts[0]
    layer = parts[1]
    class_type = '_'.join(parts[2:])  # "cls_1" 또는 "cls_9"
    return region, layer, class_type

parsed = [parse_unit_index(idx) for idx in group_score_df.index]
group_score_df['region'] = [p[0] for p in parsed]
group_score_df['layer'] = [p[1] for p in parsed]
group_score_df['class_type'] = [p[2] for p in parsed]

print("그룹 대표 점수 샘플 (처음 10개):")
print(group_score_df.head(10))


# ============================================================================
# 7. 3단계: 변별력 계산 (|점수_cls1 - 점수_cls9|)
# ============================================================================

print(f"\n{'='*80}")
print("7단계: 변별력 계산 (cls_1과 cls_9의 점수 차이)")
print(f"{'='*80}\n")

discriminability_results = {}

for region in ['hwasun', 'naju']:
    print(f"\n[{region.upper()} 지역]")
    print("-" * 80)

    region_discriminability = {}

    for layer in layers:
        # cls_1과 cls_9 점수 추출
        cls1_label = f"{region}_{layer}_cls_1"
        cls9_label = f"{region}_{layer}_cls_9"

        if cls1_label in group_score_df.index and cls9_label in group_score_df.index:
            scores_cls1 = group_score_df.loc[cls1_label, list(indicator_groups.keys())].values
            scores_cls9 = group_score_df.loc[cls9_label, list(indicator_groups.keys())].values

            # 변별력 = 절대 차이의 합
            diff = np.abs(scores_cls1 - scores_cls9)
            total_discriminability = np.sum(diff)

            region_discriminability[layer] = {
                'cls1_scores': scores_cls1,
                'cls9_scores': scores_cls9,
                'diff': diff,
                'total': total_discriminability
            }

            print(f"{layer:12s}: 변별력 = {total_discriminability:.4f}")

    discriminability_results[region] = region_discriminability


# ============================================================================
# 8. 4단계: 지역별 레이어 가중치 산정
# ============================================================================

print(f"\n{'='*80}")
print("8단계: 지역별 레이어 가중치 산정 (엔트로피 기반)")
print(f"{'='*80}\n")

final_layer_weights = {}

for region in ['hwasun', 'naju']:
    print(f"\n[{region.upper()} 지역]")
    print("-" * 80)

    # 변별력 값 추출
    discriminability_values = []
    layer_list = []

    for layer in layers:
        if layer in discriminability_results[region]:
            disc_value = discriminability_results[region][layer]['total']
            discriminability_values.append(disc_value)
            layer_list.append(layer)

    discriminability_values = np.array(discriminability_values)

    # 변별력 직접 정규화 (합이 1이 되도록)
    # Min-Max 정규화를 사용하지 않고 직접 비율로 변환
    disc_sum = np.sum(discriminability_values)

    if disc_sum < 1e-10:
        # 모든 변별력이 0에 가까운 경우: 균등 가중치
        layer_weights = np.ones(len(discriminability_values)) / len(discriminability_values)
    else:
        # 변별력 비율을 가중치로 사용
        layer_weights = discriminability_values / disc_sum

    weights_dict = {layer_list[i]: layer_weights[i] for i in range(len(layer_list))}
    final_layer_weights[region] = weights_dict

    print(f"\n[레이어 가중치]")
    print(f"변별력 기반 정규화 가중치:")
    sorted_weights = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    for layer, weight in sorted_weights:
        disc_val = discriminability_results[region][layer]['total']
        print(f"  {layer:12s}: {weight:.4f}  (변별력: {disc_val:.4f})")


# ============================================================================
# 9. 결과 저장
# ============================================================================

print(f"\n{'='*80}")
print("9단계: 결과 저장")
print(f"{'='*80}\n")

output_dir = RESULTS_DIR / f"analysis_{analysis_time}"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 그룹 내 지표 가중치
group_weights_df = pd.DataFrame({
    group: pd.Series(result['weights'])
    for group, result in group_weights_results.items()
})
group_weights_file = output_dir / "01_group_indicator_weights.csv"
group_weights_df.to_csv(group_weights_file, encoding='utf-8-sig')
print(f"  [OK] {group_weights_file.name}")

# 2. 그룹 대표 점수
group_scores_file = output_dir / "02_group_representative_scores.csv"
group_score_df.to_csv(group_scores_file, encoding='utf-8-sig')
print(f"  [OK] {group_scores_file.name}")

# 3. 변별력 상세
discriminability_detail = []
for region, layers_data in discriminability_results.items():
    for layer, data in layers_data.items():
        row = {
            'region': region,
            'layer': layer,
            'discriminability_total': data['total']
        }
        for i, group_name in enumerate(indicator_groups.keys()):
            row[f'{group_name}_diff'] = data['diff'][i]
        discriminability_detail.append(row)

disc_df = pd.DataFrame(discriminability_detail)
disc_file = output_dir / "03_discriminability_details.csv"
disc_df.to_csv(disc_file, index=False, encoding='utf-8-sig')
print(f"  [OK] {disc_file.name}")

# 4. 최종 레이어 가중치
layer_weights_df = pd.DataFrame(final_layer_weights).T
layer_weights_df.index.name = 'region'
layer_weights_file = output_dir / "04_final_layer_weights.csv"
layer_weights_df.to_csv(layer_weights_file, encoding='utf-8-sig')
print(f"  [OK] {layer_weights_file.name}")

# 5. 전체 요약 JSON
summary = {
    'analysis_time': analysis_time,
    'regions': ['hwasun', 'naju'],
    'layers': layers,
    'final_layer_weights': final_layer_weights,
    'group_indicator_weights': {k: v['weights'] for k, v in group_weights_results.items()}
}

summary_file = output_dir / "00_analysis_summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  [OK] {summary_file.name}")

# 6. 중간 계산 과정 저장 (엔트로피, 효용도 등)
intermediate_results = {}
for group_name, result in group_weights_results.items():
    intermediate_results[group_name] = {
        'entropy': result['details']['entropy'],
        'utility': result['details']['utility']
    }

intermediate_file = output_dir / "05_intermediate_calculations.json"
with open(intermediate_file, 'w', encoding='utf-8') as f:
    json.dump(intermediate_results, f, ensure_ascii=False, indent=2)
print(f"  [OK] {intermediate_file.name}")

print(f"\n모든 결과가 저장되었습니다:")
print(f"  폴더: {output_dir}")

print(f"\n{'='*80}")
print("분석 완료!")
print(f"{'='*80}\n")

# 최종 결과 요약 출력
print("\n" + "="*80)
print("최종 레이어 가중치 요약")
print("="*80)
for region in ['hwasun', 'naju']:
    print(f"\n[{region.upper()}]")
    sorted_weights = sorted(final_layer_weights[region].items(), key=lambda x: x[1], reverse=True)
    for rank, (layer, weight) in enumerate(sorted_weights, 1):
        print(f"  {rank}. {layer:12s}: {weight:.4f} ({weight*100:.2f}%)")
