#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
엔트로피 가중치법을 이용한 농업진흥지역 평가 모델
Entropy Weight Method for Agricultural Promotion Area Evaluation
"""

import math
from pathlib import Path
from datetime import datetime

work_dir = Path("/Users/yunhyungchang/Documents/FRAGSTATS")

def safe_float(value):
    """안전하게 float로 변환"""
    try:
        if value == 'N/A' or value == '' or value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def read_fragstats_file(file_path):
    """FRAGSTATS 결과 파일 읽기"""
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
    except Exception as e:
        return None

def normalize_data(values, direction='positive'):
    """
    데이터 정규화 (Min-Max Normalization)

    direction:
    - 'positive': 클수록 좋음 (PLAND, LPI, AI, COHESION 등)
    - 'negative': 작을수록 좋음 (NP, PD, ED, DIVISION 등)
    """
    valid_values = [v for v in values if v is not None]
    if not valid_values or len(valid_values) == 0:
        return [None] * len(values)

    min_val = min(valid_values)
    max_val = max(valid_values)

    if max_val == min_val:
        # 모든 값이 같으면 중간값 반환
        return [0.5 if v is not None else None for v in values]

    normalized = []
    for v in values:
        if v is None:
            normalized.append(None)
        elif direction == 'positive':
            normalized.append((v - min_val) / (max_val - min_val))
        else:  # negative
            normalized.append((max_val - v) / (max_val - min_val))

    return normalized

def calculate_entropy(normalized_values):
    """
    엔트로피 계산

    E_j = -k * Σ(p_ij * ln(p_ij))
    where k = 1/ln(n), n = number of samples
    p_ij = normalized value / sum of normalized values
    """
    valid_values = [v for v in normalized_values if v is not None]
    n = len(valid_values)

    if n == 0 or n == 1:
        return 0

    # 0을 포함한 값 처리를 위해 작은 값 추가
    epsilon = 1e-10
    adjusted_values = [v + epsilon for v in valid_values]

    # 비율 계산
    total = sum(adjusted_values)
    proportions = [v / total for v in adjusted_values]

    # 엔트로피 계산
    k = 1.0 / math.log(n)
    entropy = -k * sum(p * math.log(p) for p in proportions if p > 0)

    return entropy

def calculate_entropy_weight(entropy_values):
    """
    엔트로피 가중치 계산

    w_j = (1 - E_j) / Σ(1 - E_j)
    where E_j is entropy of indicator j
    """
    diversities = [1 - e for e in entropy_values]
    total_diversity = sum(diversities)

    if total_diversity == 0:
        # 모든 지표의 분산도가 0이면 균등 가중치
        return [1.0 / len(entropy_values)] * len(entropy_values)

    weights = [d / total_diversity for d in diversities]
    return weights

def extract_class_data():
    """Class 레벨 데이터 추출 (농지 적합 타입만)"""
    categories = ['infra', 'toyang', 'nongeup', 'pibok']
    all_data = []

    for category in categories:
        file_path = work_dir / f"{category}_class.txt"
        file_data = read_fragstats_file(file_path)

        if file_data:
            for row in file_data['data']:
                type_val = row.get('TYPE', '')
                # 농지 적합 타입만
                if ('benefited' in type_val and 'not' not in type_val) or 'cls_1' in type_val:
                    loc = row.get('LOC', '')
                    region = "hwasun" if "hwasun" in loc else "naju"

                    data_point = {
                        'region': region,
                        'category': category,
                        'type': type_val,
                        'data': row
                    }
                    all_data.append(data_point)

    return all_data

def define_indicators():
    """
    농업진흥지역 평가 지표 정의

    각 지표별 방향성 및 의미 설정
    """
    indicators = {
        # 면적 및 비율 지표 (positive: 클수록 좋음)
        'CA': {'direction': 'positive', 'name': '총면적', 'weight_group': 'area'},
        'PLAND': {'direction': 'positive', 'name': '경관비율', 'weight_group': 'area'},
        'LPI': {'direction': 'positive', 'name': '최대패치비율', 'weight_group': 'connectivity'},

        # 파편화 지표 (negative: 작을수록 좋음)
        'NP': {'direction': 'negative', 'name': '패치수', 'weight_group': 'fragmentation'},
        'PD': {'direction': 'negative', 'name': '패치밀도', 'weight_group': 'fragmentation'},
        'ED': {'direction': 'negative', 'name': '가장자리밀도', 'weight_group': 'fragmentation'},

        # 형태 및 크기 지표
        'AREA_MN': {'direction': 'positive', 'name': '평균면적', 'weight_group': 'size'},
        'AREA_MD': {'direction': 'positive', 'name': '중앙면적', 'weight_group': 'size'},
        'GYRATE_MN': {'direction': 'positive', 'name': '평균회전반경', 'weight_group': 'size'},

        # 핵심 농지 지표 (positive)
        'TCA': {'direction': 'positive', 'name': '총핵심면적', 'weight_group': 'core'},
        'CPLAND': {'direction': 'positive', 'name': '핵심면적비율', 'weight_group': 'core'},
        'CORE_MN': {'direction': 'positive', 'name': '평균핵심면적', 'weight_group': 'core'},

        # 집적도 지표 (positive: 클수록 좋음)
        'CLUMPY': {'direction': 'positive', 'name': '집괴도', 'weight_group': 'aggregation'},
        'PLADJ': {'direction': 'positive', 'name': '인접비율', 'weight_group': 'aggregation'},
        'AI': {'direction': 'positive', 'name': '집적지수', 'weight_group': 'aggregation'},
        'COHESION': {'direction': 'positive', 'name': '응집도', 'weight_group': 'aggregation'},

        # 형태 복잡도 지표 (낮을수록 관리 용이)
        'SHAPE_MN': {'direction': 'negative', 'name': '평균형태지수', 'weight_group': 'shape'},
        'FRAC_MN': {'direction': 'negative', 'name': '평균프랙탈차원', 'weight_group': 'shape'},

        # 분산도 지표
        'NDCA': {'direction': 'negative', 'name': '분리핵심영역수', 'weight_group': 'core'},
        'DCAD': {'direction': 'negative', 'name': '분리핵심영역밀도', 'weight_group': 'core'},
    }

    return indicators

def calculate_indicator_weights(data_points):
    """엔트로피 가중치법으로 지표별 가중치 계산"""
    indicators = define_indicators()
    indicator_names = list(indicators.keys())

    # 각 지표별 값 수집
    indicator_values = {name: [] for name in indicator_names}

    for dp in data_points:
        for name in indicator_names:
            val = safe_float(dp['data'].get(name))
            indicator_values[name].append(val)

    # 정규화 및 엔트로피 계산
    entropies = {}
    normalized_data = {}

    for name in indicator_names:
        direction = indicators[name]['direction']
        normalized = normalize_data(indicator_values[name], direction)
        normalized_data[name] = normalized

        entropy = calculate_entropy(normalized)
        entropies[name] = entropy

    # 가중치 계산
    weights = calculate_entropy_weight(list(entropies.values()))

    # 지표별 가중치 매핑
    indicator_weights = {}
    for i, name in enumerate(indicator_names):
        indicator_weights[name] = {
            'weight': weights[i],
            'entropy': entropies[name],
            'diversity': 1 - entropies[name],
            'direction': indicators[name]['direction'],
            'name': indicators[name]['name'],
            'group': indicators[name]['weight_group']
        }

    return indicator_weights, normalized_data

def calculate_group_weights(indicator_weights):
    """
    지표 그룹별 가중치 계산
    (면적, 파편화, 집적도, 핵심면적, 크기, 형태)
    """
    groups = {}

    for indicator, info in indicator_weights.items():
        group = info['group']
        if group not in groups:
            groups[group] = []
        groups[group].append(info['weight'])

    # 그룹별 가중치 합계
    group_weights = {}
    for group, weights in groups.items():
        group_weights[group] = sum(weights)

    return group_weights

def calculate_layer_weights():
    """
    레이어별(카테고리별) 가중치 계산

    각 레이어의 데이터를 기반으로 엔트로피 가중치 산정
    """
    categories = ['infra', 'toyang', 'nongeup', 'pibok']
    layer_data = {}

    # 각 레이어별 대표 지표 값 추출
    for category in categories:
        file_path = work_dir / f"{category}_class.txt"
        file_data = read_fragstats_file(file_path)

        if file_data:
            # 농지 적합 타입의 PLAND 값 사용 (경관 내 비중)
            pland_values = []
            for row in file_data['data']:
                type_val = row.get('TYPE', '')
                if ('benefited' in type_val and 'not' not in type_val) or 'cls_1' in type_val:
                    pland = safe_float(row.get('PLAND'))
                    if pland is not None:
                        pland_values.append(pland)

            layer_data[category] = pland_values

    # 각 레이어별 평균값으로 엔트로피 계산
    layer_means = []
    for category in categories:
        if layer_data[category]:
            layer_means.append(sum(layer_data[category]) / len(layer_data[category]))
        else:
            layer_means.append(0)

    # 정규화
    normalized_means = normalize_data(layer_means, 'positive')

    # 엔트로피 계산
    entropy = calculate_entropy(normalized_means)

    # 각 레이어별 분산도 기반 가중치
    layer_variances = []
    for category in categories:
        if layer_data[category] and len(layer_data[category]) > 1:
            mean = sum(layer_data[category]) / len(layer_data[category])
            variance = sum((x - mean) ** 2 for x in layer_data[category]) / len(layer_data[category])
            layer_variances.append(variance)
        else:
            layer_variances.append(0)

    # 분산도 기반 가중치 (분산이 클수록 정보량이 많음)
    total_variance = sum(layer_variances)
    if total_variance > 0:
        layer_weights = {
            categories[i]: layer_variances[i] / total_variance
            for i in range(len(categories))
        }
    else:
        # 균등 가중치
        layer_weights = {cat: 0.25 for cat in categories}

    return layer_weights

def calculate_comprehensive_score(data_point, indicator_weights, layer_weights, normalized_data):
    """종합 점수 계산"""
    category = data_point['category']
    layer_weight = layer_weights[category]

    # 지표별 정규화된 값과 가중치를 곱해서 합산
    score = 0
    total_weight = 0

    for indicator, info in indicator_weights.items():
        val = safe_float(data_point['data'].get(indicator))
        if val is not None:
            # 정규화된 값 찾기
            indicator_values = [safe_float(dp['data'].get(indicator))
                              for dp in [data_point]]
            direction = info['direction']
            normalized = normalize_data([val], direction)[0]

            if normalized is not None:
                score += normalized * info['weight']
                total_weight += info['weight']

    # 레이어 가중치 적용
    if total_weight > 0:
        indicator_score = score / total_weight
    else:
        indicator_score = 0

    final_score = indicator_score * layer_weight * 100  # 100점 만점으로 스케일링

    return final_score, indicator_score

def generate_evaluation_model():
    """농업진흥지역 평가 모델 생성"""
    results = []

    results.append("=" * 80)
    results.append("엔트로피 가중치법을 이용한 농업진흥지역 평가 모델")
    results.append("Entropy Weight Method for Agricultural Promotion Area Evaluation")
    results.append("=" * 80)
    results.append("")
    results.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.append("")

    # 데이터 추출
    results.append("\n" + "=" * 80)
    results.append("STEP 1. 데이터 수집 및 전처리")
    results.append("=" * 80)
    results.append("")

    data_points = extract_class_data()
    results.append(f"총 데이터 포인트: {len(data_points)}개")
    results.append(f"- 화순: {len([d for d in data_points if d['region'] == 'hwasun'])}개")
    results.append(f"- 나주: {len([d for d in data_points if d['region'] == 'naju'])}개")
    results.append("")

    # 레이어별 가중치 계산
    results.append("\n" + "=" * 80)
    results.append("STEP 2. 레이어별 가중치 산정 (엔트로피 기반)")
    results.append("=" * 80)
    results.append("")

    layer_weights = calculate_layer_weights()
    results.append("레이어별 가중치 (분산도 기반):")
    results.append("-" * 80)

    layer_names = {
        'infra': '기반시설 수혜도',
        'toyang': '토양 등급',
        'nongeup': '농업용지 등급',
        'pibok': '토지피복 등급'
    }

    for category, weight in sorted(layer_weights.items(), key=lambda x: x[1], reverse=True):
        results.append(f"{layer_names[category]:20s}: {weight:6.4f} ({weight*100:5.2f}%)")
    results.append("")

    # 지표별 가중치 계산
    results.append("\n" + "=" * 80)
    results.append("STEP 3. 지표별 가중치 산정 (엔트로피 기반)")
    results.append("=" * 80)
    results.append("")

    indicator_weights, normalized_data = calculate_indicator_weights(data_points)

    results.append(f"{'지표명':20s} {'영문명':12s} {'가중치':>8s} {'엔트로피':>8s} {'분산도':>8s} {'방향':>8s}")
    results.append("-" * 80)

    sorted_indicators = sorted(indicator_weights.items(),
                              key=lambda x: x[1]['weight'], reverse=True)

    for indicator, info in sorted_indicators:
        direction_kr = '높을수록' if info['direction'] == 'positive' else '낮을수록'
        results.append(f"{info['name']:20s} {indicator:12s} {info['weight']:8.4f} "
                      f"{info['entropy']:8.4f} {info['diversity']:8.4f} {direction_kr:>8s}")
    results.append("")

    # 그룹별 가중치
    results.append("\n" + "=" * 80)
    results.append("STEP 4. 지표 그룹별 가중치 분석")
    results.append("=" * 80)
    results.append("")

    group_weights = calculate_group_weights(indicator_weights)

    group_names = {
        'area': '면적 지표',
        'fragmentation': '파편화 지표',
        'aggregation': '집적도 지표',
        'core': '핵심면적 지표',
        'size': '크기 지표',
        'shape': '형태 지표'
    }

    results.append(f"{'그룹명':20s} {'총 가중치':>10s} {'비율(%)':>10s}")
    results.append("-" * 80)

    for group, weight in sorted(group_weights.items(), key=lambda x: x[1], reverse=True):
        results.append(f"{group_names.get(group, group):20s} {weight:10.4f} {weight*100:10.2f}")
    results.append("")

    # 종합 점수 계산
    results.append("\n" + "=" * 80)
    results.append("STEP 5. 지역별·레이어별 종합 점수 산정")
    results.append("=" * 80)
    results.append("")

    # 지역별, 카테고리별 점수
    scores = {}
    for dp in data_points:
        region = dp['region']
        category = dp['category']
        key = f"{region}_{category}"

        final_score, indicator_score = calculate_comprehensive_score(
            dp, indicator_weights, layer_weights, normalized_data
        )

        scores[key] = {
            'region': region,
            'category': category,
            'type': dp['type'],
            'final_score': final_score,
            'indicator_score': indicator_score,
            'layer_weight': layer_weights[category]
        }

    results.append(f"{'지역':10s} {'레이어':15s} {'타입':25s} {'지표점수':>10s} {'레이어가중치':>12s} {'최종점수':>10s}")
    results.append("-" * 80)

    for key, score_info in sorted(scores.items()):
        region_kr = "화순" if score_info['region'] == 'hwasun' else "나주"
        results.append(f"{region_kr:10s} {layer_names[score_info['category']]:15s} "
                      f"{score_info['type']:25s} {score_info['indicator_score']*100:10.2f} "
                      f"{score_info['layer_weight']:12.4f} {score_info['final_score']:10.2f}")
    results.append("")

    # 지역별 종합 점수
    results.append("\n" + "=" * 80)
    results.append("STEP 6. 지역별 최종 종합 점수")
    results.append("=" * 80)
    results.append("")

    region_scores = {}
    for key, score_info in scores.items():
        region = score_info['region']
        if region not in region_scores:
            region_scores[region] = []
        region_scores[region].append(score_info['final_score'])

    results.append(f"{'지역':10s} {'평균점수':>12s} {'최고점수':>12s} {'최저점수':>12s} {'표준편차':>12s}")
    results.append("-" * 80)

    for region in ['hwasun', 'naju']:
        if region in region_scores:
            scores_list = region_scores[region]
            avg = sum(scores_list) / len(scores_list)
            max_score = max(scores_list)
            min_score = min(scores_list)
            std = (sum((s - avg)**2 for s in scores_list) / len(scores_list)) ** 0.5

            region_kr = "화순" if region == 'hwasun' else "나주"
            results.append(f"{region_kr:10s} {avg:12.2f} {max_score:12.2f} {min_score:12.2f} {std:12.2f}")
    results.append("")

    # 평가 기준 제시
    results.append("\n" + "=" * 80)
    results.append("STEP 7. 농업진흥지역 평가 기준 (엔트로피 가중치 기반)")
    results.append("=" * 80)
    results.append("")

    results.append("【점수 기반 등급 분류】")
    results.append("-" * 80)
    results.append("")
    results.append("1등급 (절대보전): 80점 이상")
    results.append("  - 우량농지 + 기반시설 수혜 + 집단화 우수")
    results.append("  - 농업진흥지역 지정 유지 필수")
    results.append("  - 개발행위 엄격 제한")
    results.append("")
    results.append("2등급 (우선보전): 60~80점")
    results.append("  - 우량농지 또는 양호한 기반시설")
    results.append("  - 농업진흥지역 유지 권장")
    results.append("  - 부분적 개선사업 추진")
    results.append("")
    results.append("3등급 (조건부보전): 40~60점")
    results.append("  - 농지로서 가치는 있으나 개선 필요")
    results.append("  - 농지 정비사업 병행 시 유지")
    results.append("  - 집단화·기반시설 투자 우선")
    results.append("")
    results.append("4등급 (해제검토): 20~40점")
    results.append("  - 파편화 심각 또는 영농여건 불리")
    results.append("  - 정비사업 효과 분석 후 결정")
    results.append("  - 집단화 불가능 시 해제 검토")
    results.append("")
    results.append("5등급 (해제우선): 20점 미만")
    results.append("  - 농업적 이용가치 매우 낮음")
    results.append("  - 농업진흥지역 지정해제 우선 검토")
    results.append("  - 타 용도 전환 또는 환경보전 고려")
    results.append("")

    # 모델 활용 방안
    results.append("\n" + "=" * 80)
    results.append("STEP 8. 모델 활용 및 정책 제언")
    results.append("=" * 80)
    results.append("")

    results.append("【엔트로피 가중치법의 장점】")
    results.append("-" * 80)
    results.append("")
    results.append("1. 객관성: 데이터 자체의 정보량을 기반으로 가중치 산정")
    results.append("   - 주관적 판단 배제")
    results.append("   - 지역별 데이터 특성 반영")
    results.append("")
    results.append("2. 정보의 다양성 반영: 엔트로피가 낮을수록 높은 가중치")
    results.append("   - 변별력 있는 지표에 높은 가중치 부여")
    results.append("   - 동일한 값을 가진 지표는 낮은 가중치")
    results.append("")
    results.append("3. 재현성: 동일한 데이터에 대해 일관된 결과")
    results.append("   - 투명한 의사결정")
    results.append("   - 과학적 근거 제시")
    results.append("")

    results.append("【정책 활용 방안】")
    results.append("-" * 80)
    results.append("")
    results.append("1. 1단계: 전수 조사 및 점수 산정")
    results.append("   - 모든 농업진흥지역에 대해 FRAGSTATS 분석")
    results.append("   - 엔트로피 가중치 모델로 점수 계산")
    results.append("")
    results.append("2. 2단계: 등급별 관리 방안 수립")
    results.append("   - 1~2등급: 절대/우선 보전 구역 지정")
    results.append("   - 3등급: 정비사업 우선 추진")
    results.append("   - 4~5등급: 해제 검토 또는 용도 재조정")
    results.append("")
    results.append("3. 3단계: 정비사업 효과 분석")
    results.append("   - 사업 전후 점수 변화 측정")
    results.append("   - ROI 분석 및 우선순위 조정")
    results.append("")
    results.append("4. 4단계: 주기적 재평가")
    results.append("   - 3~5년 주기 재평가")
    results.append("   - 가중치 재계산 (여건 변화 반영)")
    results.append("")

    results.append("【모델의 한계 및 보완 방안】")
    results.append("-" * 80)
    results.append("")
    results.append("1. 공간적 연결성 미반영")
    results.append("   → 보완: 인접 농지와의 연결성 지표 추가")
    results.append("")
    results.append("2. 사회경제적 요인 미포함")
    results.append("   → 보완: 농가 의향, 경제성 분석 병행")
    results.append("")
    results.append("3. 지역별 특성 차이")
    results.append("   → 보완: 지역별 가중치 조정 또는 별도 모델 구축")
    results.append("")

    return "\n".join(results), scores, indicator_weights, layer_weights

def main():
    """메인 함수"""
    print("엔트로피 가중치 기반 농업진흥지역 평가 모델 생성 중...")
    print("")

    report, scores, indicator_weights, layer_weights = generate_evaluation_model()

    # 결과 저장
    output_file = work_dir / "엔트로피가중치_평가모델_결과.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print("=" * 80)
    print("모델 생성 완료!")
    print("=" * 80)
    print(f"결과 파일: {output_file}")
    print(f"총 {len(report):,} 문자")
    print(f"총 {len(report.split(chr(10))):,} 줄")
    print("")

    # 요약 통계 출력
    print("【주요 결과 요약】")
    print("-" * 80)
    print("\n▶ 레이어별 가중치:")
    for cat, weight in sorted(layer_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"   {cat:10s}: {weight:.4f} ({weight*100:.2f}%)")

    print("\n▶ 상위 5개 핵심 지표:")
    sorted_indicators = sorted(indicator_weights.items(),
                              key=lambda x: x[1]['weight'], reverse=True)
    for i, (indicator, info) in enumerate(sorted_indicators[:5], 1):
        print(f"   {i}. {info['name']:15s} ({indicator}): {info['weight']:.4f}")

    print("")

if __name__ == "__main__":
    main()
