#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRAGSTATS 분석 결과 데이터 분석 스크립트
농업진흥지역 지정해제 기준 정립을 위한 분석
"""

import os
from datetime import datetime
from pathlib import Path

# 작업 디렉토리 설정
work_dir = Path("/Users/yunhyungchang/Documents/FRAGSTATS")

def read_fragstats_file(file_path):
    """FRAGSTATS 결과 파일 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if len(lines) < 2:
            return None

        # 헤더 파싱
        header = lines[0].strip().split('\t')
        header = [h.strip() for h in header]

        # 데이터 파싱
        data = []
        for line in lines[1:]:
            if line.strip():
                values = line.strip().split('\t')
                values = [v.strip() for v in values]
                if len(values) == len(header):
                    row = dict(zip(header, values))
                    data.append(row)

        return {'header': header, 'data': data}

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def safe_float(value):
    """안전하게 float로 변환"""
    try:
        if value == 'N/A' or value == '' or value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def calculate_stats(values):
    """기술통계량 계산"""
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return None

    n = len(valid_values)
    mean = sum(valid_values) / n
    sorted_vals = sorted(valid_values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    min_val = min(valid_values)
    max_val = max(valid_values)

    # 표준편차
    variance = sum((x - mean) ** 2 for x in valid_values) / n
    std = variance ** 0.5

    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'median': median,
        'count': n
    }

def analyze_class_metrics():
    """Class 레벨 지표 분석 (31개 지표)"""
    results = []
    results.append("=" * 80)
    results.append("CLASS 레벨 지표 분석 (31개 지표)")
    results.append("=" * 80)
    results.append("")

    # 분석할 파일들
    class_files = {
        'infra': 'infra_class.txt',
        'toyang': 'toyang_class.txt',
        'nongeup': 'nongeup_class.txt',
        'pibok': 'pibok_class.txt'
    }

    for category, filename in class_files.items():
        file_path = work_dir / filename
        file_data = read_fragstats_file(file_path)

        if file_data is None:
            continue

        data = file_data['data']

        results.append(f"\n{'=' * 80}")
        results.append(f"카테고리: {category.upper()}")
        results.append(f"{'=' * 80}")

        # LOC별로 그룹화
        loc_groups = {}
        for row in data:
            loc = row.get('LOC', '')
            if loc not in loc_groups:
                loc_groups[loc] = []
            loc_groups[loc].append(row)

        # TYPE별 분석
        for loc, rows in loc_groups.items():
            results.append(f"\n지역: {loc}")
            results.append("-" * 80)

            type_groups = {}
            for row in rows:
                type_val = row.get('TYPE', '')
                if type_val not in type_groups:
                    type_groups[type_val] = []
                type_groups[type_val].append(row)

            for type_val, type_rows in type_groups.items():
                results.append(f"\n  타입: {type_val}")

                # 모든 지표 출력 (31개)
                key_metrics = ['CA', 'PLAND', 'NP', 'PD', 'LPI', 'TE', 'ED',
                             'AREA_MN', 'AREA_AM', 'AREA_MD', 'AREA_CV',
                             'GYRATE_MN', 'GYRATE_AM', 'GYRATE_MD', 'GYRATE_CV',
                             'SHAPE_MN', 'SHAPE_AM', 'FRAC_MN', 'FRAC_AM', 'PARA_MN',
                             'TCA', 'CPLAND', 'NDCA', 'DCAD', 'CORE_MN', 'CAI_MN',
                             'CLUMPY', 'PLADJ', 'IJI', 'COHESION', 'AI']

                if type_rows:
                    row = type_rows[0]  # Class 레벨은 각 타입당 1개 행
                    for metric in key_metrics:
                        if metric in row:
                            value = safe_float(row[metric])
                            if value is not None:
                                results.append(f"    {metric:15s}: {value:>15.4f}")
                            else:
                                results.append(f"    {metric:15s}: {row[metric]:>15s}")

        # 농지 적합 타입 vs 비적합 타입 비교
        results.append(f"\n\n{'=' * 80}")
        results.append(f"{category.upper()} - 농지 적합 타입 vs 비적합 타입 비교")
        results.append(f"{'=' * 80}")

        if category == 'infra':
            suitable_type = 'giban_benefited'
            unsuitable_type = 'giban_not_benefited'
        else:
            suitable_type = 'cls_1'
            unsuitable_type = 'cls_9'

        # 지역별 비교
        for loc, rows in loc_groups.items():
            results.append(f"\n지역: {loc}")
            results.append("-" * 80)

            suitable = None
            unsuitable = None

            for row in rows:
                if row.get('TYPE') == suitable_type:
                    suitable = row
                elif row.get('TYPE') == unsuitable_type:
                    unsuitable = row

            if suitable and unsuitable:
                results.append(f"\n{'지표':15s} {'농지적합':>15s} {'비적합':>15s} {'차이':>15s} {'차이율(%)':>15s}")
                results.append("-" * 80)

                key_metrics = ['CA', 'PLAND', 'NP', 'PD', 'LPI', 'TE', 'ED',
                             'AREA_MN', 'AREA_AM', 'AREA_MD', 'AREA_CV',
                             'GYRATE_MN', 'GYRATE_AM', 'GYRATE_MD', 'GYRATE_CV',
                             'SHAPE_MN', 'SHAPE_AM', 'FRAC_MN', 'FRAC_AM', 'PARA_MN',
                             'TCA', 'CPLAND', 'NDCA', 'DCAD', 'CORE_MN', 'CAI_MN',
                             'CLUMPY', 'PLADJ', 'IJI', 'COHESION', 'AI']

                for metric in key_metrics:
                    if metric in suitable and metric in unsuitable:
                        s_val = safe_float(suitable[metric])
                        u_val = safe_float(unsuitable[metric])

                        if s_val is not None and u_val is not None:
                            diff = s_val - u_val
                            if u_val != 0:
                                diff_rate = (diff / u_val) * 100
                            else:
                                diff_rate = 0
                            results.append(f"{metric:15s} {s_val:15.4f} {u_val:15.4f} {diff:15.4f} {diff_rate:15.2f}")
                        else:
                            s_str = str(suitable[metric]) if s_val is None else f"{s_val:.4f}"
                            u_str = str(unsuitable[metric]) if u_val is None else f"{u_val:.4f}"
                            results.append(f"{metric:15s} {s_str:>15s} {u_str:>15s} {'N/A':>15s} {'N/A':>15s}")

        results.append("\n")

    return "\n".join(results)

def analyze_land_metrics():
    """Land 레벨 지표 분석 (19개 지표)"""
    results = []
    results.append("\n" + "=" * 80)
    results.append("LAND 레벨 지표 분석 (19개 지표)")
    results.append("=" * 80)
    results.append("")

    # 분석할 파일들
    land_files = {
        'infra': 'infra_land.txt',
        'toyang': 'toyang_land.txt',
        'nongeup': 'nongeup_land.txt',
        'pibok': 'pibok_land.txt'
    }

    for category, filename in land_files.items():
        file_path = work_dir / filename
        file_data = read_fragstats_file(file_path)

        if file_data is None:
            continue

        data = file_data['data']

        results.append(f"\n{'=' * 80}")
        results.append(f"카테고리: {category.upper()}")
        results.append(f"{'=' * 80}")

        # 지역별 분석
        for row in data:
            loc = row.get('LOC', '')
            results.append(f"\n지역: {loc}")
            results.append("-" * 80)

            # 모든 지표 출력 (19개)
            key_metrics = ['TA', 'NP', 'PD', 'LPI', 'TE', 'ED', 'TCA', 'CONTAG',
                         'COHESION', 'DIVISION', 'MESH', 'SPLIT', 'PR', 'PRD',
                         'SHDI', 'SIDI', 'MSIDI', 'SHEI', 'AI']

            for metric in key_metrics:
                if metric in row:
                    value = safe_float(row[metric])
                    if value is not None:
                        results.append(f"  {metric:15s}: {value:>15.4f}")
                    else:
                        results.append(f"  {metric:15s}: {row[metric]:>15s}")

        # 지역 간 비교
        results.append(f"\n\n{'=' * 80}")
        results.append(f"{category.upper()} - 나주 vs 화순 지역 비교")
        results.append(f"{'=' * 80}")

        hwasun_data = None
        naju_data = None

        for row in data:
            loc = row.get('LOC', '').lower()
            if 'hwasun' in loc:
                hwasun_data = row
            elif 'naju' in loc:
                naju_data = row

        if hwasun_data and naju_data:
            results.append(f"\n{'지표':15s} {'화순':>15s} {'나주':>15s} {'차이':>15s} {'차이율(%)':>15s}")
            results.append("-" * 80)

            key_metrics = ['TA', 'NP', 'PD', 'LPI', 'TE', 'ED', 'TCA', 'CONTAG',
                         'COHESION', 'DIVISION', 'MESH', 'SPLIT', 'PR', 'PRD',
                         'SHDI', 'SIDI', 'MSIDI', 'SHEI', 'AI']

            for metric in key_metrics:
                if metric in hwasun_data and metric in naju_data:
                    h_val = safe_float(hwasun_data[metric])
                    n_val = safe_float(naju_data[metric])

                    if h_val is not None and n_val is not None:
                        diff = n_val - h_val
                        if h_val != 0:
                            diff_rate = (diff / h_val) * 100
                        else:
                            diff_rate = 0
                        results.append(f"{metric:15s} {h_val:15.4f} {n_val:15.4f} {diff:15.4f} {diff_rate:15.2f}")

        results.append("\n")

    return "\n".join(results)

def analyze_patch_metrics():
    """Patch 레벨 지표 분석 (8개 지표) - 기술통계량"""
    results = []
    results.append("\n" + "=" * 80)
    results.append("PATCH 레벨 지표 분석 (8개 지표)")
    results.append("=" * 80)
    results.append("※ Patch 파일은 크기가 커서 기술통계량만 제공합니다.")
    results.append("")

    # 분석할 파일들
    patch_files = {
        'infra': 'infra_patch.txt',
        'toyang': 'toyang_patch.txt',
        'nongeup': 'nongeup_patch.txt',
    }

    metrics = ['AREA', 'PERIM', 'GYRATE', 'SHAPE', 'FRAC', 'CORE', 'NCORE', 'CAI']

    for category, filename in patch_files.items():
        file_path = work_dir / filename

        results.append(f"\n{'=' * 80}")
        results.append(f"카테고리: {category.upper()}")
        results.append(f"{'=' * 80}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
                header = [h.strip() for h in header_line.split('\t')]

                # 지역/타입별 데이터 수집
                loc_type_data = {}

                for line in f:
                    if not line.strip():
                        continue

                    values = [v.strip() for v in line.strip().split('\t')]
                    if len(values) != len(header):
                        continue

                    row = dict(zip(header, values))
                    loc = row.get('LOC', '')
                    type_val = row.get('TYPE', '')
                    key = (loc, type_val)

                    if key not in loc_type_data:
                        loc_type_data[key] = {metric: [] for metric in metrics}

                    for metric in metrics:
                        if metric in row:
                            val = safe_float(row[metric])
                            if val is not None:
                                loc_type_data[key][metric].append(val)

            # 통계 계산 및 출력
            for (loc, type_val), metric_data in sorted(loc_type_data.items()):
                results.append(f"\n지역: {loc}")
                results.append(f"타입: {type_val}")
                results.append("-" * 80)

                # 패치 수
                patch_count = len(metric_data.get('AREA', []))
                results.append(f"패치 수: {patch_count}")

                results.append(f"\n{'지표':10s} {'평균':>12s} {'표준편차':>12s} {'최소':>12s} {'최대':>12s} {'중앙값':>12s}")
                results.append("-" * 80)

                for metric in metrics:
                    values = metric_data.get(metric, [])
                    stats = calculate_stats(values)

                    if stats:
                        results.append(
                            f"{metric:10s} {stats['mean']:12.4f} {stats['std']:12.4f} "
                            f"{stats['min']:12.4f} {stats['max']:12.4f} {stats['median']:12.4f}"
                        )

                results.append("")

        except Exception as e:
            results.append(f"Error processing {filename}: {str(e)}\n")

    # pibok 파일들 (나주/화순 별도)
    results.append(f"\n{'=' * 80}")
    results.append(f"카테고리: PIBOK (지역별 별도 파일)")
    results.append(f"{'=' * 80}")

    pibok_files = [
        ('pibok_patch_naju.txt', 'naju_pibok'),
        ('pibok_patch_hwasun.txt', 'hwasun_pibok')
    ]

    for filename, loc_name in pibok_files:
        file_path = work_dir / filename

        results.append(f"\n지역: {loc_name}")
        results.append("-" * 80)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                header_line = f.readline().strip()
                header = [h.strip() for h in header_line.split('\t')]

                # 타입별 데이터 수집
                type_data = {}

                for line in f:
                    if not line.strip():
                        continue

                    values = [v.strip() for v in line.strip().split('\t')]
                    if len(values) != len(header):
                        continue

                    row = dict(zip(header, values))
                    type_val = row.get('TYPE', '')

                    if type_val not in type_data:
                        type_data[type_val] = {metric: [] for metric in metrics}

                    for metric in metrics:
                        if metric in row:
                            val = safe_float(row[metric])
                            if val is not None:
                                type_data[type_val][metric].append(val)

            # 통계 계산 및 출력
            for type_val, metric_data in sorted(type_data.items()):
                results.append(f"\n타입: {type_val}")

                # 패치 수
                patch_count = len(metric_data.get('AREA', []))
                results.append(f"패치 수: {patch_count}")

                results.append(f"\n{'지표':10s} {'평균':>12s} {'표준편차':>12s} {'최소':>12s} {'최대':>12s} {'중앙값':>12s}")
                results.append("-" * 80)

                for metric in metrics:
                    values = metric_data.get(metric, [])
                    stats = calculate_stats(values)

                    if stats:
                        results.append(
                            f"{metric:10s} {stats['mean']:12.4f} {stats['std']:12.4f} "
                            f"{stats['min']:12.4f} {stats['max']:12.4f} {stats['median']:12.4f}"
                        )

                results.append("")

        except Exception as e:
            results.append(f"Error processing {filename}: {str(e)}\n")

    return "\n".join(results)

def generate_summary():
    """종합 요약 및 권장사항"""
    results = []
    results.append("\n" + "=" * 80)
    results.append("종합 분석 요약 및 농업진흥지역 지정해제 기준 권장사항")
    results.append("=" * 80)
    results.append("")

    results.append("1. 분석 개요")
    results.append("-" * 80)
    results.append("   - 분석 지역: 나주(naju), 화순(hwasun)")
    results.append("   - 분석 카테고리: 인프라(infra), 토양(toyang), 농업(nongeup), 피복(pibok)")
    results.append("   - 분석 레벨: Class(31개 지표), Land(19개 지표), Patch(8개 지표)")
    results.append("")

    results.append("2. 농지 적합 타입 식별")
    results.append("-" * 80)
    results.append("   - infra: giban_benefited (기반시설 수혜 농지)")
    results.append("   - toyang: cls_1 (1등급 토양)")
    results.append("   - nongeup: cls_1 (1등급 농업용지)")
    results.append("   - pibok: cls_1 (1등급 피복)")
    results.append("")

    results.append("3. 주요 지표 해석")
    results.append("-" * 80)
    results.append("   CA (Class Area): 클래스의 총 면적 - 높을수록 해당 타입이 넓게 분포")
    results.append("   PLAND (Percentage of Landscape): 경관 내 비율 - 해당 타입의 지배도")
    results.append("   NP (Number of Patches): 패치 수 - 파편화 정도")
    results.append("   PD (Patch Density): 패치 밀도 - 단위면적당 패치 수")
    results.append("   LPI (Largest Patch Index): 최대 패치 비율 - 연결성/집중도")
    results.append("   ED (Edge Density): 가장자리 밀도 - 파편화 정도")
    results.append("   CLUMPY: 집괴도 - 0에 가까우면 분산, 1에 가까우면 집중")
    results.append("   AI (Aggregation Index): 집적도 - 높을수록 집중 분포")
    results.append("   COHESION: 응집도 - 패치의 물리적 연결성")
    results.append("   CONTAG: 전염도 - 경관의 집적 정도")
    results.append("")

    results.append("4. 농업진흥지역 지정해제 기준 제안")
    results.append("-" * 80)
    results.append("   분석 결과를 바탕으로 다음 기준을 고려할 수 있습니다:")
    results.append("")
    results.append("   [보전 우선 지역 - 지정 유지 권장]")
    results.append("   - giban_benefited 타입이고 PLAND가 높은 지역 (경관 내 높은 비율)")
    results.append("   - cls_1 타입이면서 CLUMPY > 0.85, AI > 90인 집중 분포 지역")
    results.append("   - LPI가 높아 대규모 농지가 연결된 지역 (예: LPI > 50)")
    results.append("   - NP와 PD가 낮아 파편화가 적은 지역")
    results.append("   - COHESION이 높아 물리적 연결성이 우수한 지역 (예: > 98)")
    results.append("   - TCA/CPLAND 비율이 높아 핵심 농지 면적이 큰 지역")
    results.append("")
    results.append("   [지정해제 검토 가능 지역]")
    results.append("   - giban_not_benefited 타입이면서 NP가 매우 높은 파편화 지역")
    results.append("   - cls_9 타입이고 AREA_MN이 매우 작은 소규모 분산 지역")
    results.append("   - ED가 매우 높아 가장자리가 복잡한 지역 (파편화 지표)")
    results.append("   - CLUMPY < 0.8이고 AI < 85인 분산 분포 지역")
    results.append("   - PLAND가 낮아 경관 내 차지하는 비율이 작은 지역")
    results.append("   - DIVISION이 높아 경관이 세분화된 지역")
    results.append("")

    results.append("5. 종합 평가 기준 제안")
    results.append("-" * 80)
    results.append("   다음 기준을 종합적으로 고려하여 점수화할 수 있습니다:")
    results.append("")
    results.append("   [농지 보전 우선순위 점수 = (가중치 합)]")
    results.append("   - 면적 지표 (30%): CA, PLAND, LPI")
    results.append("   - 집적도 지표 (30%): CLUMPY, AI, COHESION")
    results.append("   - 파편화 지표 (20%): NP, PD, ED (역방향)")
    results.append("   - 핵심면적 지표 (20%): TCA, CPLAND, CORE_MN")
    results.append("")
    results.append("   점수가 높은 지역일수록 농업진흥지역으로 유지하고,")
    results.append("   점수가 낮은 지역은 지정해제를 검토할 수 있습니다.")
    results.append("")

    results.append("6. 추가 분석 제안")
    results.append("-" * 80)
    results.append("   - 4개 카테고리(infra, toyang, nongeup, pibok)의 종합 평가 모델 개발")
    results.append("   - 각 지표에 대한 가중치 설정 및 종합 점수 산정")
    results.append("   - 공간적 연결성과 접근성을 고려한 추가 분석")
    results.append("   - 사회경제적 요인과의 통합 분석")
    results.append("   - 시계열 분석을 통한 변화 추세 파악")
    results.append("")

    return "\n".join(results)

def main():
    """메인 분석 함수"""
    print("FRAGSTATS 데이터 분석을 시작합니다...")
    print("")

    all_results = []

    # 헤더
    all_results.append("=" * 80)
    all_results.append("FRAGSTATS 분석 결과 종합 보고서")
    all_results.append("농업진흥지역 지정해제 기준 정립을 위한 데이터 분석")
    all_results.append("=" * 80)
    all_results.append("")
    all_results.append(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    all_results.append("")

    # Class 레벨 분석
    print("Class 레벨 지표 분석 중...")
    class_results = analyze_class_metrics()
    all_results.append(class_results)

    # Land 레벨 분석
    print("Land 레벨 지표 분석 중...")
    land_results = analyze_land_metrics()
    all_results.append(land_results)

    # Patch 레벨 분석
    print("Patch 레벨 지표 분석 중...")
    patch_results = analyze_patch_metrics()
    all_results.append(patch_results)

    # 종합 요약
    print("종합 요약 생성 중...")
    summary = generate_summary()
    all_results.append(summary)

    # 결과 저장
    output_file = work_dir / "FRAGSTATS_분석결과_종합보고서.txt"
    final_output = "\n".join(all_results)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)

    print("")
    print("=" * 80)
    print("분석 완료!")
    print("=" * 80)
    print(f"결과 파일: {output_file}")
    print(f"총 {len(final_output):,} 문자 저장됨")
    print(f"총 {len(final_output.split(chr(10))):,} 줄 저장됨")
    print("")

if __name__ == "__main__":
    main()
