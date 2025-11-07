#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRAGSTATS 분석 결과 해석 생성 스크립트
농업진흥지역 지정해제를 위한 농지기능적 측면 결과 해석
"""

from datetime import datetime
from pathlib import Path

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

def interpret_infra_class(region_name, data_row, type_name):
    """인프라 Class 레벨 해석"""
    results = []

    pland = safe_float(data_row.get('PLAND', 0))
    lpi = safe_float(data_row.get('LPI', 0))
    np = safe_float(data_row.get('NP', 0))
    ai = safe_float(data_row.get('AI', 0))
    clumpy = safe_float(data_row.get('CLUMPY', 0))
    cohesion = safe_float(data_row.get('COHESION', 0))
    core_mn = safe_float(data_row.get('CORE_MN', 0))
    ed = safe_float(data_row.get('ED', 0))
    area_mn = safe_float(data_row.get('AREA_MN', 0))

    is_suitable = 'benefited' in type_name

    results.append(f"【{region_name} - {type_name}】")
    results.append("")

    # 농지 면적 비율 분석
    if pland:
        if is_suitable:
            if pland < 10:
                results.append(f"- 기반시설 수혜 농지가 경관의 {pland:.1f}%만 차지하여 매우 열악한 상황")
            elif pland < 30:
                results.append(f"- 기반시설 수혜 농지가 경관의 {pland:.1f}%로 개선 필요")
            else:
                results.append(f"- 기반시설 수혜 농지가 경관의 {pland:.1f}%로 양호한 수준")
        else:
            if pland > 80:
                results.append(f"- 기반시설 미수혜 농지가 경관의 {pland:.1f}%로 매우 높은 비중")
            results.append("  → 기반시설 투자 우선순위 지역 검토 필요")

    # 연결성 및 집적도 분석
    if lpi and ai and clumpy:
        if is_suitable:
            if lpi < 1:
                results.append(f"- 최대 패치가 경관의 {lpi:.2f}%에 불과하여 매우 분산된 상태")
                results.append("  → 농지 집단화 사업 필요")
            else:
                results.append(f"- 최대 패치가 경관의 {lpi:.2f}%로 일정 수준의 집적")

            if ai < 85:
                results.append(f"- 집적도(AI={ai:.1f})가 낮아 농기계 이동 및 영농 효율성 저하")
            elif ai < 90:
                results.append(f"- 집적도(AI={ai:.1f})가 보통 수준")
            else:
                results.append(f"- 집적도(AI={ai:.1f})가 높아 영농 효율성 양호")
        else:
            if lpi > 70:
                results.append(f"- 미수혜 농지가 하나의 큰 덩어리(LPI={lpi:.1f})로 존재")
                results.append("  → 대규모 기반시설 투자 효과 기대 가능")

    # 파편화 분석
    if np and ed:
        if is_suitable:
            if np > 2000:
                results.append(f"- 패치 수({np:.0f}개)가 매우 많아 극심한 파편화 상태")
                results.append(f"- 가장자리 밀도(ED={ed:.1f})가 높아 관리 비용 증가")
                results.append("  → 농지 정비 및 교환·분합 사업 필요")
            elif np > 500:
                results.append(f"- 패치 수({np:.0f}개)가 많아 파편화된 상태")
                results.append("  → 농지 정비 고려 필요")

    # 핵심 농지 분석
    if core_mn:
        if is_suitable:
            if core_mn < 5:
                results.append(f"- 핵심 농지 면적(CORE_MN={core_mn:.2f})이 매우 작음")
                results.append("  → 가장자리 효과에 취약, 생산성 저하 우려")
            elif core_mn < 20:
                results.append(f"- 핵심 농지 면적(CORE_MN={core_mn:.2f})이 작은 편")
            else:
                results.append(f"- 핵심 농지 면적(CORE_MN={core_mn:.2f})이 양호")

    # 농지 규모 분석
    if area_mn:
        if is_suitable:
            if area_mn < 10:
                results.append(f"- 평균 패치 면적({area_mn:.2f}ha)이 매우 작아 영농 규모화 어려움")
                results.append("  → 경쟁력 있는 농업 경영 곤란")
            elif area_mn < 30:
                results.append(f"- 평균 패치 면적({area_mn:.2f}ha)이 작은 편")

    # 종합 평가
    results.append("")
    results.append("【농지기능 종합 평가】")
    if is_suitable:
        score = 0
        if pland and pland > 15: score += 1
        if lpi and lpi > 1: score += 1
        if ai and ai > 90: score += 1
        if cohesion and cohesion > 98: score += 1
        if core_mn and core_mn > 10: score += 1

        if score >= 4:
            results.append("✓ 농업진흥지역으로 유지 권장 (농지기능 우수)")
            results.append("  - 기반시설 수혜, 집적화 양호, 생산성 유지 가능")
        elif score >= 2:
            results.append("△ 조건부 유지 또는 정비 후 유지 권장")
            results.append("  - 농지 정비사업 병행 시 농업진흥지역 유지 가능")
        else:
            results.append("✗ 지정해제 검토 가능 (농지기능 미흡)")
            results.append("  - 극심한 파편화, 기반시설 부족으로 농업생산성 저하")
    else:
        if pland and pland > 80:
            results.append("※ 기반시설 투자 우선 지역")
            results.append("  - 대규모 미수혜 농지 존재, 기반시설 투자 효과 클 것으로 예상")

    results.append("")
    return "\n".join(results)

def interpret_toyang_class(region_name, data_row, type_name):
    """토양 Class 레벨 해석"""
    results = []

    pland = safe_float(data_row.get('PLAND', 0))
    lpi = safe_float(data_row.get('LPI', 0))
    np = safe_float(data_row.get('NP', 0))
    ai = safe_float(data_row.get('AI', 0))
    clumpy = safe_float(data_row.get('CLUMPY', 0))
    cohesion = safe_float(data_row.get('COHESION', 0))
    area_mn = safe_float(data_row.get('AREA_MN', 0))

    is_suitable = 'cls_1' in type_name

    results.append(f"【{region_name} - {type_name}】")
    results.append("")

    # 토양 등급별 면적 분석
    if pland:
        if is_suitable:
            results.append(f"- 1등급 토양(우량농지)이 경관의 {pland:.1f}% 차지")
            if pland > 30:
                results.append("  → 우량농지 비율이 높아 농업생산성 우수")
            elif pland > 20:
                results.append("  → 우량농지 비율이 적정 수준")
            else:
                results.append("  → 우량농지 비율이 낮은 편")
        else:
            results.append(f"- 9등급 토양(저위생산 농지)이 경관의 {pland:.1f}% 차지")
            if pland > 70:
                results.append("  → 토양조건 불량 지역, 토양개량 필요성 높음")

    # 우량농지 집적도 분석
    if is_suitable and lpi and ai:
        if lpi > 5:
            results.append(f"- 최대 우량농지 패치가 경관의 {lpi:.1f}%로 집중 분포")
            results.append("  → 집단화된 우량농지 존재, 규모화 영농 가능")
        else:
            results.append(f"- 최대 우량농지 패치가 경관의 {lpi:.1f}%로 분산")
            results.append("  → 소규모 우량농지 산재, 집단화 필요")

        if ai > 93:
            results.append(f"- 집적도(AI={ai:.1f})가 매우 높아 우량농지 집중")
        elif ai > 90:
            results.append(f"- 집적도(AI={ai:.1f})가 높은 편")

    # 파편화 및 농지 규모
    if is_suitable and np and area_mn:
        if np > 500:
            results.append(f"- 우량농지가 {np:.0f}개 패치로 분산되어 파편화 심각")
            results.append(f"- 평균 면적({area_mn:.2f}ha)이 작아 영농 규모화 제약")
            results.append("  → 농지 교환·분합 사업 필요")
        elif np > 200:
            results.append(f"- 우량농지가 {np:.0f}개 패치로 분산")

    # 종합 평가
    results.append("")
    results.append("【토양기능 종합 평가】")
    if is_suitable:
        score = 0
        if pland and pland > 25: score += 1
        if lpi and lpi > 2: score += 1
        if ai and ai > 93: score += 1
        if cohesion and cohesion > 99: score += 1
        if area_mn and area_mn > 30: score += 1

        if score >= 4:
            results.append("✓ 우량농지 보전 최우선 지역")
            results.append("  - 1등급 토양 집중, 농업생산성 최고 수준")
            results.append("  - 농업진흥지역 지정 유지 필수")
        elif score >= 2:
            results.append("△ 우량농지 보전 권장")
            results.append("  - 토양조건 양호하나 파편화 개선 필요")
        else:
            results.append("△ 조건부 보전")
            results.append("  - 우량농지이나 분산도가 높아 정비사업 병행 필요")
    else:
        if pland and pland > 70:
            results.append("※ 토양개량 또는 용도전환 검토 지역")
            results.append("  - 저위생산 토양 우세, 농업적 이용가치 낮음")
            results.append("  - 토양개량사업 또는 농업진흥지역 지정해제 검토")

    results.append("")
    return "\n".join(results)

def interpret_land_level(category, region_name, data_row):
    """Land 레벨 해석"""
    results = []

    contag = safe_float(data_row.get('CONTAG', 0))
    division = safe_float(data_row.get('DIVISION', 0))
    shdi = safe_float(data_row.get('SHDI', 0))
    ai = safe_float(data_row.get('AI', 0))
    cohesion = safe_float(data_row.get('COHESION', 0))
    pd = safe_float(data_row.get('PD', 0))

    results.append(f"【{region_name} - {category.upper()} 경관 레벨】")
    results.append("")

    # 경관 집적도 분석
    if contag:
        if contag > 60:
            results.append(f"- 전염도(CONTAG={contag:.1f})가 높아 경관이 단순하고 집중")
            results.append("  → 동일 용도 농지가 집중 분포, 영농 효율성 높음")
        elif contag > 45:
            results.append(f"- 전염도(CONTAG={contag:.1f})가 중간 수준")
            results.append("  → 경관 복잡도가 보통")
        else:
            results.append(f"- 전염도(CONTAG={contag:.1f})가 낮아 경관이 복잡하고 분산")
            results.append("  → 토지이용 혼재, 영농 효율성 저하")

    # 경관 분할도 분석
    if division:
        if division < 0.2:
            results.append(f"- 경관 분할도(DIVISION={division:.3f})가 낮아 통합성 높음")
            results.append("  → 대규모 연속 농지 존재")
        elif division < 0.5:
            results.append(f"- 경관 분할도(DIVISION={division:.3f})가 보통 수준")
        else:
            results.append(f"- 경관 분할도(DIVISION={division:.3f})가 높아 세분화")
            results.append("  → 경관이 여러 작은 조각으로 분할, 연속성 부족")

    # 경관 다양성 분석
    if shdi:
        if shdi < 0.3:
            results.append(f"- 다양성 지수(SHDI={shdi:.3f})가 낮아 단순한 경관")
            results.append("  → 특정 토지이용 우세, 농업 집중도 높음")
        elif shdi < 0.6:
            results.append(f"- 다양성 지수(SHDI={shdi:.3f})가 중간 수준")
        else:
            results.append(f"- 다양성 지수(SHDI={shdi:.3f})가 높아 복잡한 경관")
            results.append("  → 다양한 토지이용 혼재")

    # 경관 연결성
    if cohesion and ai:
        if cohesion > 99.5:
            results.append(f"- 응집도(COHESION={cohesion:.2f})가 매우 높음")
            results.append("  → 경관 요소들의 물리적 연결성 우수")

        if ai > 97:
            results.append(f"- 집적도(AI={ai:.1f})가 매우 높음")
            results.append("  → 경관 전체적으로 집중 분포 패턴")
        elif ai < 90:
            results.append(f"- 집적도(AI={ai:.1f})가 낮은 편")
            results.append("  → 경관 요소 분산")

    # 파편화 정도
    if pd:
        if pd > 15:
            results.append(f"- 패치 밀도(PD={pd:.1f})가 매우 높음")
            results.append("  → 극심한 파편화, 경관 복잡도 증가")
        elif pd > 5:
            results.append(f"- 패치 밀도(PD={pd:.1f})가 높은 편")
        elif pd < 1:
            results.append(f"- 패치 밀도(PD={pd:.2f})가 낮음")
            results.append("  → 대규모 통합 농지 존재")

    # 종합 평가
    results.append("")
    results.append("【경관 기능 종합 평가】")
    score = 0
    if contag and contag > 50: score += 1
    if division and division < 0.3: score += 1
    if cohesion and cohesion > 99.5: score += 1
    if ai and ai > 95: score += 1
    if pd and pd < 10: score += 1

    if score >= 4:
        results.append("✓ 우수한 농업 경관")
        results.append("  - 집중·통합된 농지 경관, 영농 효율성 최고")
        results.append("  - 농업진흥지역으로 보전 필수")
    elif score >= 3:
        results.append("○ 양호한 농업 경관")
        results.append("  - 전반적으로 양호한 경관 구조")
        results.append("  - 농업진흥지역 유지 권장")
    elif score >= 2:
        results.append("△ 보통 수준의 농업 경관")
        results.append("  - 부분적 개선 필요")
    else:
        results.append("✗ 열악한 농업 경관")
        results.append("  - 경관 파편화 및 분산 심각")
        results.append("  - 농지 정비 없이는 경쟁력 있는 농업 곤란")
        results.append("  - 농업진흥지역 지정해제 검토 또는 대규모 정비사업 필요")

    results.append("")
    return "\n".join(results)

def generate_comprehensive_interpretation():
    """종합 해석 보고서 생성"""
    results = []

    results.append("=" * 80)
    results.append("FRAGSTATS 분석 결과 해석 보고서")
    results.append("농업진흥지역 지정해제를 위한 농지기능적 측면 결과 해석")
    results.append("=" * 80)
    results.append("")
    results.append(f"작성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results.append("")

    # CLASS 레벨 해석
    results.append("\n" + "=" * 80)
    results.append("제1장. CLASS 레벨 분석 - 농지 유형별 세부 해석")
    results.append("=" * 80)
    results.append("")

    # 1-1. INFRA 카테고리
    results.append("\n1-1. 기반시설 수혜도 분석 (INFRA)")
    results.append("-" * 80)
    results.append("※ 기반시설 수혜 농지(giban_benefited)만 분석 대상")
    results.append("")

    infra_file = work_dir / "infra_class.txt"
    infra_data = read_fragstats_file(infra_file)
    if infra_data:
        for row in infra_data['data']:
            loc = row.get('LOC', '')
            type_val = row.get('TYPE', '')
            # giban_benefited만 분석
            if 'benefited' in type_val and 'not' not in type_val:
                region = "화순" if "hwasun" in loc else "나주"
                results.append(interpret_infra_class(region, row, type_val))

    # 1-2. TOYANG 카테고리
    results.append("\n1-2. 토양 등급별 분석 (TOYANG)")
    results.append("-" * 80)
    results.append("※ 1등급 토양(cls_1)만 분석 대상")
    results.append("")

    toyang_file = work_dir / "toyang_class.txt"
    toyang_data = read_fragstats_file(toyang_file)
    if toyang_data:
        for row in toyang_data['data']:
            loc = row.get('LOC', '')
            type_val = row.get('TYPE', '')
            # cls_1만 분석
            if 'cls_1' in type_val:
                region = "화순" if "hwasun" in loc else "나주"
                results.append(interpret_toyang_class(region, row, type_val))

    # 1-3. NONGEUP 카테고리 (토양과 유사한 해석)
    results.append("\n1-3. 농업용지 등급별 분석 (NONGEUP)")
    results.append("-" * 80)
    results.append("※ 1등급 농업용지(cls_1)만 분석 대상")
    results.append("")

    nongeup_file = work_dir / "nongeup_class.txt"
    nongeup_data = read_fragstats_file(nongeup_file)
    if nongeup_data:
        for row in nongeup_data['data']:
            loc = row.get('LOC', '')
            type_val = row.get('TYPE', '')
            # cls_1만 분석
            if 'cls_1' in type_val:
                region = "화순" if "hwasun" in loc else "나주"
                # nongeup도 토양과 유사하게 해석
                results.append(interpret_toyang_class(region, row, type_val).replace("토양", "농업용지").replace("TOYANG", "NONGEUP"))

    # 1-4. PIBOK 카테고리
    results.append("\n1-4. 토지피복 등급별 분석 (PIBOK)")
    results.append("-" * 80)
    results.append("※ 1등급 토지피복(cls_1)만 분석 대상")
    results.append("")

    pibok_file = work_dir / "pibok_class.txt"
    pibok_data = read_fragstats_file(pibok_file)
    if pibok_data:
        for row in pibok_data['data']:
            loc = row.get('LOC', '')
            type_val = row.get('TYPE', '')
            # cls_1만 분석
            if 'cls_1' in type_val:
                region = "화순" if "hwasun" in loc else "나주"
                results.append(interpret_toyang_class(region, row, type_val).replace("토양", "토지피복").replace("TOYANG", "PIBOK").replace("우량농지", "우량피복지"))

    # LAND 레벨 해석
    results.append("\n" + "=" * 80)
    results.append("제2장. LAND 레벨 분석 - 경관 전체 특성 해석")
    results.append("=" * 80)
    results.append("")

    categories = ['infra', 'toyang', 'nongeup', 'pibok']
    for category in categories:
        land_file = work_dir / f"{category}_land.txt"
        land_data = read_fragstats_file(land_file)
        if land_data:
            for row in land_data['data']:
                loc = row.get('LOC', '')
                region = "화순" if "hwasun" in loc else "나주"
                results.append(interpret_land_level(category, region, row))

    # 지역 간 비교
    results.append("\n" + "=" * 80)
    results.append("제3장. 지역 간 비교 분석")
    results.append("=" * 80)
    results.append("")

    results.append("3-1. 나주 vs 화순 종합 비교")
    results.append("-" * 80)
    results.append("")

    results.append("【화순 지역 - 농지 적합 타입】")
    results.append("- 기반시설 수혜 농지: 경관의 4.1%로 매우 낮음")
    results.append("  → 극심한 파편화 (812개 패치), 평균 4.02ha로 소규모")
    results.append("  → 농지 집단화 사업 및 기반시설 확충 시급")
    results.append("- 1등급 토양: 경관의 25.8%로 적정 수준")
    results.append("  → 우량농지 산재 (452개 패치), 집단화 필요")
    results.append("  → 집적도 높음 (AI=93.1)")
    results.append("- 1등급 농업용지: 경관의 15.5%")
    results.append("  → 파편화 상태 (405개 패치)")
    results.append("- 1등급 토지피복: 경관의 11.6%")
    results.append("  → 매우 심각한 파편화 (14,933개 패치)")
    results.append("")
    results.append("【화순 종합 평가】")
    results.append("✓ 우량농지(토양 1등급)는 보전 가치 있으나 파편화 심각")
    results.append("✗ 기반시설 수혜 농지는 극소량으로 집단화·정비 시급")
    results.append("→ 우량농지 중심 집단화 사업 우선 추진 필요")
    results.append("")

    results.append("【나주 지역 - 농지 적합 타입】")
    results.append("- 기반시설 수혜 농지: 경관의 19.8%로 화순보다 양호")
    results.append("  → 파편화 심각 (2,209개 패치), 평균 5.42ha")
    results.append("  → 농지 정비사업 필요")
    results.append("- 1등급 토양: 경관의 29.0%로 우수")
    results.append("  → 파편화 매우 심각 (749개 패치), 평균 23.49ha")
    results.append("  → 집적도 높음 (AI=94.1)")
    results.append("- 1등급 농업용지: 경관의 37.3%로 높음")
    results.append("  → 파편화 (356개 패치), 평균 63.29ha로 상대적으로 양호")
    results.append("- 1등급 토지피복: 경관의 35.7%")
    results.append("  → 심각한 파편화 (12,068개 패치)")
    results.append("")
    results.append("【나주 종합 평가】")
    results.append("✓ 우량농지 비율이 높아 농업생산성 우수")
    results.append("✗ 극심한 파편화가 최대 약점")
    results.append("→ 농지 교환·분합 사업을 통한 집단화가 최우선 과제")
    results.append("")

    # 최종 권고사항
    results.append("\n" + "=" * 80)
    results.append("제4장. 농업진흥지역 지정해제 기준 및 정책 권고")
    results.append("=" * 80)
    results.append("")

    results.append("4-1. 지정 유지 우선순위 기준")
    results.append("-" * 80)
    results.append("")
    results.append("【1순위 - 절대보전】")
    results.append("✓ 조건:")
    results.append("  - 1등급 토양 + 기반시설 수혜 농지")
    results.append("  - PLAND > 20% AND LPI > 5%")
    results.append("  - AI > 93 AND COHESION > 99")
    results.append("  - CORE_MN > 20ha")
    results.append("✓ 사유: 우량농지가 집단화되어 생산성 최고 수준")
    results.append("")

    results.append("【2순위 - 우선보전】")
    results.append("✓ 조건:")
    results.append("  - 1등급 토양 또는 기반시설 수혜 농지")
    results.append("  - PLAND > 15%")
    results.append("  - AI > 90 AND COHESION > 98")
    results.append("  - 경관 전체 CONTAG > 50")
    results.append("✓ 사유: 양호한 농지 조건 및 경관 구조")
    results.append("")

    results.append("【3순위 - 조건부 보전】")
    results.append("✓ 조건:")
    results.append("  - 1등급 토양이나 파편화 심각 (NP > 500)")
    results.append("  - 기반시설 수혜이나 분산도 높음 (LPI < 1)")
    results.append("✓ 사유: 농지 정비사업 병행 시 보전 가능")
    results.append("✓ 요구사항: 농지 교환·분합, 집단화 사업 선행")
    results.append("")

    results.append("\n4-2. 지정해제 검토 기준 (농지 적합 타입 내 차등화)")
    results.append("-" * 80)
    results.append("")
    results.append("【해제 우선 검토 대상】")
    results.append("✗ 조건: (농지 적합 타입이지만 다음 조건 충족 시)")
    results.append("  - PLAND < 5% (경관 내 비율 극히 낮음)")
    results.append("  - NP > 2000 AND AREA_MN < 3ha (극심한 소규모 파편화)")
    results.append("  - ED > 80 (가장자리 밀도 높음)")
    results.append("  - LPI < 0.3 (최대 패치도 미미)")
    results.append("  - AI < 85 (낮은 집적도)")
    results.append("✗ 사유: 우량 농지이지만 극도로 파편화되어 정비 비용 과다")
    results.append("✗ 판단: 농지 정비사업 효과 분석 후 결정")
    results.append("       - 정비 가능: 집단화 후 보전")
    results.append("       - 정비 곤란: 해제 검토")
    results.append("")

    results.append("【조건부 해제 검토 대상】")
    results.append("△ 조건:")
    results.append("  - PLAND < 10% (경관 내 비율 낮음)")
    results.append("  - NP > 1000 AND AREA_MN < 5ha (심각한 파편화)")
    results.append("  - CORE_MN < 3ha (핵심 농지 면적 매우 작음)")
    results.append("  - DIVISION > 0.5 (경관 분할 심각)")
    results.append("△ 사유: 우량 농지이나 분산도 높아 영농 효율성 저하")
    results.append("△ 판단: 주변 여건 및 집단화 가능성 검토 후 결정")
    results.append("")

    results.append("\n4-3. 정책 제안 (농지 적합 타입 중심)")
    results.append("-" * 80)
    results.append("")
    results.append("1. 우량 농지 차등 관리 체계")
    results.append("   - 1등급 토양 + 기반시설 수혜 + 집단화: 절대보전")
    results.append("   - 1등급 토양이지만 파편화: 집단화 사업 우선")
    results.append("   - 기반시설 수혜이지만 분산: 정비 후 보전")
    results.append("   - 극도로 파편화된 소규모 농지: 집단화 가능성 검토 후 결정")
    results.append("")
    results.append("2. 농지 집단화 사업 추진 전략")
    results.append("   - 우선순위 1: 1등급 토양 집중 지역 (PLAND > 25%)")
    results.append("   - 우선순위 2: 기반시설 수혜 농지 밀집 지역")
    results.append("   - 교환·분합을 통한 필지 통합")
    results.append("   - 집단화 효과가 큰 지역부터 단계적 추진")
    results.append("")
    results.append("3. 기반시설 확충 전략")
    results.append("   - 1등급 토양 지역 우선 투자")
    results.append("   - 집단화된 농지에 기반시설 집중 투자")
    results.append("   - 투자 효과 분석 및 우선순위 설정")
    results.append("")
    results.append("4. 단계적 의사결정 프로세스")
    results.append("   - 1단계: 절대보전 구역 지정 (1순위 농지)")
    results.append("   - 2단계: 집단화 가능 구역 선정 및 사업 추진")
    results.append("   - 3단계: 집단화 사업 완료 후 재평가")
    results.append("   - 4단계: 집단화 불가능 또는 비효율적 지역 해제 검토")
    results.append("   - 5단계: 주민 의견 수렴 및 최종 결정")
    results.append("")

    results.append("\n4-4. 지역별 맞춤 전략 (농지 적합 타입 중심)")
    results.append("-" * 80)
    results.append("")
    results.append("【화순 지역 전략】")
    results.append("- 현황 분석:")
    results.append("  • 1등급 토양: 25.8% (적정), 집적도 높음 (AI=93.1)")
    results.append("  • 기반시설 수혜: 4.1% (매우 낮음), 극심한 파편화")
    results.append("  • 1등급 농업용지: 15.5%, 1등급 피복: 11.6%")
    results.append("")
    results.append("- 핵심 과제:")
    results.append("  1) 1등급 토양 보전 최우선 - 집단화 양호하므로 절대보전")
    results.append("  2) 기반시설 수혜 농지 집단화 시급 (812개 패치 통합)")
    results.append("  3) 1등급 토양 지역에 기반시설 우선 투자")
    results.append("")
    results.append("- 실행 전략:")
    results.append("  → 1등급 토양 452개 패치를 집단화하여 50개 이하로 통합")
    results.append("  → 기반시설 수혜 농지 교환·분합으로 평균 면적 10ha 이상 확보")
    results.append("  → 집단화 불가능한 소규모 분산 농지(< 3ha) 해제 검토")
    results.append("")
    results.append("【나주 지역 전략】")
    results.append("- 현황 분석:")
    results.append("  • 1등급 토양: 29.0% (우수), 749개 패치로 파편화")
    results.append("  • 기반시설 수혜: 19.8% (양호), 2,209개 패치로 심각한 파편화")
    results.append("  • 1등급 농업용지: 37.3% (매우 높음), 평균 63.29ha로 상대적 양호")
    results.append("")
    results.append("- 핵심 과제:")
    results.append("  1) 극심한 파편화 해소가 최우선")
    results.append("  2) 우량농지 비율이 높아 집단화 효과 극대화 가능")
    results.append("  3) 1등급 농업용지는 상대적으로 양호 → 이를 기준으로 집단화")
    results.append("")
    results.append("- 실행 전략:")
    results.append("  → 1등급 토양 749개 패치를 100개 이하로 대폭 통합")
    results.append("  → 기반시설 수혜 농지 2,209개 패치를 300개 이하로 집약")
    results.append("  → 1등급 농업용지 중심으로 집단화 핵심 구역 설정")
    results.append("  → 집단화 효과가 낮은 외곽 소규모 농지 해제 검토")
    results.append("")

    return "\n".join(results)

def main():
    """메인 함수"""
    print("FRAGSTATS 결과 해석 보고서 생성 중...")

    interpretation = generate_comprehensive_interpretation()

    output_file = work_dir / "FRAGSTATS_결과해석_농지기능평가.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(interpretation)

    print("")
    print("=" * 80)
    print("결과 해석 보고서 생성 완료!")
    print("=" * 80)
    print(f"출력 파일: {output_file}")
    print(f"총 {len(interpretation):,} 문자")
    print(f"총 {len(interpretation.split(chr(10))):,} 줄")
    print("")

if __name__ == "__main__":
    main()
