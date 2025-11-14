#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
레이어 가중치 검증 스크립트
원본 데이터로부터 최종 레이어 가중치를 재계산하여 검증
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("레이어 가중치 검증 시작")
print("="*80)

# 최신 분석 결과 로드
RESULT_DIR = Path("results/06_hierarchical_entropy_analysis/analysis_20251113_081818")

# 1. 최종 레이어 가중치 확인
print("\n1. 최종 레이어 가중치 (CSV 파일)")
print("-"*80)
layer_weights = pd.read_csv(RESULT_DIR / "04_final_layer_weights.csv", index_col=0)
print(layer_weights)

# 2. 변별력 상세 확인
print("\n2. 변별력 상세 (Discriminability)")
print("-"*80)
disc_details = pd.read_csv(RESULT_DIR / "03_discriminability_details.csv")
print(disc_details[['region', 'layer', 'discriminability_total']])

# 3. 수동 재계산
print("\n3. 수동 재계산으로 검증")
print("-"*80)

for region in ['hwasun', 'naju']:
    print(f"\n[{region.upper()}]")
    region_disc = disc_details[disc_details['region'] == region]

    # 변별력 값들
    disc_values = region_disc['discriminability_total'].values
    layers = region_disc['layer'].values

    # 합계
    disc_sum = np.sum(disc_values)

    # 가중치 = 변별력 / 합계
    weights = disc_values / disc_sum

    print(f"변별력 합계: {disc_sum:.6f}")
    print("\n레이어별 계산:")
    for i, layer in enumerate(layers):
        calc_weight = weights[i]
        csv_weight = layer_weights.loc[region, layer]
        diff = abs(calc_weight - csv_weight)

        print(f"  {layer:12s}: {disc_values[i]:.6f} / {disc_sum:.6f} = {calc_weight:.6f}")
        print(f"                CSV값: {csv_weight:.6f}, 차이: {diff:.10f}")

        if diff < 1e-9:
            print(f"                [OK] 일치")
        else:
            print(f"                [ERROR] 불일치!")

# 4. 최종 결론
print("\n" + "="*80)
print("4. 검증 결과 요약")
print("="*80)

print("\n[화순 지역 최종 가중치]")
hwasun_weights = layer_weights.loc['hwasun'].sort_values(ascending=False)
for rank, (layer, weight) in enumerate(hwasun_weights.items(), 1):
    print(f"  {rank}. {layer:12s}: {weight:.4f} ({weight*100:.2f}%)")

print("\n[나주 지역 최종 가중치]")
naju_weights = layer_weights.loc['naju'].sort_values(ascending=False)
for rank, (layer, weight) in enumerate(naju_weights.items(), 1):
    print(f"  {rank}. {layer:12s}: {weight:.4f} ({weight*100:.2f}%)")

print("\n" + "="*80)
print("검증 완료!")
print("="*80)

# 5. 기존 보고서와 비교
print("\n5. 기존 '종합_분석_보고서_최종.md'와의 비교")
print("="*80)

old_report_hwasun = {
    'pibok': 0.584,
    'infra': 0.243,
    'toyang': 0.136,
    'nongeup': 0.037
}

old_report_naju = {
    'pibok': 0.680,
    'nongeup': 0.176,
    'toyang': 0.062,
    'infra': 0.023
}

print("\n[화순 지역 비교]")
print(f"{'레이어':<12s} {'구보고서':<12s} {'신보고서':<12s} {'차이':<12s}")
print("-"*50)
for layer in ['infra', 'pibok', 'nongeup', 'toyang3']:
    old_layer = layer if layer != 'toyang3' else 'toyang'
    old_val = old_report_hwasun.get(old_layer, 0)
    new_val = layer_weights.loc['hwasun', layer]
    diff = new_val - old_val
    print(f"{layer:<12s} {old_val:<12.4f} {new_val:<12.4f} {diff:+12.4f}")

print("\n[나주 지역 비교]")
print(f"{'레이어':<12s} {'구보고서':<12s} {'신보고서':<12s} {'차이':<12s}")
print("-"*50)
for layer in ['infra', 'pibok', 'nongeup', 'toyang3']:
    old_layer = layer if layer != 'toyang3' else 'toyang'
    old_val = old_report_naju.get(old_layer, 0)
    new_val = layer_weights.loc['naju', layer]
    diff = new_val - old_val
    print(f"{layer:<12s} {old_val:<12.4f} {new_val:<12.4f} {diff:+12.4f}")

print("\n" + "="*80)
print("결론: FINAL_ANALYSIS_REPORT.md (2025-11-13 08:18:18)가 올바른 최신 결과입니다.")
print("      종합_분석_보고서_최종.md는 이전 분석 결과로 업데이트가 필요합니다.")
print("="*80)
