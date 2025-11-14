#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
엔트로피 및 효용도 값 추출 스크립트
"""

import json
import pandas as pd

# JSON 파일 로드
with open('results/06_hierarchical_entropy_analysis/analysis_20251113_081818/05_intermediate_calculations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 가중치 파일 로드
weights_df = pd.read_csv('results/06_hierarchical_entropy_analysis/analysis_20251113_081818/01_group_indicator_weights.csv', index_col=0)

print("="*80)
print("그룹별 엔트로피, 효용도, 가중치 값")
print("="*80)

groups = {
    'area_density': '면적/밀도',
    'shape': '형태/경계',
    'core': '코어',
    'aggregation': '응집'
}

for group_id, group_name in groups.items():
    print(f"\n### {group_name} 그룹")
    print("-"*80)

    # 데이터 추출
    indicators = list(data[group_id]['entropy'].keys())
    results = []

    for ind in indicators:
        e = data[group_id]['entropy'][ind]
        d = data[group_id]['utility'][ind]
        w = weights_df.loc[ind, group_id]
        results.append({
            '지표': ind,
            'E': e,
            'D': d,
            'W': w
        })

    # 가중치 기준으로 정렬 (내림차순)
    results_df = pd.DataFrame(results).sort_values('W', ascending=False)

    print("\n| 지표 | 엔트로피(E) | 효용도(D) | 가중치(W) |")
    print("|:---|---:|---:|---:|")

    for _, row in results_df.iterrows():
        print(f"| {row['지표']:12s} | {row['E']:.4f} | {row['D']:.4f} | {row['W']:.4f} |")

    print(f"\n**합계:** W_sum = {results_df['W'].sum():.4f}")

    # 가장 중요한 지표와 가장 덜 중요한 지표
    most_important = results_df.iloc[0]
    least_important = results_df.iloc[-1]

    print(f"\n**핵심:**")
    print(f"- 가장 중요: {most_important['지표']} (E={most_important['E']:.4f}, W={most_important['W']*100:.2f}%)")
    print(f"- 가장 덜 중요: {least_important['지표']} (E={least_important['E']:.4f}, W={least_important['W']*100:.2f}%)")

print("\n" + "="*80)
print("완료!")
print("="*80)
