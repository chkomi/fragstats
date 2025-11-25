"""
농지 잠재력 지수(API) 래스터를 4단계 등급으로 재분류하고 폴리곤으로 변환하여 면적 산출

작성일: 2025-01-13
목적:
1. API 래스터를 0.25 간격으로 4개 등급으로 재분류
2. 재분류된 래스터를 폴리곤으로 변환
3. 등급별 면적 계산 및 통계 산출
"""

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
from pathlib import Path
import json

def reclassify_api_raster(input_raster_path, output_raster_path):
    """
    API 래스터를 4단계 등급으로 재분류

    등급 구분:
    - 4등급 (저급): 0.00 <= API < 0.25 → 4
    - 3등급 (보통): 0.25 <= API < 0.50 → 3
    - 2등급 (우량): 0.50 <= API < 0.75 → 2
    - 1등급 (최우량): 0.75 <= API <= 1.00 → 1
    - NoData: API < 0 or API > 1 → NoData
    """
    print(f"래스터 재분류 시작: {input_raster_path}")

    with rasterio.open(input_raster_path) as src:
        # 래스터 데이터 읽기
        api_data = src.read(1)
        profile = src.profile.copy()

        # 재분류를 위한 빈 배열 생성
        reclassified = np.zeros_like(api_data, dtype=np.uint8)

        # NoData 마스크
        nodata = src.nodata
        if nodata is not None:
            nodata_mask = (api_data == nodata)
        else:
            nodata_mask = np.isnan(api_data)

        # 등급 분류
        # 4등급 (저급): 0.00 <= API < 0.25
        reclassified[(api_data >= 0.00) & (api_data < 0.25)] = 4

        # 3등급 (보통): 0.25 <= API < 0.50
        reclassified[(api_data >= 0.25) & (api_data < 0.50)] = 3

        # 2등급 (우량): 0.50 <= API < 0.75
        reclassified[(api_data >= 0.50) & (api_data < 0.75)] = 2

        # 1등급 (최우량): 0.75 <= API <= 1.00
        reclassified[(api_data >= 0.75) & (api_data <= 1.00)] = 1

        # NoData 처리
        reclassified[nodata_mask] = 0

        # 통계 출력
        unique, counts = np.unique(reclassified[~nodata_mask], return_counts=True)
        print("\n재분류 결과:")
        grade_names = {1: "1등급 (최우량)", 2: "2등급 (우량)", 3: "3등급 (보통)", 4: "4등급 (저급)"}
        for grade, count in zip(unique, counts):
            if grade > 0:
                print(f"  {grade_names.get(grade, f'등급 {grade}')}: {count:,} 픽셀")

        # 프로파일 업데이트
        profile.update(
            dtype=rasterio.uint8,
            nodata=0,
            compress='lzw'
        )

        # 재분류된 래스터 저장
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(reclassified, 1)

        print(f"재분류 완료: {output_raster_path}\n")

        return profile

def raster_to_polygon(raster_path, output_vector_path, region_name):
    """
    재분류된 래스터를 폴리곤으로 변환하고 면적 계산
    """
    print(f"폴리곤 변환 시작: {raster_path}")

    with rasterio.open(raster_path) as src:
        image = src.read(1)
        transform = src.transform
        crs = src.crs

        # 래스터를 폴리곤으로 변환
        # mask: 0(NoData)이 아닌 픽셀만 변환
        mask = image != 0

        results = []
        for geom, value in shapes(image, mask=mask, transform=transform):
            results.append({
                'geometry': shape(geom),
                'grade': int(value),
                'region': region_name
            })

    # GeoDataFrame 생성
    gdf = gpd.GeoDataFrame(results, crs=crs)

    # 면적 계산 (단위: m²)
    gdf['area_m2'] = gdf.geometry.area

    # 면적을 헥타르(ha)로 변환
    gdf['area_ha'] = gdf['area_m2'] / 10000

    # 등급별 통계
    grade_stats = gdf.groupby('grade').agg({
        'area_ha': ['sum', 'count']
    }).reset_index()
    grade_stats.columns = ['grade', 'total_area_ha', 'polygon_count']

    # 비율 계산
    total_area = grade_stats['total_area_ha'].sum()
    grade_stats['percentage'] = (grade_stats['total_area_ha'] / total_area * 100).round(2)

    # 등급 이름 추가
    grade_names = {
        1: "1등급 (최우량, 0.75~1.00)",
        2: "2등급 (우량, 0.50~0.74)",
        3: "3등급 (보통, 0.25~0.49)",
        4: "4등급 (저급, 0.00~0.24)"
    }
    grade_stats['grade_name'] = grade_stats['grade'].map(grade_names)

    # 등급 순서로 정렬
    grade_stats = grade_stats.sort_values('grade')

    print(f"\n{region_name} 등급별 면적 통계:")
    print("=" * 80)
    for _, row in grade_stats.iterrows():
        print(f"{row['grade_name']}")
        print(f"  면적: {row['total_area_ha']:,.2f} ha ({row['percentage']:.2f}%)")
        print(f"  폴리곤 수: {row['polygon_count']:,}개")
    print("=" * 80)
    print(f"전체 면적: {total_area:,.2f} ha\n")

    # Shapefile로 저장
    gdf.to_file(output_vector_path, driver='ESRI Shapefile', encoding='utf-8')
    print(f"폴리곤 저장 완료: {output_vector_path}")

    # 통계 CSV 저장
    stats_csv_path = output_vector_path.replace('.shp', '_stats.csv')
    grade_stats.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
    print(f"통계 저장 완료: {stats_csv_path}\n")

    return gdf, grade_stats

def compare_regions(hwasun_stats, naju_stats, output_path):
    """
    화순군과 나주시의 등급별 분포 비교
    """
    print("지역 간 비교 분석")
    print("=" * 80)

    # 통계 병합
    hwasun_stats = hwasun_stats.copy()
    naju_stats = naju_stats.copy()

    hwasun_stats['region'] = '화순군'
    naju_stats['region'] = '나주시'

    # 비교 테이블 생성
    comparison = pd.merge(
        hwasun_stats[['grade', 'grade_name', 'total_area_ha', 'percentage']],
        naju_stats[['grade', 'total_area_ha', 'percentage']],
        on='grade',
        suffixes=('_hwasun', '_naju')
    )

    comparison['diff_percentage'] = (comparison['percentage_hwasun'] -
                                      comparison['percentage_naju']).round(2)

    print("\n등급별 비율 비교:")
    print(comparison[['grade_name', 'percentage_hwasun', 'percentage_naju', 'diff_percentage']].to_string(index=False))
    print("=" * 80)

    # 비교 결과 저장
    comparison.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n비교 분석 결과 저장: {output_path}\n")

    return comparison

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 80)
    print("농지 잠재력 지수(API) 재분류 및 폴리곤 변환")
    print("=" * 80 + "\n")

    # 기본 경로 설정
    base_dir = Path(r"G:\내 드라이브\2. 개인자료\#논문\FRAGSTATS")

    # 입력 래스터 파일 경로 (사용자가 수정해야 함)
    # 가중치 계산으로 생성된 API 래스터 파일 경로를 지정하세요
    hwasun_api_raster = base_dir / "results" / "api_rasters" / "hwasun_api.tif"
    naju_api_raster = base_dir / "results" / "api_rasters" / "naju_api.tif"

    # 출력 디렉토리 생성
    output_dir = base_dir / "results" / "api_classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 출력 파일 경로
    hwasun_reclass_raster = output_dir / "hwasun_api_classified.tif"
    naju_reclass_raster = output_dir / "naju_api_classified.tif"

    hwasun_polygon = output_dir / "hwasun_api_polygon.shp"
    naju_polygon = output_dir / "naju_api_polygon.shp"

    comparison_csv = output_dir / "region_comparison.csv"

    # 화순군 처리
    print("\n[1/4] 화순군 래스터 재분류")
    print("-" * 80)
    if hwasun_api_raster.exists():
        reclassify_api_raster(hwasun_api_raster, hwasun_reclass_raster)

        print("\n[2/4] 화순군 폴리곤 변환 및 면적 계산")
        print("-" * 80)
        hwasun_gdf, hwasun_stats = raster_to_polygon(
            hwasun_reclass_raster,
            str(hwasun_polygon),
            "화순군"
        )
    else:
        print(f"경고: 화순군 API 래스터를 찾을 수 없습니다: {hwasun_api_raster}")
        print("스크립트 상단의 파일 경로를 확인하세요.\n")
        hwasun_stats = None

    # 나주시 처리
    print("\n[3/4] 나주시 래스터 재분류")
    print("-" * 80)
    if naju_api_raster.exists():
        reclassify_api_raster(naju_api_raster, naju_reclass_raster)

        print("\n[4/4] 나주시 폴리곤 변환 및 면적 계산")
        print("-" * 80)
        naju_gdf, naju_stats = raster_to_polygon(
            naju_reclass_raster,
            str(naju_polygon),
            "나주시"
        )
    else:
        print(f"경고: 나주시 API 래스터를 찾을 수 없습니다: {naju_api_raster}")
        print("스크립트 상단의 파일 경로를 확인하세요.\n")
        naju_stats = None

    # 지역 간 비교
    if hwasun_stats is not None and naju_stats is not None:
        print("\n[5/4] 지역 간 비교 분석")
        print("-" * 80)
        comparison = compare_regions(hwasun_stats, naju_stats, comparison_csv)

    print("\n" + "=" * 80)
    print("모든 처리 완료!")
    print("=" * 80)
    print(f"\n결과 파일 위치: {output_dir}")
    print("\n생성된 파일:")
    print(f"  1. 재분류 래스터: {hwasun_reclass_raster.name}, {naju_reclass_raster.name}")
    print(f"  2. 폴리곤 파일: {hwasun_polygon.name}, {naju_polygon.name}")
    print(f"  3. 통계 CSV: *_stats.csv")
    print(f"  4. 비교 분석: {comparison_csv.name}")
    print()

if __name__ == "__main__":
    main()
