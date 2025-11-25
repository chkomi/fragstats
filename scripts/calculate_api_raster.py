"""
4개 레이어 래스터를 가중치로 결합하여 농지 잠재력 지수(API) 래스터 생성

작성일: 2025-01-13
목적:
1. infra, pibok, nongeup, toyang 레이어 래스터 로드
2. 지역별 가중치 적용하여 API 계산
3. 결과 래스터 저장
"""

import numpy as np
import rasterio
from pathlib import Path

def calculate_api_raster(layer_paths, weights, output_path, region_name):
    """
    4개 레이어 래스터를 가중치로 결합하여 API 계산

    Parameters:
    -----------
    layer_paths : dict
        레이어별 래스터 파일 경로 {'infra': path, 'pibok': path, 'nongeup': path, 'toyang': path}
    weights : dict
        레이어별 가중치 {'infra': w1, 'pibok': w2, 'nongeup': w3, 'toyang': w4}
    output_path : str or Path
        출력 API 래스터 경로
    region_name : str
        지역명 (화순군/나주시)
    """
    print(f"\n{'='*80}")
    print(f"{region_name} API 래스터 계산")
    print(f"{'='*80}\n")

    # 가중치 출력
    print("적용 가중치:")
    for layer, weight in weights.items():
        print(f"  {layer}: {weight:.4f}")
    print()

    # 첫 번째 레이어로부터 메타데이터 가져오기
    first_layer = list(layer_paths.values())[0]
    with rasterio.open(first_layer) as src:
        profile = src.profile.copy()
        height, width = src.shape

    # API 계산을 위한 배열 초기화
    api_array = np.zeros((height, width), dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    # 각 레이어 로드 및 가중치 적용
    for layer_name, layer_path in layer_paths.items():
        if not Path(layer_path).exists():
            raise FileNotFoundError(f"{layer_name} 레이어를 찾을 수 없습니다: {layer_path}")

        with rasterio.open(layer_path) as src:
            data = src.read(1)
            nodata = src.nodata

            # NoData 처리
            if nodata is not None:
                mask = (data != nodata)
            else:
                mask = ~np.isnan(data)

            # cls_1은 1, cls_9는 0으로 변환 (이진 분류 가정)
            # 실제 데이터 형식에 맞게 조정 필요
            # 예: cls_1 = 1, cls_9 = 9 인 경우
            binary_data = np.where(data == 1, 1.0, 0.0)

            # 가중치 적용
            weighted_data = binary_data * weights[layer_name]

            # API에 더하기
            api_array += weighted_data

            # 유효한 데이터 마스크 업데이트
            valid_mask |= mask

        print(f"  {layer_name} 레이어 처리 완료")

    # 유효하지 않은 픽셀은 NoData로 설정
    api_array[~valid_mask] = -9999

    # 통계 출력
    valid_api = api_array[valid_mask]
    print(f"\nAPI 통계:")
    print(f"  최소값: {valid_api.min():.4f}")
    print(f"  최대값: {valid_api.max():.4f}")
    print(f"  평균값: {valid_api.mean():.4f}")
    print(f"  중앙값: {np.median(valid_api):.4f}")
    print(f"  표준편차: {valid_api.std():.4f}")

    # 등급별 분포
    grade_1 = np.sum((valid_api >= 0.75) & (valid_api <= 1.00))
    grade_2 = np.sum((valid_api >= 0.50) & (valid_api < 0.75))
    grade_3 = np.sum((valid_api >= 0.25) & (valid_api < 0.50))
    grade_4 = np.sum((valid_api >= 0.00) & (valid_api < 0.25))
    total_pixels = len(valid_api)

    print(f"\n등급별 픽셀 분포:")
    print(f"  1등급 (0.75~1.00): {grade_1:,} ({grade_1/total_pixels*100:.2f}%)")
    print(f"  2등급 (0.50~0.74): {grade_2:,} ({grade_2/total_pixels*100:.2f}%)")
    print(f"  3등급 (0.25~0.49): {grade_3:,} ({grade_3/total_pixels*100:.2f}%)")
    print(f"  4등급 (0.00~0.24): {grade_4:,} ({grade_4/total_pixels*100:.2f}%)")

    # 프로파일 업데이트
    profile.update(
        dtype=rasterio.float32,
        nodata=-9999,
        compress='lzw'
    )

    # API 래스터 저장
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(api_array, 1)

    print(f"\nAPI 래스터 저장 완료: {output_path}")
    print(f"{'='*80}\n")

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 80)
    print("농지 잠재력 지수(API) 래스터 계산")
    print("=" * 80 + "\n")

    # 기본 경로 설정
    base_dir = Path(r"G:\내 드라이브\2. 개인자료\#논문\FRAGSTATS")

    # 입력 레이어 래스터 경로 (사용자가 수정해야 함)
    # 각 레이어는 cls_1(적합)과 cls_9(부적합)으로 분류된 래스터여야 함
    # cls_1 = 1, cls_9 = 9 (또는 0) 형식을 가정

    # 화순군 레이어 경로
    hwasun_layers = {
        'infra': base_dir / "data" / "layers" / "hwasun" / "infra_cls.tif",
        'pibok': base_dir / "data" / "layers" / "hwasun" / "pibok_cls.tif",
        'nongeup': base_dir / "data" / "layers" / "hwasun" / "nongeup_cls.tif",
        'toyang': base_dir / "data" / "layers" / "hwasun" / "toyang_cls.tif"
    }

    # 나주시 레이어 경로
    naju_layers = {
        'infra': base_dir / "data" / "layers" / "naju" / "infra_cls.tif",
        'pibok': base_dir / "data" / "layers" / "naju" / "pibok_cls.tif",
        'nongeup': base_dir / "data" / "layers" / "naju" / "nongeup_cls.tif",
        'toyang': base_dir / "data" / "layers" / "naju" / "toyang_cls.tif"
    }

    # 화순군 가중치 (4장 결과)
    hwasun_weights = {
        'infra': 0.1710,
        'pibok': 0.2336,
        'nongeup': 0.4153,
        'toyang': 0.1801
    }

    # 나주시 가중치 (4장 결과)
    naju_weights = {
        'infra': 0.3816,
        'pibok': 0.3855,
        'nongeup': 0.1598,
        'toyang': 0.0731
    }

    # 출력 디렉토리 생성
    output_dir = base_dir / "results" / "api_rasters"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 출력 파일 경로
    hwasun_api_output = output_dir / "hwasun_api.tif"
    naju_api_output = output_dir / "naju_api.tif"

    try:
        # 화순군 API 계산
        print("\n[1/2] 화순군 API 래스터 계산")
        calculate_api_raster(
            layer_paths={k: str(v) for k, v in hwasun_layers.items()},
            weights=hwasun_weights,
            output_path=str(hwasun_api_output),
            region_name="화순군"
        )

        # 나주시 API 계산
        print("\n[2/2] 나주시 API 래스터 계산")
        calculate_api_raster(
            layer_paths={k: str(v) for k, v in naju_layers.items()},
            weights=naju_weights,
            output_path=str(naju_api_output),
            region_name="나주시"
        )

        print("\n" + "=" * 80)
        print("모든 API 래스터 계산 완료!")
        print("=" * 80)
        print(f"\n결과 파일 위치: {output_dir}")
        print(f"  - {hwasun_api_output.name}")
        print(f"  - {naju_api_output.name}")
        print("\n다음 단계: reclassify_and_vectorize.py를 실행하여 등급 분류 및 폴리곤 변환")
        print()

    except FileNotFoundError as e:
        print(f"\n오류: {e}")
        print("\n스크립트 상단의 레이어 파일 경로를 확인하세요.")
        print("각 레이어는 cls_1(적합)과 cls_9(부적합)으로 분류된 래스터여야 합니다.")
        print()

if __name__ == "__main__":
    main()
