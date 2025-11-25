# 농지 잠재력 지수(API) 계산 및 폴리곤 변환 가이드

## 개요

4개 평가 레이어(infra, pibok, nongeup, toyang)를 가중치로 결합하여 농지 잠재력 지수(API)를 계산하고, 0.25 간격으로 4개 등급으로 재분류한 후 폴리곤으로 변환하여 면적을 산출하는 작업 흐름입니다.

## 작업 흐름

```
[레이어 래스터] → [API 계산] → [등급 재분류] → [폴리곤 변환] → [면적 산출]
```

## 필요한 패키지

```bash
pip install rasterio geopandas shapely numpy pandas
```

## 1단계: 레이어 데이터 준비

### 입력 데이터 형식

각 레이어는 다음 형식의 래스터 파일이어야 합니다:
- **cls_1 (적합)**: 픽셀 값 = 1
- **cls_9 (부적합)**: 픽셀 값 = 0 또는 9
- **NoData**: -9999 또는 NaN

### 디렉토리 구조 예시

```
G:\내 드라이브\2. 개인자료\#논문\FRAGSTATS\
├── data\
│   └── layers\
│       ├── hwasun\          # 화순군 레이어
│       │   ├── infra_cls.tif
│       │   ├── pibok_cls.tif
│       │   ├── nongeup_cls.tif
│       │   └── toyang_cls.tif
│       └── naju\            # 나주시 레이어
│           ├── infra_cls.tif
│           ├── pibok_cls.tif
│           ├── nongeup_cls.tif
│           └── toyang_cls.tif
├── results\
│   ├── api_rasters\         # API 래스터 출력
│   └── api_classification\   # 분류 및 폴리곤 출력
└── scripts\
    ├── calculate_api_raster.py
    ├── reclassify_and_vectorize.py
    └── README_API_처리.md
```

## 2단계: API 래스터 계산

### 스크립트: `calculate_api_raster.py`

이 스크립트는 4개 레이어 래스터를 가중치로 결합하여 API 래스터를 생성합니다.

### 가중치 (4장 계층적 엔트로피 분석 결과)

**화순군:**
```python
hwasun_weights = {
    'infra': 0.1710,
    'pibok': 0.2336,
    'nongeup': 0.4153,  # 논 적성등급 (가장 중요)
    'toyang': 0.1801
}
```

**나주시:**
```python
naju_weights = {
    'infra': 0.3816,    # 기반시설 (가장 중요)
    'pibok': 0.3855,    # 피복현황 (가장 중요)
    'nongeup': 0.1598,
    'toyang': 0.0731
}
```

### 실행 방법

1. 스크립트를 열고 레이어 파일 경로 수정:

```python
# 화순군 레이어 경로
hwasun_layers = {
    'infra': base_dir / "data" / "layers" / "hwasun" / "infra_cls.tif",
    'pibok': base_dir / "data" / "layers" / "hwasun" / "pibok_cls.tif",
    'nongeup': base_dir / "data" / "layers" / "hwasun" / "nongeup_cls.tif",
    'toyang': base_dir / "data" / "layers" / "hwasun" / "toyang_cls.tif"
}
```

2. 실행:

```bash
python calculate_api_raster.py
```

### 출력

- `results/api_rasters/hwasun_api.tif` - 화순군 API 래스터
- `results/api_rasters/naju_api.tif` - 나주시 API 래스터

각 픽셀의 값은 0.0~1.0 범위의 농지 잠재력 지수입니다.

## 3단계: 등급 재분류 및 폴리곤 변환

### 스크립트: `reclassify_and_vectorize.py`

이 스크립트는 API 래스터를 4개 등급으로 재분류하고 폴리곤으로 변환하여 면적을 계산합니다.

### 등급 분류 기준

| 등급 | API 범위 | 명칭 | 픽셀 값 |
|------|----------|------|---------|
| 1등급 | 0.75~1.00 | 최우량 농지 | 1 |
| 2등급 | 0.50~0.74 | 우량 농지 | 2 |
| 3등급 | 0.25~0.49 | 보통 농지 | 3 |
| 4등급 | 0.00~0.24 | 저급 농지 | 4 |

### 실행 방법

1. 스크립트를 열고 API 래스터 경로 확인/수정:

```python
hwasun_api_raster = base_dir / "results" / "api_rasters" / "hwasun_api.tif"
naju_api_raster = base_dir / "results" / "api_rasters" / "naju_api.tif"
```

2. 실행:

```bash
python reclassify_and_vectorize.py
```

### 출력

`results/api_classification/` 디렉토리에 다음 파일이 생성됩니다:

**1. 재분류된 래스터**
- `hwasun_api_classified.tif` - 화순군 등급 분류 래스터
- `naju_api_classified.tif` - 나주시 등급 분류 래스터

**2. 폴리곤 Shapefile**
- `hwasun_api_polygon.shp` (및 관련 파일) - 화순군 등급별 폴리곤
- `naju_api_polygon.shp` (및 관련 파일) - 나주시 등급별 폴리곤

각 폴리곤은 다음 속성을 가집니다:
- `grade`: 등급 (1, 2, 3, 4)
- `region`: 지역명 (화순군/나주시)
- `area_m2`: 면적 (m²)
- `area_ha`: 면적 (ha)

**3. 통계 CSV**
- `hwasun_api_polygon_stats.csv` - 화순군 등급별 면적 통계
- `naju_api_polygon_stats.csv` - 나주시 등급별 면적 통계
- `region_comparison.csv` - 지역 간 비교 분석

### 통계 CSV 형식

```csv
grade,grade_name,total_area_ha,polygon_count,percentage
1,"1등급 (최우량, 0.75~1.00)",1234.56,150,15.5
2,"2등급 (우량, 0.50~0.74)",3456.78,320,43.2
...
```

## 결과 활용

### 1. 논문 표 작성

생성된 통계 CSV 파일을 사용하여 논문 본문의 표 5-6, 5-7, 5-8을 작성할 수 있습니다:

- **표 5-6**: `hwasun_api_polygon_stats.csv` → 화순군 등급별 분포
- **표 5-7**: `naju_api_polygon_stats.csv` → 나주시 등급별 분포
- **표 5-8**: `region_comparison.csv` → 지역 간 비교

### 2. 지도 시각화

생성된 Shapefile을 QGIS, ArcGIS 등에서 열어 등급별 공간 분포를 시각화할 수 있습니다:

**QGIS에서 시각화:**
1. QGIS에서 `hwasun_api_polygon.shp` 열기
2. 레이어 속성 → 심볼로지 → 분류됨(Categorized)
3. 값(Value): `grade` 선택
4. 색상표(Color ramp) 선택 (예: RdYlGn - 빨강-노랑-초록)
5. 분류(Classify) 클릭

**색상 추천:**
- 1등급: 진한 초록 (#006400)
- 2등급: 연한 초록 (#90EE90)
- 3등급: 노랑 (#FFFF00)
- 4등급: 빨강 (#FF0000)

### 3. 농업진흥지역 정합성 분석

생성된 폴리곤을 농업진흥지역 경계와 중첩 분석하여 정합성을 평가할 수 있습니다 (5장 다. 섹션).

## 문제 해결

### 오류: "파일을 찾을 수 없습니다"

→ 스크립트 상단의 파일 경로를 실제 데이터 위치에 맞게 수정하세요.

### 오류: "NoData 처리 오류"

→ 각 레이어 래스터의 NoData 값을 확인하고, 스크립트의 NoData 처리 부분을 수정하세요.

### 오류: "좌표계 불일치"

→ 모든 레이어 래스터가 동일한 좌표계(CRS)를 사용하는지 확인하세요.

### 결과 검증

1. **가중치 합 확인**: 각 지역의 가중치 합이 1.0인지 확인
2. **API 범위 확인**: API 값이 0.0~1.0 범위 내에 있는지 확인
3. **면적 합계 확인**: 4개 등급의 면적 합계가 전체 면적과 일치하는지 확인

## 참고

- 제4장: 계층적 엔트로피 분석 (가중치 산출)
- 제5장 가.: 평가지표 통합 및 농지 잠재력 지수 산출
- 제5장 나.: 나주시·화순군 농지 종합 등급화 및 결과 분석
- 제5장 다.: 농업진흥지역과의 정합성 분석
