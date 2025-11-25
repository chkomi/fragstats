# 2. 지표 그룹핑 이유 및 그룹별 상세 정보

본 연구에서 사용된 30개의 FRAGSTATS 경관지표는 개념적 연관성에 따라 4개의 그룹으로 분류되었습니다. 이 그룹핑은 다음과 같은 두 가지 주요 목적을 가집니다.

1.  **체계적인 다차원 평가:** 농경지의 공간적 특성은 단일 지표로 설명하기 어렵습니다. 따라서 '규모', '형태', '내부 안정성', '연결성' 등 다차원적인 특성을 체계적으로 평가하기 위해 개념적으로 유사한 지표들을 하나의 그룹으로 묶었습니다. 이는 **Elkie et al. (1999)** 의 연구에서 제시된 경관지표 분류 체계를 기반으로 합니다.

2.  **통계적 안정성 확보:** 개념적으로 유사한 지표들은 통계적으로도 높은 상관관계를 보이는 경향이 있습니다(다중공선성 문제). 만약 모든 지표를 단일 계층에서 한 번에 분석할 경우, 특정 특성을 가진 지표 그룹이 결과에 과도한 영향을 미칠 수 있습니다. 이를 방지하고 분석의 안정성을 높이기 위해, 계층적 분석(Hierarchical Analysis)의 일부로 그룹핑을 수행했습니다.

## 4개 그룹의 구성 및 상세 정보

각 그룹은 농경지의 특정 공간적 특성을 평가하기 위해 설계되었으며, 포함된 지표는 다음과 같습니다.

### 1. 면적/밀도 그룹 (Area and Density Group)
- **목적:** 농경지 패치의 양, 면적, 분포 등 기본적인 '규모'와 관련된 특성을 평가합니다.
- **포함된 지표 (9개):**
  - `CA` (Class Area): 총 클래스 면적
  - `PLAND` (Percentage of Landscape): 경관 대비 비율
  - `NP` (Number of Patches): 패치 수
  - `PD` (Patch Density): 패치 밀도
  - `LPI` (Largest Patch Index): 가장 큰 패치의 비율
  - `AREA_MN` (Mean Patch Area): 평균 패치 면적
  - `AREA_AM` (Area-Weighted Mean Patch Area): 면적 가중 평균 패치 면적
  - `AREA_MD` (Median Patch Area): 패치 면적 중앙값
  - `AREA_CV` (Patch Area Coefficient of Variation): 패치 면적 변이 계수

### 2. 형태/경계 그룹 (Shape and Edge Group)
- **목적:** 패치 형태의 복잡성과 경계의 길이를 평가합니다. 단순한 형태일수록 기계화 영농에 효율적이므로 농업 생산성과 직접적인 관련이 있습니다.
- **포함된 지표 (7개):**
  - `TE` (Total Edge): 총 경계 길이
  - `ED` (Edge Density): 경계 밀도
  - `SHAPE_MN` (Mean Shape Index): 평균 형태 지수
  - `SHAPE_AM` (Area-Weighted Mean Shape Index): 면적 가중 평균 형태 지수
  - `FRAC_MN` (Mean Fractal Dimension): 평균 프랙탈 차원
  - `FRAC_AM` (Area-Weighted Mean Fractal Dimension): 면적 가중 평균 프랙탈 차원
  - `PARA_MN` (Mean Perimeter-Area Ratio): 평균 둘레-면적 비율

### 3. 코어 그룹 (Core Area Group)
- **목적:** 패치 경계의 영향을 받지 않는 안정적인 내부 '핵심 면적'의 크기와 분포를 평가합니다. 이는 실제 생산 활동이 이루어지는 '알짜' 면적의 질을 나타냅니다.
- **포함된 지표 (6개):**
  - `TCA` (Total Core Area): 총 핵심 면적
  - `CPLAND` (Core Area Percentage of Landscape): 경관 대비 핵심 면적 비율
  - `NDCA` (Number of Disjunct Core Areas): 분리된 핵심 지역 수
  - `DCAD` (Disjunct Core Area Density): 분리된 핵심 지역 밀도
  - `CORE_MN` (Mean Core Area per Patch): 평균 핵심 면적
  - `CAI_MN` (Mean Core Area Index): 평균 핵심 면적 지수

### 4. 응집/연결성 그룹 (Aggregation and Connectivity Group)
- **목적:** 패치들이 서로 얼마나 가깝게 모여 있는지(응집성)와 기능적으로 연결되어 있는지(연결성)를 평가합니다. 집단화 수준이 높을수록 공동 영농 및 자원 이동에 유리합니다.
- **포함된 지표 (8개):**
  - `GYRATE_MN` (Mean Radius of Gyration): 평균 회전 반경
  - `GYRATE_AM` (Area-Weighted Mean Radius of Gyration): 면적 가중 평균 회전 반경
  - `GYRATE_MD` (Median Radius of Gyration): 회전 반경 중앙값
  - `GYRATE_CV` (Radius of Gyration Coefficient of Variation): 회전 반경 변이 계수
  - `CLUMPY` (Clumpiness Index): 덩어리 지수
  - `PLADJ` (Percentage of Like Adjacencies): 유사 인접 비율
  - `AI` (Aggregation Index): 응집 지수
  - `COHESION` (Patch Cohesion Index): 패치 응집 지수
