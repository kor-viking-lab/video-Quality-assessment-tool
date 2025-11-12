# OTT/DCI 품질평가 도구 v1.1.0

META(PQL) / NETFLIX(VMAF) 기준 종합 품질 분석 도구

## 개요

이 도구는 비디오 파일의 품질을 종합적으로 분석하여 OTT(Over-The-Top) 및 DCI(Digital Cinema Initiative) 표준 준수 여부를 확인하는 도구입니다.

## 주요 기능

- 비디오 메타데이터 분석 (해상도, 코덱, 비트레이트 등)
- 품질 메트릭 계산 (PSNR, SSIM, VMAF)
- 디노이즈 품질 평가 (노이즈 제거율, 디테일 보존율, 선명도)
- 색복원 품질 평가 (색상 정확도, 채도 복원율, 색조 차이)
- 업스케일링 품질 평가
- DCI/OTT 표준 준수 검사
- HTML 보고서 생성


## 지원 형식

`.mp4`, `.mov`, `.avi`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.ts`, `.mts`, `.mxf`

## 시스템 요구사항

- Python 3.8 이상
- Windows 10/11 또는 macOS 또는 Linux
- 최소 4GB RAM (8GB 권장)
- 충분한 디스크 공간 (임시 파일 생성)

## 설치 방법

### 1. 파이썬 환경 설정

**Conda 사용 (권장):**

```bash
# 새로운 환경 생성
conda create -n quality_env python=3.11

# 환경 활성화
conda activate quality_env
```

**또는 기존 Python 사용:**

```bash
# 가상환경 생성
python -m venv quality_env

# Windows에서 활성화
quality_env\Scripts\activate

# macOS/Linux에서 활성화
source quality_env/bin/activate
```

### 2. 시스템 의존성 설치

**FFmpeg 설치:**

```bash
# Conda로 설치 (권장)
conda install -c conda-forge ffmpeg

# Windows에서 winget 사용
winget install ffmpeg

# 또는 https://ffmpeg.org 에서 직접 다운로드
```

**MediaInfo 설치:**

```bash
# Conda로 설치 (권장)
conda install -c conda-forge mediainfo

# 또는 https://mediaarea.net/MediaInfo 에서 다운로드
```

### 3. Python 패키지 설치

```bash
# 필수 패키지 설치
pip install opencv-python numpy scipy scikit-image
pip install pymediainfo ffmpeg-python
pip install pathlib typing-extensions

# 보고서 생성용 패키지
pip install jinja2 matplotlib seaborn plotly

# VMAF 지원 (선택사항)
pip install vmaf
```

**또는 한번에 설치:**

```bash
# conda 환경에서
conda install -c conda-forge ffmpeg opencv mediainfo
pip install pymediainfo scikit-image numpy opencv-python ffmpeg-python matplotlib seaborn plotly
```

### 4. 설치 확인

```bash
# 시스템 도구 확인
ffmpeg -version
mediainfo --version

# Python 패키지 확인
python -c "import cv2, numpy, skimage, pymediainfo, ffmpeg; print('설치 완료!')"
```

## 프로젝트 구조

```
QualityComparison/
├── src/
│   ├── main.py                           # 메인 실행 파일
│   ├── analyzers/
│   │   ├── __init__.py
│   │   └── video_metadata_analyzer.py    # 메타데이터 분석기
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── quality_metrics.py           # 품질 메트릭 계산기
│   └── reports/
│       ├── __init__.py
│       ├── dci_ott_standards_checker.py # DCI/OTT 표준 검사기
│       └── html_report_generator.py      # HTML 보고서 생성기
├── vmaf_models/ (선택사항)
│   └── vmaf_v0.6.1.json                 # VMAF 모델 파일
└── README.md
```

## 사용 방법

### 기본 실행

```bash
# 프로젝트 폴더로 이동
cd D:\Development\python\QualityComparison

# 가상환경 활성화
conda activate quality_env

# 프로그램 실행
python src/main.py
```

### 작동 영상

https://github.com/user-attachments/assets/f7b8cc4e-755c-4cf7-a8b6-99fb7dba94f8


### 메뉴 옵션

프로그램 실행 후 다음 기능을 선택할 수 있습니다:

1. **비디오 메타데이터 분석** - 기본적인 파일 정보 추출
2. **DCI/OTT 표준 준수 검사** - 업계 표준 준수 여부 확인


## 품질 메트릭 해석

### PSNR (Peak Signal-to-Noise Ratio)

- **40dB 이상**: 매우 우수한 품질
- **30-40dB**: 우수한 품질
- **20-30dB**: 보통 품질
- **20dB 미만**: 낮은 품질

### SSIM (Structural Similarity Index)

- **0.95 이상**: 매우 유사함
- **0.8-0.95**: 유사함
- **0.6-0.8**: 보통 유사함
- **0.6 미만**: 유사하지 않음

### VMAF (Video Multi-method Assessment Fusion)

- **90 이상**: 매우 우수한 지각적 품질
- **70-90**: 우수한 지각적 품질
- **50-70**: 보통 지각적 품질
- **50 미만**: 낮은 지각적 품질

## DCI/OTT 표준

### 지원 표준

- **컨테이너**: MP4, MOV, IMF, MXF
- **코덱**: HEVC/H.265 (권장), H.264 (허용), ProRes, XAVC
- **해상도**: UHD 4K (3840x2160), DCI 4K (4096x2160)
- **비트 깊이**: 10-bit 이상 권장
- **색상 공간**: Rec. 2020 권장, Rec. 709 허용
- **프레임 레이트**: 24, 25, 30, 60fps
- **HDR 지원**: HDR10, Dolby Vision, HLG


## 출력 파일

분석 완료 후 다음 파일들이 생성됩니다:

- `{filename}_metadata_{timestamp}.json` - 메타데이터 분석 결과
- `{filename}_quality_analysis_{timestamp}.json` - 품질 분석 결과
- `{filename}_standards_check_{timestamp}.json` - 표준 준수 검사 결과
- `{filename}_comprehensive_{timestamp}.json` - 종합 분석 결과
- `{filename}_report.html` - HTML 보고서 (선택 시)

## 문제 해결

### 일반적인 오류

**Python 실행 경로 오류:**

```bash
# 해결방법 1: conda 환경 재생성
deactivate
conda create -n quality_env python=3.11
conda activate quality_env

# 해결방법 2: 직접 Python 경로 사용
quality_env\Scripts\python.exe src/main.py
```

**FFmpeg 오류:**

```bash
# FFmpeg 재설치
conda install -c conda-forge ffmpeg
```

**MediaInfo 오류:**

```bash
# MediaInfo 재설치
conda install -c conda-forge mediainfo
pip install --upgrade pymediainfo
```

**VMAF 계산 실패:**

- 해상도가 다른 비디오는 VMAF 비교 불가
- FFmpeg에 VMAF 필터가 포함되어야 함
- 처리 시간이 오래 걸릴 수 있음 (5분 제한)

### 성능 최적화

**메모리 사용량 줄이기:**

- 분석할 프레임 수를 줄임 (기본 20 → 10)
- 배치 처리 시 파일 수 제한

**처리 속도 향상:**

- SSD 사용 권장
- 메모리 8GB 이상 권장
- CPU 코어 수가 많을수록 유리

## 기술 스택

- **Python 3.8+** - 메인 언어
- **OpenCV** - 이미지/비디오 처리
- **FFmpeg** - 비디오 메타데이터 및 VMAF 계산
- **MediaInfo** - 상세 메타데이터 분석
- **scikit-image** - 이미지 품질 메트릭 (PSNR, SSIM)
- **Chart.js** - HTML 보고서 차트 생성

## 라이선스

이 프로젝트는 개발 및 연구 목적으로 제작되었습니다.

## 업데이트 내역

### v1.0.0 (2025-07-13)

- 초기 릴리스
- 기본적인 비디오 분석 기능
- DCI/OTT 표준 검사
- HTML 보고서 생성
- 배치 처리 지원

### v1.1.0 (2025-11-10)

- 디노이즈 품질 평가 기능 추가
  - 노이즈 제거율 계산
  - 디테일 보존율 측정
  - 선명도 점수 분석
- 색복원 품질 평가 기능 추가
  - 색상 정확도 (Delta E) 측정
  - 채도 복원율 계산
  - 색조 차이 분석
- 업스케일링 품질 평가 개선
- HTML 보고서 개선
  - 분석 타입별 맞춤형 보고서 생성
  - 디노이즈/색복원 시 DCI/OTT 표준 준수도 섹션 자동 숨김
  - 차트 및 시각화 향상

 
## 지원

문제가 발생하면 다음을 확인하세요:

- 모든 의존성이 올바르게 설치되었는지
- 비디오 파일이 지원되는 형식인지
- 파일 경로에 특수문자나 공백이 없는지
- 충분한 디스크 공간이 있는지
