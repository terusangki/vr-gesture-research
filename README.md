# vr-gesture-research
Deep learning-based gesture recognition for VR rhythm game interface

# VR Gesture Recognition — VideoPrism + Soft-DTW

> 딥러닝 기반 제스처 인식 기술을 활용한 VR 리듬게임 인터페이스 개발  
> 중앙대학교 AI·SW융합 학부연구생 프로그램 - 손봉수 교수 가상현실연구실

---

## 연구 개요

기존 VR 리듬게임은 핸드헬드 컨트롤러에 의존합니다. 본 연구는 **맨손 제스처만으로 VR을 조작**할 수 있도록, 딥러닝 기반 영상 분류 모델에 시계열 특화 손실 함수인 **Soft-DTW**를 통합하여 8방향 탭핑 제스처 인식 성능을 검증했습니다.

| 항목 | 내용 |
|------|------|
| 모델 | VideoPrism-Base (Google, JAX) · TimeSformer (Meta, PyTorch) |
| 데이터셋 | Kinetics-400 (40,000개) · Something-Something V2 (168,913개) · 자체 제스처 영상 |
| 인프라 | KT Cloud NVIDIA A100 · Google Cloud TPU v4-8 |
| 핵심 기법 | Soft-DTW Loss · Label Smoothing · Frozen Backbone Fine-tuning |

---

## 주요 결과

### K400 4-case 비교 (5,000 steps)

| 실험 | 손실 함수 | Accuracy |
|------|-----------|----------|
| Exp 0 | Cross Entropy (Baseline) | 64.00% |
| Exp 1 | + Label Smoothing | 66.12% |
| Exp 2 | + SoftDTW | 64.38% |
| **Exp 3** | **+ Label Smoothing + SoftDTW** | **66.88% ↑** |

### 시간 방향성 검증 (SoftDTW 효과 집중 검증)

| 실험 | Baseline | SoftDTW 적용 | 개선 |
|------|----------|--------------|------|
| 합성 시퀀스 (증가 vs. 감소) | 47% | **100%** | **+53%p** |
| K400 역재생 (원본 vs. 뒤집기) | 10% | **40%** | **+30%p** |

> **핵심 발견**: SoftDTW는 시간 순서가 분류의 핵심인 과제에서만 효과가 발현된다.  
> K400(객체 중심)에서는 효과가 미미하지만, 역재생·합성 시퀀스처럼 시간 방향이 유일한 단서일 때 극적인 성능 향상을 보인다.

---

## 핵심 구현

### JAX ↔ PyTorch 브릿지
VideoPrism(JAX)에서 추출한 특징 벡터를 Soft-DTW(PyTorch)에 넘기기 위한 브릿지 코드를 직접 작성했습니다.

```python
# JAX array → NumPy → PyTorch Tensor 변환
features_np = np.array(jax_features)           # JAX → NumPy
features_pt = torch.from_numpy(features_np)    # NumPy → PyTorch
```

### CombinedLoss — 4-case 자동 전환
`EXP_MODE` 변수 하나로 4가지 손실 함수 조합을 자동으로 전환하는 클래스입니다.

```python
class CombinedLoss(nn.Module):
    def __init__(self, mode, num_classes):
        super().__init__()
        # Mode 0: CrossEntropy
        # Mode 1: CrossEntropy + Label Smoothing
        # Mode 2: CrossEntropy + SoftDTW
        # Mode 3: CrossEntropy + Label Smoothing + SoftDTW
        ls_factor = 0.1 if mode in [1, 3] else 0.0
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=ls_factor)
        self.sdtw = SoftDTW(gamma=0.1) if mode in [2, 3] else None
```

### Frozen Backbone Fine-tuning
VideoPrism 전체 재학습 대신 가중치를 고정하고 뒷단의 Linear Classifier만 학습했습니다. CPU 환경에서도 수 시간 내 학습 가능한 경량 파인튜닝 구조입니다.

---

## 레포 구조

```
vr-gesture-research/
├── README.md
├── experiments/
│   ├── videoprism/
│   │   ├── real_video_text.py     # 베이스라인 (특징 평균 매칭)
│   │   ├── softdtw_test.py        # Soft-DTW 시간 궤적 비교 (핵심)
│   │   └── ssv2_finetune.py       # SSv2 파인튜닝
│   └── timesformer/
│       ├── train_net.py           # CombinedLoss 포함 학습 스크립트
│       ├── defaults.py            # EXP_MODE 설정
│       └── softdtw_loss.py        # SoftDTW nn.Module
├── reports/
│   ├── 01_ktcloud_test_report.pdf
│   ├── 02_timesformer_k400_report.pdf
│   ├── 03_videoprism_softdtw_report.pdf
│   └── 04_final_report.pdf
├── results/
│   ├── k400_results.md
│   └── ssv2_results.md
└── assets/
    └── gesture_capture.png
```

---

## 트러블슈팅 기록

실제 연구 과정에서 해결한 주요 엔지니어링 문제들입니다.

| 문제 | 원인 | 해결 |
|------|------|------|
| JAX가 A100 GPU를 못 찾음 | CUDA 드라이버 버전 ↔ JAX 버전 불일치 | Anaconda 독립 가상환경 구축, 경로 수동 설정 |
| TimeSformer 실행 오류 | 2021 구버전 코드의 deprecated 모듈 | `_LinearWithBias`, `torch._six` 등 최신 문법으로 수정 |
| K400 다운로드 봇 차단 | 데이터센터 IP 대량 요청 차단 | PyAV 검증 스크립트로 유효 영상 필터링 |
| Colab 메모리 부족 | VideoPrism 모델 용량 | KT Cloud A100 → Google TPU로 이관 |

---

## 인사이트

1. **SoftDTW는 데이터셋 특성에 따라 효과가 달라진다** — 객체 중심(K400)에서는 미미하지만, 시간 순서가 핵심인 과제(역재생, SSv2)에서는 유의미한 성능 향상을 보인다.
2. **학습량이 성능 평가의 전제** — 1,000 steps(47%) vs 5,000 steps(64%), 같은 모델이라도 학습량이 결과를 크게 좌우한다.
3. **이종 프레임워크 연동** — JAX와 PyTorch 사이의 브릿지 코드를 작성하며 딥러닝 프레임워크 내부 구조를 이해하는 계기가 됐다.

---

## 실험 환경

```
Python     3.11
JAX        0.4.x
PyTorch    2.x
CUDA       12.x
Hardware   KT Cloud NVIDIA A100 40GB / Google Cloud TPU v4-8
Dataset    Kinetics-400 (40K, 66GB) / Something-Something V2 (168K, 19GB)
```
