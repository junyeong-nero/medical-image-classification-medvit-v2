# 뇌종양 분류 모델 (Brain Tumor Classifier)

## 📌 프로젝트 개요
이 프로젝트는 뇌 영상 데이터를 분석하여 뇌종양의 유무 및 종류를 식별하는 딥러닝 모델을 개발하는 것을 목표로 합니다. 최신 컴퓨터 비전 기술인 **Vision Transformer (ViT)**를 기반으로 하며, 분류 성능을 높이기 위해 FNN(Feed-Forward Network) 레이어를 결합한 아키텍처를 사용합니다.

## 📊 데이터셋 (Dataset)
이 프로젝트는 Hugging Face의 [PranomVignesh/MRI-Images-of-Brain-Tumor](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor) 데이터셋을 활용합니다.

- **입력 데이터 (Input)**: 뇌 MRI 이미지
- **출력 라벨 (Output Labels - 4 Classes)**:
  1. **glioma** (신경교종)
  2. **meningioma** (수막종)
  3. **pituitary** (뇌하수체 종양)
  4. **no-tumor** (종양 없음)

## 🏗 모델 아키텍처 (Model Architecture)
모델은 이미지 내의 전역적인 문맥 정보를 효과적으로 포착할 수 있는 Transformer 구조를 기반으로 설계되었습니다.

1. **Backbone**: Vision Transformer (ViT)
   - 이미지를 패치 단위로 분할하여 임베딩하고, Self-Attention 메커니즘을 통해 특징을 추출합니다.
2. **Classifier Head**: FNN (Feed-Forward Network)
   - ViT의 출력 임베딩을 입력으로 받아 최종 4개의 클래스 확률을 예측하는 Dense Layer로 구성됩니다.

## 🛠 개발 환경 및 요구사항 (Requirements)
이 프로젝트는 Python 3.13 이상 환경에서 동작합니다.

### 주요 라이브러리 (예정)
- `torch`, `torchvision` (Deep Learning)
- `transformers` (Hugging Face Models)
- `datasets` (Data Loading)
- `scikit-learn` (Evaluation)
- `numpy`, `pandas`, `matplotlib` (Data Processing & Visualization)

## 🚀 시작하기 (Getting Started)
*(추후 코드 구현에 따라 업데이트 예정)*

```bash
# 가상 환경 생성 및 의존성 설치 (예시)
uv sync  # 또는 pip install -r requirements.txt
python main.py
```
