# CAM-Back-Again
📝 Overview

이 저장소는 CVPR 2024 논문
“CAM Back Again: Large Kernel CNNs from a Weakly Supervised Localization Perspective”
의 핵심 실험을 재현(Reproduction)하고,

논문의 주장 —“대형 커널 CNN의 WSOL 성능은 ERF가 아니라 Feature Map Quality 때문이다.”— 가 실제로 성립하는지를 검증하는 프로젝트입니다.

본 저장소에서는 다음을 수행합니다:

RepLKNet / ConvNeXt의 CAM 기반 WSOL 성능(MaxBoxAcc) 재현

Gradient 기반 ERF(Effective Receptive Field) 계산 코드 직접 구현

ERF vs WSOL 관계 분석 (산점도 + 회귀 분석)

SLaK의 strip-conv 구조 기반 십자형 ERF 모양 보정(backbone 재구현)

다양한 파라미터 실험 및 자동 수집 파이프라인 구축

본 리포지토리는 CAM-Back-Again 논문 코드를 기반으로 한 재현/확장 버전입니다.
재현을 진행하기 전에, 먼저 아래 원본 저장소를 클론한 뒤 본 리포지토리의 스크립트와 코드를 추가·수정하여 사용합니다.

git clone https://github.com/snskysk/CAM-Back-Again.git
cd CAM-Back-Again
# 이후 이 리포지토리에서 제공하는 코드와 스크립트를 덮어쓰기/추가

⚙️ Environment Setup
conda create -n cam_repro python=3.10 -y
conda activate cam_repro

pip install torch torchvision timm tqdm numpy matplotlib pandas


Dataset 준비:

wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz
cp -r CUB_200_2011/images datasets/cub-200-2011

🚀 1. WSOL Heatmap 생성
RepLKNet
python generate_heatmap.py \
  --model_family replknet \
  --fine_tuned_weight_name weights/replknet_31B1K384_CUB.pth \
  --test_dataset cub-200-2011 \
  --heatmap_output heatmap_replk_cam \
  --localization_method cam

ConvNeXt
python generate_heatmap.py \
  --model_family convnext \
  --fine_tuned_weight_name weights/convnext_base_384_in22ft1k_CUB.pth \
  --test_dataset cub-200-2011 \
  --heatmap_output heatmap_convnext_cam \
  --localization_method cam

🚀 2. WSOL 성능 평가 (MaxBoxAcc)
RepLKNet
python wsol_eval.py \
  --np_root np_heatmap_replk_cam \
  --cub_root datasets/CUB_200_2011 \
  --iou_thr 0.5

ConvNeXt
python wsol_eval.py \
  --np_root np_heatmap_convnext_cam \
  --cub_root datasets/CUB_200_2011 \
  --iou_thr 0.5

✔️ Reproduced Results
Model	MaxBoxAcc (%)
RepLKNet-31B	89.68%
ConvNeXt-B	74.43%

➡️ RepLKNet이 ConvNeXt보다 15%p 이상 우수, 논문 패턴과 동일 재현.

➡️ RepLKNet optimal threshold = 0.15
➡️ ConvNeXt optimal threshold = 0.45

→ RepLKNet의 CAM이 전역적으로 객체 전체를 활성화한다는 증거.

🚀 3. ERF 계산 (Gradient-based)

코드(erf_compute.py)는 입력 gradient로부터 saliency map을 만들고,

전체 gradient 에너지의 상위 20%를 차지하는 최소 픽셀 수를 ERF 크기로 정의합니다.

실행 예시

python erf_compute.py \
  --model_family replknet \
  --fine_tuned_weight_name weights/replknet_31B1K384_CUB.pth \
  --test_dataset cub-200-2011/images \
  --num_samples 200 \
  --output erf_sizes_replknet.npy

python erf_compute.py \
  --model_family convnext \
  --fine_tuned_weight_name weights/convnext_base_384_in22ft1k_CUB.pth \
  --test_dataset cub-200-2011/images \
  --num_samples 200 \
  --output erf_sizes_convnext.npy

✔️ ERF Result Summary
Model	ERF Mean	ERF Max
RepLKNet	5611	8700
ConvNeXt	5032	9232

➡️ ConvNeXt가 더 큰 ERF를 갖는 경우가 있으나 WSOL은 훨씬 낮음
→ ERF 크기 ≠ WSOL 성능

➡️ ERF 분포도 매우 넓어 설명력이 부족함.

🚀 4. ERF vs WSOL 산점도 분석
python analysis_erf_vs_wsol.py \
  --config erf_wsol_config.json \
  --save_fig erf_vs_wsol.png \
  --show


결과:

ERF mean ↔ WSOL 상관계수 ≈ 0

R² ≈ 0

ERF 크기는 WSOL 성능을 설명하지 못함

➡️ Feature map quality가 진짜 원인임을 재현 실험이 뒷받침.

train_wsol.py 예시 실행
# ConvNeXt, 384, 100 epoch, light aug
python train_wsol.py \
  --model_family convnext \
  --epochs 50 \
  --lr 1e-4 \
  --input_size 384 \
  --aug_mode light \
  --exp_name conv_r384_e100_lr1e4_light

# RepLKNet, 384, 100 epoch, light aug
python train_wsol.py \
  --model_family replknet \
  --epochs 50 \
  --lr 5e-5 \
  --input_size 384 \
  --aug_mode light \
  --exp_name replk_r384_e100_lr5e5_light
