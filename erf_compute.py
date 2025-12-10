# coding: utf-8
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import timm

from convnext_func import Net2Head
from replknet_func import build_model


def build_model_for_erf(model_family, fine_tuned_weight_name, device):
    """
    generate_heatmap.py 와 동일한 방식으로 모델을 구성하고
    fine-tuned weight 를 로드한다.
    (DataParallel 은 사용하지 않음 - gradient 계산 단순화를 위해)
    """
    model_config = {
        "class_n": 200,
        "unit_n": 1024,
        "input_size": 384,
        "size": 12,
        "lr": 1e-5,
        "weight_decay": 5e-4,
    }

    if model_family == "convnext":
        model_config["model_name"] = "convnext_base_384_in22ft1k"
        base = timm.create_model(model_config["model_name"], pretrained=False)
        base = nn.Sequential(*list(base.children())[:-2])
        model = nn.Sequential(
            base,
            Net2Head(
                model_config["class_n"],
                model_config["unit_n"],
                model_config["size"],
            ),
        )
    elif model_family == "replknet":
        model_config["model_name"] = "RepLKNet-31B"
        model_config["channels"] = [128, 256, 512, 1024]
        model = build_model(model_config)
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    # weight 로드
    state = torch.load(fine_tuned_weight_name, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, model_config


def build_dataset_for_erf(test_dataset, input_size=384):
    """
    generate_heatmap.py 와 동일한 transform 사용.
    datasets/{test_dataset}/ 를 ImageFolder 로 읽는다.
    """
    current_path = os.getcwd() + "/"
    test_dir = current_path + f"datasets/{test_dataset}/"

    img_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = datasets.ImageFolder(test_dir, transform=img_transform)
    return dataset


def compute_erf_for_image(model, x, device="cuda", p=0.2):
    """
    하나의 이미지 텐서 x (1 x 3 x H x W)에 대해,
    top-1 클래스 score에 대한 gradient 로 ERF 를 계산.

    p: 전체 gradient "에너지" (합) 중 상위 p 비율을 차지하는
       최소 픽셀 개수를 ERF 크기로 사용.
       (ex. p=0.2 이면, gradient 값이 큰 픽셀들부터 더해가면서
        전체 합의 20%를 넘길 때까지 필요한 픽셀 수)
    """
    model.eval()
    x = x.to(device)
    x.requires_grad_(True)

    # forward
    output = model(x)  # (1, C) 혹은 tuple
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    probs = F.softmax(logits, dim=1)
    target_class = probs.argmax(dim=1)[0]
    score = probs[0, target_class]

    # backward
    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()
    score.backward()

    # 입력에 대한 gradient: 1 x 3 x H x W
    grad = x.grad.detach()
    # 채널 방향 평균, 절댓값 취해서 saliency map 생성: H x W
    grad = grad.abs().mean(dim=1)[0]  # (H, W)

    # flatten
    flat = grad.view(-1)  # (H*W,)

    # 값이 큰 순으로 정렬
    values, _ = torch.sort(flat, descending=True)

    # 누적합 기준으로 상위 p 에너지에 해당하는 최소 픽셀 수 계산
    cumsum = torch.cumsum(values, dim=0)
    total = cumsum[-1]
    if total <= 0:
        # gradient 가 모두 0 인 극단적인 경우: ERF 크기를 0 또는 전체로 처리
        erf_size = 0
    else:
        target_energy = p * total
        # 누적합이 target_energy 를 처음 초과하는 index
        idx = torch.searchsorted(cumsum, target_energy)
        erf_size = int(idx.item()) + 1  # index 는 0-based 이므로 +1

    return erf_size, grad.cpu().numpy()


def compute_erf_for_model(
    model_family,
    fine_tuned_weight_name,
    test_dataset,
    num_samples=200,
    p=0.2,
    seed=0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_config = build_model_for_erf(
        model_family, fine_tuned_weight_name, device
    )
    dataset = build_dataset_for_erf(
        test_dataset, input_size=model_config["input_size"]
    )

    # 샘플링
    rng = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    erf_sizes = []

    print(
        f"[ERF] model_family={model_family}, "
        f"num_samples={len(indices)}, p={p}"
    )

    for idx in tqdm(indices):
        x, _ = dataset[idx]
        x = x.unsqueeze(0)  # 1 x C x H x W
        size, _ = compute_erf_for_image(model, x, device=device, p=p)
        erf_sizes.append(size)

    erf_sizes = np.array(erf_sizes, dtype=np.float32)
    print(f"[ERF] mean={erf_sizes.mean():.2f}, std={erf_sizes.std():.2f}")
    return erf_sizes


def main():
    parser = argparse.ArgumentParser("ERF computation")
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        help="convnext or replknet",
    )
    parser.add_argument(
        "--fine_tuned_weight_name",
        type=str,
        required=True,
        help="weight 파일 경로 (예: weights/replknet_31B1K384_CUB.pth)",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="cub-200-2011",
        help="datasets/ 하위의 테스트 데이터셋 디렉토리 이름",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="ERF 계산에 사용할 이미지 개수",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.2,
        help="전체 gradient 에너지 중 상위 p 비율을 차지하는 최소 픽셀 수를 ERF 크기로 사용",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="ERF size 를 저장할 .npy 파일 경로",
    )
    args = parser.parse_args()

    erf_sizes = compute_erf_for_model(
        args.model_family,
        args.fine_tuned_weight_name,
        args.test_dataset,
        num_samples=args.num_samples,
        p=args.p,
        seed=args.seed,
    )

    if args.output is None:
        default_name = f"erf_sizes_{args.model_family}.npy"
        args.output = default_name

    np.save(args.output, erf_sizes)
    print(f"[ERF] saved to {args.output}")


if __name__ == "__main__":
    main()
