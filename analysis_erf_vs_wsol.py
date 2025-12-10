
# coding: utf-8
"""
ERF 크기 vs WSOL(MaxBoxAcc) 산점도 분석 스크립트

사용 예시:
    python analysis_erf_vs_wsol.py \
        --config erf_wsol_config.json \
        --save_fig erf_vs_wsol.png \
        --show
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list):
        raise ValueError("config JSON은 리스트 형식이어야 합니다.")
    return cfg


def load_erf_stats(erf_path):
    if not os.path.exists(erf_path):
        raise FileNotFoundError(f"ERF npy 파일을 찾을 수 없습니다: {erf_path}")
    erf = np.load(erf_path)
    mean = float(erf.mean())
    std = float(erf.std())
    return mean, std, erf


def normalize_maxboxacc(v):
    """
    MaxBoxAcc 값이 0~1 사이인지, 0~100 사이인지 자동 판단해서
    % 단위(0~100)로 변환.
    """
    if v <= 1.0:
        return v * 100.0
    return v


def main():
    parser = argparse.ArgumentParser("ERF vs WSOL (MaxBoxAcc) analysis")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="ERF/WSOL 설정 JSON 경로 (리스트 형식).",
    )
    parser.add_argument(
        "--save_fig",
        type=str,
        default=None,
        help="그래프를 저장할 이미지 파일 이름 (예: erf_vs_wsol.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="그래프를 화면에 표시할지 여부",
    )
    args = parser.parse_args()

    cfg_list = load_config(args.config)

    names = []
    erf_means = []
    erf_stds = []
    wsol_scores = []

    print("===== 모델별 ERF / WSOL 통계 =====")

    for item in cfg_list:
        name = item["name"]
        erf_path = item["erf_path"]
        maxboxacc_raw = float(item["maxboxacc"])

        mean_erf, std_erf, _ = load_erf_stats(erf_path)
        maxboxacc_pct = normalize_maxboxacc(maxboxacc_raw)

        names.append(name)
        erf_means.append(mean_erf)
        erf_stds.append(std_erf)
        wsol_scores.append(maxboxacc_pct)

        print(f"[{name}]")
        print(f"  ERF: mean={mean_erf:.2f}, std={std_erf:.2f}")
        print(f"  MaxBoxAcc: {maxboxacc_pct:.2f}%")
        print("")

    erf_means = np.array(erf_means, dtype=np.float32)
    erf_stds = np.array(erf_stds, dtype=np.float32)
    wsol_scores = np.array(wsol_scores, dtype=np.float32)

    # 상관계수 및 간단한 선형 회귀
    if len(erf_means) >= 2:
        corr = np.corrcoef(erf_means, wsol_scores)[0, 1]
        # y = a x + b
        a, b = np.polyfit(erf_means, wsol_scores, 1)
        # 결정계수 R^2
        y_pred = a * erf_means + b
        ss_tot = ((wsol_scores - wsol_scores.mean()) ** 2).sum()
        ss_res = ((wsol_scores - y_pred) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        print("===== 전체 상관/회귀 분석 =====")
        print(f"Pearson 상관계수 (ERF mean vs MaxBoxAcc): {corr:.4f}")
        print(f"선형 회귀: MaxBoxAcc ≈ {a:.4f} * ERF_mean + {b:.4f}")
        print(f"결정계수 R^2: {r2:.4f}")
        print("")
    else:
        corr = None
        a, b, r2 = None, None, None
        print("모델이 1개 뿐이라 상관/회귀 분석은 수행하지 않습니다.")

    # ----- 산점도 그리기 -----
    fig, ax = plt.subplots(figsize=(7, 5))

    # x축: mean ERF size, y축: MaxBoxAcc (%)
    ax.errorbar(
        erf_means,
        wsol_scores,
        xerr=erf_stds,
        fmt="o",
        ecolor="gray",
        capsize=4,
        label="models",
    )

    # 각 점에 라벨 표시
    for x, y, name in zip(erf_means, wsol_scores, names):
        ax.text(
            x,
            y,
            f" {name}",
            fontsize=9,
            verticalalignment="center",
        )

    ax.set_xlabel("Mean ERF size (pixels in top-p region)")
    ax.set_ylabel("WSOL performance (MaxBoxAcc, %)")
    ax.set_title("ERF size vs WSOL (MaxBoxAcc)")

    # 회귀선 그리기 (모델이 2개 이상일 때)
    if len(erf_means) >= 2:
        x_line = np.linspace(erf_means.min() * 0.9, erf_means.max() * 1.1, 100)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, "--", label=f"Linear fit (R²={r2:.2f})")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    # 저장
    if args.save_fig is not None:
        fig.savefig(args.save_fig, dpi=200)
        print(f"[저장 완료] figure -> {args.save_fig}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
