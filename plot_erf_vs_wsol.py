# 산점도 생성

# coding: utf-8
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

FAMILY_COLORS = {
    "replknet": "#ff7f0e",   # orange
    "convnext": "#2ca02c",   # green
}

def main():
    parser = argparse.ArgumentParser("Plot ERF size vs WSOL score (RepLKNet / ConvNeXt)")
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV file containing model_name, family, erf_path, wsol")
    parser.add_argument("--save", type=str, default="erf_vs_wsol_scatter.png",
                        help="Output plot filename")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    erf_means = []
    erf_stds = []
    wscores = []
    families = []
    names = []

    for idx, row in df.iterrows():
        erf_path = row["erf_path"]
        if not os.path.exists(erf_path):
            raise FileNotFoundError(f"ERF file not found: {erf_path}")

        erf = np.load(erf_path)
        erf_mean = erf.mean()
        erf_std = erf.std()

        wsol = row["wsol"]
        if wsol <= 1.0:
            wsol = wsol * 100.0  # WSOL score to percentage

        erf_means.append(erf_mean)
        erf_stds.append(erf_std)
        wscores.append(wsol)
        families.append(row["family"])
        names.append(row["model_name"])

    erf_means = np.array(erf_means)
    erf_stds = np.array(erf_stds)
    wscores = np.array(wscores)

    # === Plot ===
    plt.figure(figsize=(7,5))
    for x, y, s, fam, name in zip(wscores, erf_means, erf_stds, families, names):
        color = FAMILY_COLORS.get(fam, "gray")
        plt.scatter(x, y, c=color, s=40)
        # 모델명 표시를 원하면:
        # plt.text(x, y, f" {name}", fontsize=8)

    # labels
    plt.xlabel("WSOL score (MaxBoxAcc)")
    plt.ylabel("ERF size")
    plt.title("Relationship between ERF size and WSOL score\n(RepLKNet vs ConvNeXt only)")

    # legend
    handles = []
    for fam, color in FAMILY_COLORS.items():
        handles.append(plt.scatter([], [], c=color, label=fam))
    plt.legend(handles=handles)

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    print(f"[저장 완료] {args.save}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
