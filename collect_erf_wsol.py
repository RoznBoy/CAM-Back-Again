# 각 실험 결과의 ERF / WSOL을 erf_wsol_points.csv 에 누적

# collect_erf_wsol.py
# coding: utf-8
import os
import csv
import re
import subprocess
from dataclasses import dataclass

CUB_ROOT = "/home/hjshin/CAM-Back-Again/datasets/CUB_200_2011"
DATASET_NAME = "cub-200-2011"  # generate_heatmap / erf_compute 에서 쓰는 이름

@dataclass
class Experiment:
    name: str
    model_family: str       # 'convnext' or 'replknet'
    weight_path: str        # fine-tuned weight .pth path
    heatmap_dir: str        # e.g. 'heatmap_conv_r384_e100'
    np_heatmap_dir: str     # e.g. 'np_heatmap_conv_r384_e100'
    erf_path: str           # e.g. 'erf_conv_r384_e100.npy'


EXPERIMENTS = [
    Experiment(
        name="conv_r384_e100_lr1e4_light",
        model_family="convnext",
        weight_path="weights/conv_r384_e100_lr1e4_light_best.pth",
        heatmap_dir="heatmap_conv_r384_e100",
        np_heatmap_dir="np_heatmap_conv_r384_e100",
        erf_path="erf_conv_r384_e100.npy",
    ),
    Experiment(
        name="replk_r384_e100_lr5e5_light",
        model_family="replknet",
        weight_path="weights/replk_r384_e100_lr5e5_light_best.pth",
        heatmap_dir="heatmap_replk_r384_e100",
        np_heatmap_dir="np_heatmap_replk_r384_e100",
        erf_path="erf_replk_r384_e100.npy",
    ),
    # 여기에 실험들을 계속 추가
]


def run_cmd(cmd):
    print(">>", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)
    print(out)
    return out


def parse_maxboxacc(wsol_stdout: str) -> float:
    """
    wsol_eval.py 출력에서 'MaxBoxAcc    : 89.68%' 같은 줄에서 숫자만 뽑는다.
    리턴값은 0~1 스케일.
    """
    m = re.search(r"MaxBoxAcc\s*:\s*([0-9\.]+)%", wsol_stdout)
    if not m:
        raise ValueError("MaxBoxAcc line not found in wsol_eval output")
    val_pct = float(m.group(1))
    return val_pct / 100.0


def main():
    csv_path = "erf_wsol_points.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model_name", "family", "erf_path", "wsol"])

        for exp in EXPERIMENTS:
            # 1) heatmap 생성
            run_cmd([
                "python", "generate_heatmap.py",
                "--model_family", exp.model_family,
                "--fine_tuned_weight_name", exp.weight_path,
                "--test_dataset", DATASET_NAME,
                "--heatmap_output", exp.heatmap_dir,
                "--localization_method", "cam",
            ])

            # 2) WSOL 평가 (MaxBoxAcc)
            wsol_out = run_cmd([
                "python", "wsol_eval.py",
                "--np_root", f"np_{exp.heatmap_dir}",
                "--cub_root", CUB_ROOT,
                "--iou_thr", "0.5",
            ])
            wsol_score = parse_maxboxacc(wsol_out)

            # 3) ERF 계산
            run_cmd([
                "python", "erf_compute.py",
                "--model_family", exp.model_family,
                "--fine_tuned_weight_name", exp.weight_path,
                "--test_dataset", DATASET_NAME,
                "--num_samples", "200",
                "--output", exp.erf_path,
            ])

            # 4) CSV에 한 줄 추가
            writer.writerow([exp.name, exp.model_family, exp.erf_path, wsol_score])
            print(f"[LOGGED] {exp.name}: wsol={wsol_score:.4f}, erf_path={exp.erf_path}")


if __name__ == "__main__":
    main()

#python collect_erf_wsol.py
