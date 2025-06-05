import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import time

parser = argparse.ArgumentParser(description="Plot loss curves from log file.")
parser.add_argument("--log", type=str, required=True, help="Path to the log file (CSV format)")
args = parser.parse_args()

log_path = args.log

if not os.path.exists(log_path):
    raise FileNotFoundError(f"log file does not exist: {log_path}")

try:
    while True:
        # 讀取 log
        df = pd.read_csv(log_path)

        # 清除圖形，避免重疊或資源消耗
        plt.clf()
        plt.close()

        # 建立兩個 subplot
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [10, 3]}
        )

        alpha = 0.7

        # --- 第一張圖：Loss 曲線 ---
        ax1.plot(df["epoch"], df["loss_sdf"], label="SDF Loss", alpha=alpha, linewidth=1, linestyle='--')
        ax1.plot(df["epoch"], df["loss_zero"], label="Zero Constraint", alpha=alpha, linewidth=1, linestyle='--')
        ax1.plot(df["epoch"], df["loss_eikonal"], label="Eikonal", alpha=alpha, linewidth=1, linestyle='--')
        ax1.plot(df["epoch"], df["loss_normal"], label="Normal", alpha=alpha, linewidth=1, linestyle='--')
        ax1.plot(df["epoch"], df["loss_sparse"], label="Sparse", alpha=alpha, linewidth=1, linestyle='--')
        ax1.plot(df["epoch"], df["loss_render"], label="Render", alpha=alpha, linewidth=1, linestyle='--')

        ax1.set_yscale("log")
        ax1.plot(df["epoch"], df["loss_total"], label="Total Loss", color="black", linewidth=3)
        ax1.grid(True)
        ax1.legend()

        # --- 第二張圖：Learning Rate ---
        ax2.plot(df["epoch"], df["learning_rate"], color='blue', label="Learning Rate")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("LR")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show(block=False)  # 不阻斷程式執行
        plt.pause(0.1)         # 顯示圖形
        time.sleep(10)         # 等待 10 秒
except KeyboardInterrupt:
    print("\nEnd plotting.")



# ax1.plot(df["epoch"], df["loss_consistency"], label="Consistency", alpha=alpha, linewidth=1, linestyle='--')
