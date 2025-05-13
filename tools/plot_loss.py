import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description="Plot loss curves from log file.")
parser.add_argument("--log", type=str, required=True, help="Path to the log file (CSV format)")
args = parser.parse_args()

log_path = args.log

if not os.path.exists(log_path):
    raise FileNotFoundError(f"log file does not exist: {log_path}")

# read log
df = pd.read_csv(log_path)

# plot
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["loss_total"], label="Total Loss", linewidth=2)
plt.plot(df["epoch"], df["loss_sdf"], label="SDF Loss")
plt.plot(df["epoch"], df["loss_zero"], label="Zero Constraint")
plt.plot(df["epoch"], df["loss_eikonal"], label="Eikonal")
# plt.plot(df["epoch"], df["loss_edge"], label="Edge")
plt.plot(df["epoch"], df["loss_normal"], label="Normal")

plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.title("SDF Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.yscale("log")

plt.show()
