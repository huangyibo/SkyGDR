import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------
# Config
# ------------------------------
CSV_FILE = "../../results/results.csv"        # your input csv file
OUT_DIR = "../../results/"                    # output dir
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv(CSV_FILE)

# ordered by blocks
df = df.sort_values(by="blocks")

# ------------------------------
# 1. Plot OPS vs Blocks
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["blocks"], df["ops_per_s"], marker="o")
plt.xlabel("Blocks")
plt.ylabel("OPS (ops/s)")
plt.title("OPS vs Blocks")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ops_vs_blocks.png"))
plt.close()

# ------------------------------
# 2. Plot Throughput vs Blocks
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["blocks"], df["throughput_gib_s"], marker="o")
plt.xlabel("Blocks")
plt.ylabel("Throughput (GiB/s)")
plt.title("Throughput vs Blocks")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "throughput_vs_blocks.png"))
plt.close()

# ------------------------------
# 3. Plot Elapsed Time vs Blocks
# ------------------------------
plt.figure(figsize=(8, 5))
plt.plot(df["blocks"], df["elapsed_s"], marker="o")
plt.xlabel("Blocks")
plt.ylabel("Elapsed Time (s)")
plt.title("Latency (Elapsed Time) vs Blocks")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "elapsed_vs_blocks.png"))
plt.close()

print("Plots saved to:", OUT_DIR)

