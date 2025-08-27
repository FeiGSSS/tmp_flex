
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def find_col(cols, key):
    for c in cols:
        if key in c:
            return c
    raise KeyError(f"Column containing '{key}' not found!")

def load_gpu_usage(csv_file):
    df = pd.read_csv(csv_file, skipinitialspace=True)
    util_col = find_col(df.columns, 'utilization.gpu')
    mem_col = find_col(df.columns, 'memory.used')
    time_col = find_col(df.columns, 'timestamp')
    df[util_col] = df[util_col].str.replace('%', '').str.replace('[^0-9.]', '', regex=True).astype(float)
    df[mem_col] = df[mem_col].str.replace('MiB', '').str.replace('[^0-9.]', '', regex=True).astype(float)
    df[time_col] = pd.to_datetime(df[time_col])
    return df, util_col, time_col


def main():
    parser = argparse.ArgumentParser(description='Plot GPU usage curves.')
    parser.add_argument('--output', type=str, default='gpu_usage_compare.png', help='Output image filename')
    args = parser.parse_args()

    df_numa, util_col, time_col = load_gpu_usage('gpu_usage_numa.csv')
    df_disk, _, _ = load_gpu_usage('gpu_usage_disk.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(df_numa[util_col], label='NUMA', color='royalblue', linewidth=2)
    plt.plot(df_disk[util_col], label='DISK', color='orange', linewidth=2)

    plt.xlabel('Time', fontsize=14)
    plt.ylabel('GPU Utilization (%)', fontsize=14)
    plt.title('GPU0 Utilization Comparison Curve', fontsize=16)
    plt.legend(fontsize=13, loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.show()

if __name__ == '__main__':
    main()