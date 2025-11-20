import time
import pynvml

# Peak memory bandwidth lookup table (GB/s)
PEAK_MEMORY_BANDWIDTH = {
    "A100-SXM4-40GB": 1555,
    "A100-SXM4-80GB": 2039,
    "A100-PCIE-40GB": 1555,
    "A100-PCIE-80GB": 2039,
    "H100-SXM": 3350,
    "H100-PCIE": 3000,
    "V100-SXM2-32GB": 900,
    "V100-PCIE-32GB": 785,
    "RTX 4090": 1008,
    "RTX 4090 D": 936,
    "RTX 4080": 717,
    "RTX 3090": 936,
    "RTX 3080": 760,
}


def get_peak_bw(model_name: str):
    for key in PEAK_MEMORY_BANDWIDTH:
        if key.lower() in model_name.lower():
            return PEAK_MEMORY_BANDWIDTH[key]
    return None


def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    name = pynvml.nvmlDeviceGetName(handle)
    peak_bw = get_peak_bw(name)

    print(f"Monitoring GPU: {name}")
    print(f"Peak memory bandwidth used for estimate: {peak_bw} GB/s" if peak_bw else
          "[!] Peak bandwidth not found. Showing utilization only.")

    print("-" * 60)

    try:
        while True:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_util = util.memory  # percent memory controller utilization

            if peak_bw:
                est_bw = mem_util / 100 * peak_bw
                print(f"Memory Util: {mem_util:3d}%   "
                      f"Estimated Bandwidth: {est_bw:7.1f} GB/s")
            else:
                print(f"Memory Util: {mem_util:3d}%")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()

