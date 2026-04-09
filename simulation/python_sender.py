"""
python_sender.py — Interactive scenario controller for Wokwi ESP32 simulation
Launches Wokwi CLI with --interactive and communicates via subprocess stdin/stdout.

Usage:
  python python_sender.py
  (No separate terminal needed — this script starts Wokwi itself)
"""

import subprocess
import threading
import time
import sys
import os
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────
WOKWI_CLI    = "wokwi-cli"
PROJECT_DIR = os.path.join(os.getcwd(), "simulate")
CSV_PATH     = "chemicals_in_wind_tunnel_fault.csv"

# Fault injection parameters — must match training script exactly
FAULT_LOW_MIN,  FAULT_LOW_MAX  = 0.00, 0.05
FAULT_HIGH_MIN, FAULT_HIGH_MAX = 0.92, 0.99

# ── Shared state ──────────────────────────────────────
esp32_output   = []
output_lock    = threading.Lock()
ready_event    = threading.Event()
response_event = threading.Event()

# ─────────────────────────────────────────────────────
def output_reader(proc):
    """Runs in a background thread — reads ESP32 serial output continuously."""
    for line in iter(proc.stdout.readline, b""):
        decoded = line.decode("utf-8", errors="ignore").rstrip()
        if not decoded:
            continue

        with output_lock:
            esp32_output.append(decoded)

        print(f"  ESP32> {decoded}")

        if "[READY]" in decoded:
            ready_event.set()

        if "ACTION" in decoded:
            response_event.set()

    proc.stdout.close()


def load_dataset(path):
    df = pd.read_csv(path, index_col=0)
    sensor_cols = [c for c in df.columns if c != "Chemical"]
    chemicals   = sorted(df["Chemical"].unique().tolist())
    return df, sensor_cols, chemicals


def inject_fault(row_values: np.ndarray, sensor_cols: list) -> np.ndarray:
    rng      = np.random.default_rng()
    n_broken = int(rng.exponential(scale=10)) + 2
    n_broken = min(n_broken, int(len(sensor_cols) * 0.5))
    broken   = rng.choice(len(sensor_cols), size=n_broken, replace=False)
    row      = row_values.copy()
    for idx in broken:
        if rng.random() < 0.5:
            row[idx] = rng.uniform(FAULT_LOW_MIN, FAULT_LOW_MAX)
        else:
            row[idx] = rng.uniform(FAULT_HIGH_MIN, FAULT_HIGH_MAX)
    return row


def send_row(proc, values: np.ndarray):
    line = ",".join(f"{v:.6f}" for v in values) + "\n"
    proc.stdin.write(line.encode("utf-8"))
    proc.stdin.flush()


def print_menu(chemicals: list, df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("  SCENARIO SELECTOR")
    print("=" * 52)
    for i, chem in enumerate(chemicals):
        count  = len(df[df["Chemical"] == chem])
        marker = "! " if chem in ("Ammonia_10000", "CO_1000", "CO_4000") else "  "
        print(f"  [{i+1:2d}] {marker}{chem:<26} ({count} rows)")
    print(f"\n  [{len(chemicals)+1:2d}]   Sensor_Fault (inject into random row)")
    print(f"  [ 0]   Quit")
    print("=" * 52)


# ─────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    try:
        df, sensor_cols, chemicals = load_dataset(CSV_PATH)
    except FileNotFoundError:
        print(f"[ERROR] {CSV_PATH} not found in current directory.")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows · {len(sensor_cols)} features · "
          f"{len(chemicals)} chemicals\n")

    cmd = [WOKWI_CLI, "simulate", "--interactive", "--timeout", "120000", PROJECT_DIR]

    print("Starting Wokwi simulator...\n")
    proc = subprocess.Popen(
        cmd,
        stdin =subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env   ={**os.environ}
    )

    reader = threading.Thread(target=output_reader, args=(proc,), daemon=True)
    reader.start()

    print("Waiting for ESP32 to boot (TFLite init + WiFi)...")
    if not ready_event.wait(timeout=300):
        print("[WARNING] No [READY] signal within 120s — proceeding anyway")
    else:
        print("\nESP32 is ready.\n")

    while True:
        print_menu(chemicals, df)
        raw = input("  Select: ").strip()

        if raw == "0":
            print("\nExiting.")
            break

        try:
            choice = int(raw)
        except ValueError:
            print("  Invalid input.\n")
            continue

        if 1 <= choice <= len(chemicals):
            chem   = chemicals[choice - 1]
            sample = (df[df["Chemical"] == chem][sensor_cols]
                      .sample(1, random_state=None)
                      .values[0])
            label  = chem

        elif choice == len(chemicals) + 1:
            base_row = df[sensor_cols].sample(1).values[0]
            sample   = inject_fault(base_row, sensor_cols)
            label    = "Sensor_Fault (injected)"

        else:
            print("  Invalid choice.\n")
            continue

        print(f"\n  Sending  : {label}")
        print(f"  Features : [{sample[0]:.4f}, {sample[1]:.4f}, "
              f"... {sample[-1]:.4f}]")
        print("  " + "-" * 38)

        response_event.clear()
        send_row(proc, sample)

        print("  Waiting for inference...\n")
        if not response_event.wait(timeout=20):
            print("  [No inference response received]")

        input("\n  Press Enter for next scenario...")

    proc.stdin.close()
    proc.terminate()
    print("Simulator stopped.")


if __name__ == "__main__":
    main()