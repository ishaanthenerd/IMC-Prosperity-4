from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import pandas as pd

# ==== CONFIG ====
BASE_FILE = Path("C:\\Users\\ishaa\\Downloads\\IMC-Prosperity-4\\algo\\R3_ig copy.py")

Z_TH_OPTIONS = [3, 4, 5]
WINDOW_OPTIONS = [10]

# Lines to replace (1-indexed)
LINE_START = 779
LINE_END = 781


def replace_block(lines, z1, z2, w1, w2, w3):
    """
    Replace lines 765–767 with new hyperparameter values.
    """
    new_lines = lines.copy()

    new_lines[LINE_START - 1] = f"                    {z1}, {w1},\n"
    new_lines[LINE_START]     = f"                    {z2}, {w2},\n"
    new_lines[LINE_START + 1] = f"                    20, {w3}\n"

    return new_lines


def run_backtest(file_path: Path) -> int:
    cmd = [
        "py",
        "-3.12",
        "-m",
        "prosperity4bt",
        str(file_path),
        "3-0",
        "--no-out",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Backtest failed.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    final_line = lines[-1]
    token = final_line.split()[-1].replace(",", "")
    return int(token)


def main():
    original_lines = BASE_FILE.read_text(encoding="utf-8").splitlines(keepends=True)

    results = []

    for z1 in Z_TH_OPTIONS:
        for z2 in Z_TH_OPTIONS:
            for w1 in WINDOW_OPTIONS:
                for w2 in WINDOW_OPTIONS:
                    for w3 in WINDOW_OPTIONS:

                        modified = replace_block(original_lines, z1, z2, w1, w2, w3)

                        with tempfile.TemporaryDirectory() as tmpdir:
                            tmp_path = Path(tmpdir) / BASE_FILE.name
                            tmp_path.write_text("".join(modified), encoding="utf-8")

                            try:
                                profit = run_backtest(tmp_path)
                            except Exception as e:
                                print(f"Failed combo: {z1},{z2},{w1},{w2},{w3}")
                                print(e)
                                continue

                        results.append({
                            "z_th_1": z1,
                            "z_th_2": z2,
                            "window_1": w1,
                            "window_2": w2,
                            "window_3": w3,
                            "profit": profit
                        })

                        print(f"Done: {z1},{z2},{w1},{w2},{w3} -> {profit}")

    df = pd.DataFrame(results)
    df = df.sort_values(by="profit", ascending=False)

    print("\nTop 5 results:")
    print(df.head(5))


if __name__ == "__main__":
    main()