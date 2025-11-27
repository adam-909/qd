from pathlib import Path
import pandas as pd

def load_repetita_demands(path: Path) -> pd.DataFrame:
    """
    Load a Repetita demand file like the Abilene.0000.demands you showed.

    Format recap:
    DEMANDS 110
    label src dest bw
    demand_14 1 5 30036
    ...
    """
    rows = []
    mode = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Section header
            if line.startswith("DEMANDS"):
                mode = "demands"
                continue
            if line.startswith("label"):
                # skip the column header
                continue

            if mode == "demands":
                parts = line.split()
                if len(parts) != 4:
                    continue
                label, src, dest, bw = parts
                rows.append(
                    {
                        "label": label,
                        "src": int(src),
                        "dst": int(dest),
                        "bw": float(bw),
                    }
                )

    return pd.DataFrame(rows)
