import os
import re
import pickle
import numpy as np

names = ["MetabolicState_2", "State_Modality_6", "State_Modality_Valence_16"]

EXPECTED_RUNS = range(1, 51)  # 1..50 inclusive
pattern = re.compile(r"^(?P<base>.+)_(?P<run>\d+)\.pkl$")

for name in names:
    folder = f"results/_chkpt_finals/{name}/runs"
    files = sorted(f for f in os.listdir(folder) if f.endswith(".pkl"))

    # base_id -> {"runs": set(int), "accs": list(float), "best": (acc, filename)}
    groups = {}

    for file in files:
        m = pattern.match(file)
        if not m:
            print(f"[SKIP] Unexpected filename format: {os.path.join(folder, file)}")
            continue

        base = m.group("base")          # e.g., C16_E64_H16
        run = int(m.group("run"))       # e.g., 34
        path = os.path.join(folder, file)

        # init group
        if base not in groups:
            groups[base] = {"runs": set(), "accs": [], "best": (-np.inf, None), "bad": 0}

        groups[base]["runs"].add(run)

        # load accuracy
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            groups[base]["bad"] += 1
            print(f"[SKIP] Corrupted pickle: {path} ({e})")
            continue
        except Exception as e:
            groups[base]["bad"] += 1
            print(f"[SKIP] Failed to load {path}: {e}")
            continue

        acc = data.get("accuracy", None)
        if acc is None:
            groups[base]["bad"] += 1
            print(f"[SKIP] No 'accuracy' key in {path}")
            continue

        groups[base]["accs"].append(acc)

        if acc > groups[base]["best"][0]:
            groups[base]["best"] = (acc, file)

    # ---- summary per task group folder ----
    print("\n" + "=" * 80)
    print(f"{name}  |  folder: {folder}")
    if not groups:
        print("No valid base IDs found.")
        continue

    for base in sorted(groups.keys()):
        accs = groups[base]["accs"]
        present_runs = groups[base]["runs"]
        bad = groups[base]["bad"]
        best_acc, best_file = groups[base]["best"]

        missing = sorted(set(EXPECTED_RUNS) - present_runs)

        if len(accs) == 0:
            print(f"- {base}: no valid accuracies (bad files: {bad})")
            print(f"  present runs: {sorted(present_runs)[:10]}{' ...' if len(present_runs)>10 else ''}")
            print(f"  missing runs ({len(missing)}): {missing}")
            continue

        mean = float(np.mean(accs))
        std = float(np.std(accs))  # population std; use ddof=1 for sample std

        print(
            f"- {base}: "
            f"n={len(accs)} acc mean={mean:.4f} ± {std:.4f} | "
            f"best={best_acc:.4f} ({best_file}) | bad={bad}"
        )

        if missing:
            # compact printing if many missing
            if len(missing) <= 20:
                print(f"  missing runs ({len(missing)}): {missing}")
            else:
                print(f"  missing runs ({len(missing)}): {missing[:10]} ... {missing[-10:]}")
