import os 
import pickle 

names = ["MetabolicState_2", "State_Modality_6", "State_Modality_Valence_16"]

for name in names:
    folder = f"results/_chkpt_finals/{name}/runs"
    acc_best = 0
    for file in os.listdir(folder):
        if file.endswith(".pkl"):
            try:
                with open(os.path.join(folder, file), "rb") as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"[SKIP] Corrupted pickle: {os.path.join(folder, file)} ({e})")
                continue
            except Exception as e:
                print(f"[SKIP] Failed to load {os.path.join(folder, file)}: {e}")
                continue
            if data['accuracy'] > acc_best:
                acc_best = data['accuracy']
                best_file = file
    print(name, best_file, acc_best)
            