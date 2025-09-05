# import numpy as np
# import matplotlib.pyplot as plt

# # Load one processed file
# d = np.load("data/interim/kep_12366084.npz")  # replace with a real file in your folder
# print("Keys in file:", d.files)

# global_curve = d["global_0"]
# local_curve = d["local_0"]
# meta = d["meta_0"]

# print("Meta:", meta)

# # Plot global vs local views
# fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# axs[0].plot(global_curve, "k.", alpha=0.6)
# axs[0].set_title("Global View (entire orbit)")
# axs[0].set_xlabel("Phase bins")
# axs[0].set_ylabel("Normalized flux")

# axs[1].plot(local_curve, "b.", alpha=0.6)
# axs[1].set_title("Local View (zoomed transit)")
# axs[1].set_xlabel("Phase bins")
# axs[1].set_ylabel("Normalized flux")

# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# --- Manually enter NPZ file name ---
filename = "data/interim/kep_11460018.npz"   # <-- replace with your actual file name

# Load NPZ file
try:
    data = np.load(filename)

    # Show what arrays are inside the file
    print("Available arrays:", list(data.keys()))

    # --- Manually select which curve to plot from inside NPZ ---
    selected = "global_0"   # e.g., "global_0", "local_0", etc.

    if selected in data:
        y = data[selected]
        x = np.arange(len(y))  # use index as x-axis

        print(f"Plotting curve: {selected} from {filename}")

        # Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, ".", markersize=2, alpha=0.7)
        plt.xlabel("Index (Time steps)")
        plt.ylabel("Flux / Value")
        plt.title(f"{selected} curve from {filename}")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"'{selected}' not found in {filename}. Available: {list(data.keys())}")

except Exception as e:
    print("Error loading file:", e)
