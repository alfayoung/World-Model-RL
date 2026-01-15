import numpy as np
import matplotlib.pyplot as plt

# Define the function in NumPy
def compute_k_seeds_np(
    training_step,
    initial_k=5,
    final_k=1,
    decay_steps=200000,
    alpha=1.0,
):
    progress = training_step / decay_steps
    decay = np.exp(-alpha * progress)
    k = final_k + (initial_k - final_k) * decay
    return max(int(np.floor(k)), final_k)

# Generate data
steps = np.arange(0, 500000, 100)
k_values = [compute_k_seeds_np(s) for s in steps]

# Plot
plt.figure()
plt.plot(steps, k_values)
plt.xlabel("training_step")
plt.ylabel("k")
plt.title("compute_k_seeds decay behavior")
plt.savefig("k_decay_plot.png")  # Save the figure as a PNG file
