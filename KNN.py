import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

# Function to generate a sample dataset (two classes)
def generate_dataset():
    # np.random.seed(42)
    class1 = np.random.randn(25, 2) + [2, 2]   # Cluster 1
    class2 = np.random.randn(25, 2) + [4, 4]   # Cluster 2
    labels1 = np.zeros((class1.shape[0], 1))   # Label 0
    labels2 = np.ones((class2.shape[0], 1))    # Label 1
    data = np.vstack((np.hstack((class1, labels1)),
                      np.hstack((class2, labels2))))
    return data

# === Student's Task ===
def classify(x, y, dataset, k=3):
    """
    TODO: Implement your own KNN classification.
    Steps:
    1. Calculate Euclidean distance from (x, y) to each point in dataset
    2. Sort distances and pick the k nearest neighbors
    3. Return the most common label among those neighbors
    """
    pass

# Draw decision boundary using *student's classify()*
def plot_decision_boundary(ax, dataset, k):
    h = 0.05  # Grid step size
    x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    y_min, y_max = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Use student's classify() here
    Z = np.array([classify(x, y, dataset, k) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)

# --- Main Program ---
dataset = generate_dataset()
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.15)

k_val = 3  # Initial K value

# Display K value
title_text = ax.text(0.5, -0.12, f"K = {k_val}", transform=ax.transAxes,
                     ha="center", fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.7))

# Initial scatter
scatter = ax.scatter(dataset[:, 0], dataset[:, 1],
                     c=dataset[:, 2], cmap=ListedColormap(['red', 'blue']),
                     edgecolor='k', s=50)

# Plot decision boundary (students will see the effect of their classify())
plot_decision_boundary(ax, dataset, k_val)
ax.set_title('Interactive KNN')

# Mouse click classification
def onclick(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    pred_class = classify(x, y, dataset, k_val)
    color = 'red' if pred_class == 0 else 'blue'
    ax.scatter([x], [y], c=color, edgecolor='k', s=100, marker='X')
    fig.canvas.draw_idle()
    print(f"Clicked at ({x:.2f}, {y:.2f}) -> Class: {pred_class}")

# Update decision boundary
def update_plot():
    ax.clear()
    plot_decision_boundary(ax, dataset, k_val)
    ax.scatter(dataset[:, 0], dataset[:, 1],
               c=dataset[:, 2], cmap=ListedColormap(['red', 'blue']),
               edgecolor='k', s=50)
    ax.set_title('Interactive KNN')
    global title_text
    title_text = ax.text(0.5, -0.12, f"K = {k_val}", transform=ax.transAxes,
                         ha="center", fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.7))
    fig.canvas.draw_idle()

# Keyboard control for K
def on_key(event):
    global k_val
    if event.key in ['up', '+']:
        k_val += 1
        update_plot()
    elif event.key in ['down', '-'] and k_val > 1:
        k_val -= 1
        update_plot()

# Connect events
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
