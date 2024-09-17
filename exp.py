from imports import *

with open("pythia_full_data_single_token_grad.pkl", "rb") as f:
    data = pickle.load(f)


print(data)

# Visualize the gradients using a heatmap
plt.figure(figsize=(15, 6))
sns.heatmap(data, cmap='viridis', cbar=True, yticklabels=range(6), xticklabels=range(128))
plt.xlabel('Token Index')
plt.ylabel('Layer Index')
plt.title('Token-wise Gradient Norms Across Layers for mlp.dense_4h_to_h on log scale')
plt.show()
plt.close