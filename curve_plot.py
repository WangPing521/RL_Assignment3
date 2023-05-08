import json
import matplotlib.pyplot as plt
import torch

with open('../VAE_Results/results_vae/log_new/1/args.json', 'r') as f:
    exp1 = json.load(f)
train_time = exp1['train_time']
# Plot the training curve
plt.plot(train_time, label='Diffusion')
plt.xlabel('Training epoch')
plt.ylabel('Training time')

# plt.legend()
plt.show()