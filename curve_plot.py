import json
import matplotlib.pyplot as plt
import torch

with open('../SSL_results/results_ssl/log_new/1/args.json', 'r') as f:
    exp1 = json.load(f)
train_time = exp1['test_knn']
# Plot the training curve
plt.plot(train_time, label='Diffusion')
plt.xlabel('Training epoch')
plt.ylabel('Test KNN')

# plt.legend()
plt.show()