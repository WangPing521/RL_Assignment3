import json
import matplotlib.pyplot as plt
import torch

with open('../SSL_results/results_ssl_withGrad/log_new/1/args.json', 'r') as f:
    exp1 = json.load(f)
train_time1 = exp1['train_knn']

with open('../SSL_results/results_ssl_withoutGrad/log_new/1/args.json', 'r') as f:
    exp2 = json.load(f)
train_time2 = exp2['train_knn']





# Plot the training curve
plt.plot(train_time1, label='stop_Grad')
plt.plot(train_time2, label='without stop_Grad')

plt.xlabel('Training epoch')
plt.ylabel('Training knn')

plt.legend()
plt.show()