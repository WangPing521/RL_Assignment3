import torch
from tqdm.auto import tqdm
from torch.optim import Adam
from pathlib import Path
from utils.func_tool import fix_experiment_seed, Unet, get_dataloaders, t_sample, p_losses, sample, show_image, \
    save_logs, q_sample, test_sample_loop
from torchvision.utils import save_image
import time


fix_experiment_seed(seed=123)

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training Hyperparameters
train_batch_size = 64   # Batch Size
lr = 1e-4         # Learning Rate

# Define Dataset Statistics
image_size = 32
input_channels = 3
data_root = './data'
epochs = 100
T = 1000

model = Unet(
    dim=image_size,
    channels=input_channels,
    dim_mults=(1, 2, 4, 8)
  )
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

logger = dict()
logger['train_time'] = [0]

train_dataloader, test_loader = get_dataloaders(data_root, batch_size=train_batch_size, image_size=image_size)
for epoch in range(epochs):
    start_time = time.time()
    with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch: {epoch}")

            optimizer.zero_grad()
            imgs, _ = batch
            batch_size = imgs.shape[0]
            x = imgs.to(device)

            t = t_sample(T, batch_size).to(device)  # Randomly sample timesteps uniformly from [0, T-1]

            loss = p_losses(model, x, t)

            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

    freq_time = time.time() - start_time
    logger['train_time'].append(freq_time)

    # Sample and Save Generated Images
    save_image((x + 1.) * 0.5, './results/orig.png')
    samples = sample(model, image_size=image_size, batch_size=64, channels=input_channels)
    samples = (torch.Tensor(samples[-1]) + 1.) * 0.5
    save_image(samples, f'./results/samples_{epoch}.png')
    T_reverse = [50, 100, 200, 300, 500]
    if epoch % 20 == 0 or epoch == epochs - 1:
        with tqdm(test_loader, unit="batch", leave=False) as tepoch:
            batch_id = 0
            for batch in tepoch:
                tepoch.set_description(f"Test_Epoch: {epoch}")

                imgs, _ = batch
                batch_size = imgs.shape[0]
                x = imgs.to(device)

                if batch_id == 0:
                    save_image((x + 1.) * 0.5, f'./results/orig_{epoch}_{batch_id}.png')

                    for R_t in T_reverse:
                        t = torch.randint(1, R_t, (batch_size,), device=x.device).long()
                        img_noisy = q_sample(x, t)
                        with torch.no_grad():
                            img_0 = test_sample_loop(model, img_noisy, R_t)
                        test_samples = (torch.Tensor(img_0[-1]) + 1.) * 0.5
                        save_image(test_samples, f'./results/samples_{epoch}_{batch_id}_{R_t}.png')
                batch_id = batch_id + 1
save_logs(logger, "results/log_new", str(1))