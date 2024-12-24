from data import addition_mod_p_data, k_addition_mod_p_data, cycle
from model import Transformer, MLP, LSTM

from math import ceil
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"Hyperparameter setting"
torch.manual_seed(11)
p = 41
K = 2
training_fraction = 0.7
batch_size = 256
iters = 10000
save_iters = 10000

optimizer = 'AdamW'
nn_model = 'LSTM'
learning_rate = 5e-4
weight_decay = 1
dropout = 0

# Load checkpoints (if required)
resume_from_checkpoint = False
checkpoint_path = "checkpoint/checkpoint_iteration_20000.pth"

# Model initialization
if nn_model == 'Transformer':
    model = Transformer(num_layers=2, dim_model=128, num_heads=4, num_tokens=p+2, seq_len=5, dropout=dropout).to(device)
elif nn_model == 'MLP':
    model = MLP(num_layers=2, dim_model=128, num_heads=4, num_tokens=p+2, seq_len=5, dropout=dropout).to(device)
elif nn_model == 'LSTM':
    model = LSTM(num_layers=2, dim_model=128, hidden_dim=4*128, num_tokens=p+2, seq_len=5, dropout=dropout).to(device)
else:
    raise ValueError(f'Undefined network framework: {nn_model}')

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
if optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), weight_decay=weight_decay)
elif optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
elif optimizer == 'RMSprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.95, momentum=0.9, weight_decay=weight_decay)
else:
    ValueError(f'Please initialize the optimizer.: {optimizer}')
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)

# Data generation
inputs, labels = addition_mod_p_data(p, eq_token=p, op_token=p+1)
if K != 2:
    inputs, labels = k_addition_mod_p_data(p, K, eq_token=p, op_token=p+1)

dataset = torch.utils.data.TensorDataset(inputs, labels)
train_size = int(training_fraction * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
batch_size = min(batch_size, ceil(len(dataset) / 2))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_cycle_loader = cycle(train_loader)
valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

if resume_from_checkpoint:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iters = checkpoint['iteration'] + 1
    train_batch_losses = checkpoint['train_batch_losses']
    valid_losses = checkpoint['valid_losses']
    train_accuracies = checkpoint['train_accuracies']
    valid_accuracies = checkpoint['valid_accuracies']
else:
    start_iters = 1
    train_batch_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"folder'{checkpoint_dir}' is created。")
    else:
        print(f"folder'{checkpoint_dir}'already exists。")


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)[-1, :, :]
            loss = criterion(output, y)
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


# training and validating
for i in range(start_iters, iters + 1):
    model.train()
    train_batch_data = next(train_cycle_loader)
    x = torch.tensor(train_batch_data[0], device=device)
    y = torch.tensor(train_batch_data[1], device=device)
    optimizer.zero_grad()
    output = model(x)[-1, :, :]
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    scheduler.step()
    # Limit the minimum learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], 1e-8)

    train_batch_loss = loss.item()

    valid_loss, accuracy = evaluate(model, valid_loader, criterion, device)
    train_accuracy = evaluate(model, train_loader, criterion, device)[1]  # 评估训练集正确率

    train_batch_losses.append(train_batch_loss)
    valid_losses.append(valid_loss)
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(accuracy)

    print(f"Iteration {i}: Train Batch Loss={train_batch_loss:.4f}, Valid Loss={valid_loss:.4f}, "
          f"Train Accuracy={train_accuracy:.2%}, Valid Accuracy={accuracy:.2%}")

    # Save checkpoint
    if i % save_iters == 0:
        torch.save({
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_batch_losses': train_batch_losses,
            'valid_losses': valid_losses,
            'train_accuracies': train_accuracies,
            'valid_accuracies': valid_accuracies
        }, f"checkpoint/checkpoint_iteration_{i}.pth")

# plot
plt.figure(figsize=(12, 6))

# 损失图像
plt.subplot(1, 2, 1)
plt.plot(range(1, iters + 1), train_batch_losses, label="Train Batch Loss", linestyle='-')
plt.plot(range(1, iters + 1), valid_losses, label="Valid Loss", linestyle='-')
plt.xscale("log")
plt.xlabel("Optimization steps (Log Scale)")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Loss vs. Optimization steps")
plt.legend()

# 准确率图像
plt.subplot(1, 2, 2)
plt.plot(range(1, iters + 1), train_accuracies, label="Train Accuracy", linestyle='-')
plt.plot(range(1, iters + 1), valid_accuracies, label="Valid Accuracy", linestyle='-')
plt.xscale("log")
plt.xlabel("Optimization steps (Log Scale)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Optimization steps")
plt.legend()

plt.tight_layout()
plt.savefig('loss and accuracy')
plt.show()