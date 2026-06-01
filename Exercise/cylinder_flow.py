"""
============================================================
参考答案：圆柱绕流流速场预测 — 多层感知机（MLP）回归
============================================================
物理场景：海底管道/圆柱结构周围的流速场预测
目标：根据雷诺数、圆柱直径、来流速度、测点位置，预测归一化流速 V/U
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ------------------------ 1. 数据加载与预处理 ------------------------

df = pd.read_csv('cylinder_flow_dataset.csv')

# 特征与目标
feature_cols = ['Re', 'D', 'U', 'r', 'theta']
target_col = 'V_norm'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# 特征标准化（对神经网络至关重要）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 转换为Tensor
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

# ------------------------ 2. 自定义Dataset ------------------------

class FlowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 划分训练集/验证集/测试集 (70% / 15% / 15%)
dataset = FlowDataset(X_tensor, y_tensor)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size], 
    generator=torch.Generator().manual_seed(42)
)

batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# ------------------------ 3. 构建MLP模型 ------------------------

class FlowPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[64, 128, 64], output_dim=1, dropout=0.2):
        super(FlowPredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 实例化模型
model = FlowPredictor(input_dim=5, hidden_dims=[64, 128, 64], output_dim=1, dropout=0.2)
print(model)

# ------------------------ 4. 训练配置 ------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ------------------------ 5. 训练循环 ------------------------

num_epochs = 200
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_set)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(val_set)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# ------------------------ 6. 测试评估 ------------------------

model.eval()
test_preds = []
test_trues = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        test_preds.append(outputs.cpu().numpy())
        test_trues.append(y_batch.numpy())

test_preds = np.concatenate(test_preds)
test_trues = np.concatenate(test_trues)

# 反标准化到原始尺度
test_preds_inv = scaler_y.inverse_transform(test_preds)
test_trues_inv = scaler_y.inverse_transform(test_trues)

# 评估指标
mse = np.mean((test_preds_inv - test_trues_inv) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_preds_inv - test_trues_inv))
r2 = 1 - np.sum((test_trues_inv - test_preds_inv)**2) / np.sum((test_trues_inv - np.mean(test_trues_inv))**2)

print(f"\n{'='*50}")
print(f"测试集评估结果:")
print(f"  MSE  = {mse:.6f}")
print(f"  RMSE = {rmse:.6f}")
print(f"  MAE  = {mae:.6f}")
print(f"  R²   = {r2:.4f}")
print(f"{'='*50}")

# ------------------------ 7. 可视化 ------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 损失曲线
ax1 = axes[0]
ax1.plot(train_losses, label='Train Loss', color='steelblue')
ax1.plot(val_losses, label='Val Loss', color='coral')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 预测 vs 真实值
ax2 = axes[1]
ax2.scatter(test_trues_inv, test_preds_inv, alpha=0.5, s=10, color='steelblue')
ax2.plot([test_trues_inv.min(), test_trues_inv.max()], 
         [test_trues_inv.min(), test_trues_inv.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('True V/U')
ax2.set_ylabel('Predicted V/U')
ax2.set_title(f'Prediction vs Ground Truth (R² = {r2:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_prediction_results.png', dpi=200)
plt.show()
