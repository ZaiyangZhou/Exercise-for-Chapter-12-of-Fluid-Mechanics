import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 读取数据
df = pd.read_csv("coastal_wave_mlp_dataset_500.csv")

feature_cols = [
    "wind_speed",
    "wind_dir",
    "air_pressure",
    "tide_level",
    "surface_current",
    "current_dir",
    "depth"
]

target_col = "Hs"

X = df[feature_cols].values
y = df[target_col].values

# 2. 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 特征标准化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 4. 训练 MLP 回归模型
model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_std, y_train)

# 5. 预测
y_pred_train = model.predict(X_train_std)
y_pred_test = model.predict(X_test_std)

# 6. 评价指标
def calc_metrics(y_true, y_pred, name="数据集"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"===== {name}结果 =====")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R^2  = {r2:.4f}\n")
    return mae, rmse, r2

calc_metrics(y_train, y_pred_train, "训练集")
calc_metrics(y_test, y_pred_test, "测试集")

# 7. 画图：测试集真实值 vs 预测值
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_test, s=50, alpha=0.75)
xmin = min(np.min(y_test), np.min(y_pred_test))
xmax = max(np.max(y_test), np.max(y_pred_test))
plt.plot([xmin, xmax], [xmin, xmax], 'r--', linewidth=1.5)
plt.xlabel("True Hs_m")
plt.ylabel("Predicted Hs_m")
plt.title("Test Set: True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. 画图：测试集样本顺序对比
plt.figure(figsize=(10, 5))
plt.plot(y_test, 'b-o', linewidth=1.2, markersize=4, label='True')
plt.plot(y_pred_test, 'r-*', linewidth=1.2, markersize=4, label='Predicted')
plt.xlabel("Test Sample Index")
plt.ylabel("Hs_m")
plt.title("Test Set Prediction Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. 残差图
residuals = y_test - y_pred_test
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_test, residuals, s=45, alpha=0.75)
plt.axhline(0, color='r', linestyle='--', linewidth=1.5)
plt.xlabel("Predicted Hs_m")
plt.ylabel("Residuals")
plt.title("Residual Plot on Test Set")
plt.grid(True)
plt.tight_layout()
plt.show()