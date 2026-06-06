import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Chuẩn bị dữ liệu
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# 1. Tính ma trận tương quan
corr_matrix = df.corr()

# 2. Sử dụng thư viện Seaborn để vẽ Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=False,      # Đặt False để biểu đồ đỡ rối, bạn có thể đổi thành True nếu muốn xem số
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap - Wine Dataset")
plt.show()