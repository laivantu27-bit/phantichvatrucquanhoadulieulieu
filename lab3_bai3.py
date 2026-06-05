import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler

# 1. Chuẩn bị dữ liệu
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["class"] = wine.target

# 2. Chọn 5 thuộc tính
features = ["alcohol", "malic_acid", "ash", "flavanoids", "proline"]
df_plot = df[features + ["class"]]

# 3. Chuẩn hóa dữ liệu (đưa về khoảng 0 - 1 để dễ quan sát trên cùng 1 trục tung)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_plot[features])

df_scaled = pd.DataFrame(scaled_features, columns=features)
df_scaled["class"] = df_plot["class"]

# 4. Vẽ Parallel Coordinates
plt.figure(figsize=(10, 6))
parallel_coordinates(
    df_scaled,
    "class",
    colormap=plt.cm.Set1,
    linewidth=1.5,
    alpha=0.7 # Thêm độ trong suốt để dễ nhìn các đường chồng chéo
)

plt.title("Parallel Coordinates Plot - Wine Dataset (Scaled)")
plt.xlabel("Features")
plt.ylabel("Normalized Value")
plt.legend(loc='upper right', title="Class")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()