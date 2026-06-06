import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

# Chuẩn bị dữ liệu và chọn thuộc tính
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
features = ["alcohol", "malic_acid", "ash", "flavanoids", "proline", "color_intensity"]

# Chuẩn hóa dữ liệu về [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

def star_plot(values, label, ax):
    num_vars = len(values)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    # Khép kín đa giác
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
    ax.plot(angles, values, linewidth=1.5)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=8)
    ax.set_title(label, size=12, color='navy', y=1.1)

# Vẽ Star Glyph cho 3 mẫu đại diện từ 3 lớp khác nhau
fig = plt.figure(figsize=(15, 5))
# Chọn 3 index thuộc 3 class khác nhau của tập Wine: Lớp 0 (idx 0), Lớp 1 (idx 60), Lớp 2 (idx 130)
sample_indices = [0, 60, 130] 

for i, idx in enumerate(sample_indices):
    ax = fig.add_subplot(1, 3, i+1, polar=True)
    star_plot(scaled_data[idx], f"Class {df['target'].iloc[idx]} (Sample {idx})", ax)

plt.tight_layout()
plt.show()