from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Tải dataset Wine
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine["target"] = wine.target

# Chọn ra khoảng 5 thuộc tính tiêu biểu để biểu đồ Pairplot không bị quá dày đặc và dễ nhìn
selected_features = ["alcohol", "malic_acid", "ash", "flavanoids", "color_intensity", "target"]
df_subset = df_wine[selected_features]

# 2. Vẽ Scatter Matrix (Pairplot)
sns.pairplot(
    df_subset,
    hue="target",
    diag_kind="kde",
    palette="Set2"
)
plt.suptitle("Pairplot of Selected Wine Features", y=1.02)
plt.show()