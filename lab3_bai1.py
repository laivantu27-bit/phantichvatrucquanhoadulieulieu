import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Tải dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

# Map target (0, 1, 2) sang tên loài hoa để biểu đồ dễ đọc hơn
df["species_name"] = df["species"].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())

# 2. Vẽ Scatter Plot sử dụng Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species_name",
    palette="Set1",
    s=100 # Tăng kích thước điểm ảnh
)

plt.title("Scatter Plot: Petal Length vs Petal Width (Iris Dataset)")
plt.show()