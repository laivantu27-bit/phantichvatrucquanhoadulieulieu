import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Tải và chuẩn bị dữ liệu (chỉ lấy 2000 mẫu để chạy nhanh hơn)
print("Đang tải dữ liệu MNIST, vui lòng đợi vài giây...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='auto')
X = mnist.data[:2000]
y = mnist.target[:2000]

# 2. Áp dụng PCA (Giảm xuống 2 chiều)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(14, 6))

# Vẽ biểu đồ PCA
plt.subplot(1, 2, 1)
scatter_pca = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y.astype(int),
    cmap="tab10",
    s=15,
    alpha=0.8
)
plt.title("PCA Visualization of MNIST")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter_pca, label="Digit Label")

# 3. Áp dụng t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Vẽ biểu đồ t-SNE
plt.subplot(1, 2, 2)
scatter_tsne = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y.astype(int),
    cmap="tab10",
    s=15,
    alpha=0.8
)
plt.title("t-SNE Visualization of MNIST")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(scatter_tsne, label="Digit Label")

plt.tight_layout()
plt.show()