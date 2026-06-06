import numpy as np
import matplotlib.pyplot as plt

# 1. Chuẩn bị dataset có nhiều bản ghi (Mô phỏng dữ liệu lớn)
# Tạo 10,000 điểm dữ liệu từ một hàm sóng để tạo pattern
x = np.linspace(0, 100, 10000)
values = np.sin(x) + np.random.normal(0, 0.1, 10000)

# 2. Xây dựng ma trận pixel
size = int(np.ceil(np.sqrt(len(values)))) # Tính kích thước ma trận vuông
pixel_matrix = np.zeros(size * size)
pixel_matrix[:len(values)] = values
pixel_matrix = pixel_matrix.reshape(size, size)

# 3. Sử dụng màu sắc để biểu diễn giá trị dữ liệu
plt.figure(figsize=(6, 6))
plt.imshow(pixel_matrix, cmap="viridis", aspect='auto')
plt.colorbar(label="Value Magnitude")
plt.title("Pixel-based Visualization (Sine Wave + Noise)")
plt.show()