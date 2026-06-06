import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# 1. Chọn dataset Iris (có đúng 4 thuộc tính, rất phù hợp với 4 đặc điểm khuôn mặt)
iris = load_iris()
scaler = MinMaxScaler()
samples = scaler.fit_transform(iris.data[:6]) # Chọn 6 mẫu đầu tiên để vẽ

def draw_face(ax, data):
    # Ánh xạ thuộc tính (đã chuẩn hóa [0,1]) vào kích thước
    face_size = 0.5 + data[0] * 0.5   # Đặc trưng 0 -> Cỡ mặt
    eye_size = 0.02 + data[1] * 0.05  # Đặc trưng 1 -> Cỡ mắt (thu nhỏ hệ số để mắt không tràn)
    mouth_curve = data[2] - 0.5       # Đặc trưng 2 -> Độ cong miệng
    nose_size = 0.02 + data[3] * 0.05 # Đặc trưng 3 -> Cỡ mũi
    
    # Vẽ khuôn mặt
    face = plt.Circle((0.5, 0.5), face_size * 0.4, fill=False, linewidth=2)
    ax.add_patch(face)
    
    # Vẽ mắt
    left_eye = plt.Circle((0.35, 0.6), eye_size, color="black")
    right_eye = plt.Circle((0.65, 0.6), eye_size, color="black")
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    
    # Vẽ mũi
    nose = plt.Circle((0.5, 0.5), nose_size, color="black")
    ax.add_patch(nose)
    
    # Vẽ miệng
    x = np.linspace(0.35, 0.65, 100)
    # Điều chỉnh lại công thức miệng để nó nằm gọn trong khuôn mặt
    y = 0.35 + mouth_curve * (x - 0.5)**2 * -4 
    ax.plot(x, y, linewidth=2, color="black")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis("off")

# 2. Tạo nhiều khuôn mặt đại diện cho các bản ghi dữ liệu
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for i, ax in enumerate(axes.flat):
    draw_face(ax, samples[i])
    ax.set_title(f"Sample {i+1}")

plt.suptitle("Chernoff Faces (Iris Dataset Mapping)")
plt.show()