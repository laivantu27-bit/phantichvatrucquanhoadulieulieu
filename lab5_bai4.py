import tensorflow as tf # [cite: 330]
from tensorflow.keras.models import Sequential # [cite: 331]
from tensorflow.keras.layers import Dense, Input # [cite: 331]
from tensorflow.keras.utils import plot_model # [cite: 331]

# 1. Xây dựng mô hình mạng nơ-ron 
model = Sequential([ # [cite: 332]
    Input(shape=(4,)),           # 4 đặc trưng đầu vào (Ví dụ: 4 thuộc tính của Iris) 
    Dense(8, activation='relu'), # Hidden layer 1: 8 nơ-ron [cite: 336, 337]
    Dense(16, activation='relu'),# Hidden layer 2: 16 nơ-ron [cite: 338, 339]
    Dense(3, activation='softmax') # Output layer: 3 lớp (Ví dụ: 3 loại hoa) 
])

model.compile(
    optimizer='adam', # [cite: 344]
    loss='categorical_crossentropy', # [cite: 345]
    metrics=['accuracy'] # [cite: 346]
)

# 2. In cấu trúc mạng dạng text
model.summary() # [cite: 347]

# 3. Trực quan hóa các lớp bằng hình ảnh (Yêu cầu có pydot và graphviz) 
plot_model(
    model, # [cite: 349]
    to_file='model_architecture.png', 
    show_shapes=True, # [cite: 350]
    show_layer_names=True # [cite: 351]
)