import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df_before = df.copy()

print("\n--- Số lượng dữ liệu bị thiếu ở mỗi cột ---")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df = df.dropna(subset=['Embarked'])

print(f"\nSố lượng bản ghi trùng lặp trước khi xóa: {df.duplicated().sum()}\n")

df.drop_duplicates(inplace=True)
print(f"Số lượng bản ghi trùng lặp sau khi xóa: {df.duplicated().sum()}")

# Chuẩn hóa dữ liệu số
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.boxplot(y=df_before['Age'], color='skyblue')
plt.title('Age - Trước làm sạch')

plt.subplot(2, 2, 2)
sns.boxplot(y=df_before['Fare'], color='salmon')
plt.title('Fare - Trước làm sạch')

plt.subplot(2, 2, 3)
sns.boxplot(y=df_scaled['Age'], color='lightgreen')
plt.title('Age - Sau chuẩn hóa và làm sạch')

plt.subplot(2, 2, 4)
sns.boxplot(y=df_scaled['Fare'], color='gold')
plt.title('Fare - Sau chuẩn hóa và làm sạch')

plt.tight_layout()

os.makedirs('plots', exist_ok=True)
plot_path = os.path.join('plots', 'lab1_bai2_boxplots.png')
plt.savefig(plot_path)
print(f"Biểu đồ đã được lưu tại: {plot_path}")

try:
    plt.show()
except Exception:
    print("Không thể hiển thị biểu đồ trực tiếp trong môi trường này, bạn có thể mở file ảnh đã lưu.")

print("\n--- Tổng kết ---")
print(f"Số bản ghi gốc: {len(df_before)}")
print(f"Số bản ghi sau làm sạch: {len(df)}")
print("Các cột số đã chuẩn hóa:", numeric_cols)
