import seaborn as sns
import matplotlib.pyplot as plt

# Ma trận tương quan
plt.figure(figsize=(8, 6))
sns.heatmap(df[['study_time', 'completion_rate', 'video_watched', 'purchase_flag', 'ai_clicks']].corr(), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan giữa các hành vi")
plt.show()

# Scatter plot: Thời gian học vs Tỷ lệ hoàn thành
sns.scatterplot(data=df, x='study_time', y='completion_rate', hue='purchase_flag')
plt.title("Thời gian học và Tỷ lệ hoàn thành")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Lựa chọn các features quan trọng
features = ['study_time', 'completion_rate', 'quiz_taken', 'login_count']
X = df[features]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chạy K-Means với 4 cụm
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Gán nhãn cho các cụm (Dựa trên việc kiểm tra giá trị trung bình - centroids)
# Mã giả: 0: Power, 1: Casual, 2: Hunters, 3: Passive   

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("User Behavior Dendrogram")
# Lấy một mẫu nhỏ (vd 500 users) để vẽ cho dễ nhìn
dend = shc.dendrogram(shc.linkage(X_scaled[:500], method='ward'))
plt.show()