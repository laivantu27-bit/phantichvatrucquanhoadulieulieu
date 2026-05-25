import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# BÀI 1: PHÂN TÍCH PHÂN PHỐI DỮ LIỆU
# ============================================

# 1. TẢI DATASET VÀ LỰA CHỌN DỮ LIỆU
print("="*60)
print("PHÂN TÍCH PHÂN PHỐI DỮ LIỆU")
print("="*60)

# Tải dataset mẫu từ Seaborn 
df = sns.load_dataset('tips')

# Chọn 3 thuộc tính số: 
# total_bill (tổng hóa đơn), tip (tiền tip), size (số lượng người)
features = ['total_bill', 'tip', 'size']

print(f"\nDataset: Tips (244 observations)")
print(f"Thuộc tính đã chọn: {features}")
print(f"Kích thước dataset: {df.shape}")

# 2. THỐNG KÊ MÔ TẢ
print("\n" + "="*60)
print("THỐNG KÊ MÔ TẢ CỦA CÁC THUỘC TÍNH")
print("="*60)

for feature in features:
    print(f"\n{feature.upper()}:")
    print(f"  Mean (Trung bình):     {df[feature].mean():.2f}")
    print(f"  Median (Trung vị):     {df[feature].median():.2f}")
    print(f"  Std Dev (Độ lệch chuẩn): {df[feature].std():.2f}")
    print(f"  Min:                   {df[feature].min():.2f}")
    print(f"  Max:                   {df[feature].max():.2f}")
    print(f"  Skewness (Độ lệch):    {df[feature].skew():.3f}")
    print(f"  Kurtosis:              {df[feature].kurtosis():.3f}")

# 3. VẼ BIỂU ĐỒ HISTOGRAM
print("\n" + "="*60)
print("VẼ BIỂU ĐỒ HISTOGRAM")
print("="*60)

plt.figure(figsize=(15, 4))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    # Vẽ histogram với 20 bins theo yêu cầu
    plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram: {feature}', fontsize=12, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel('Tần suất')
    plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 4. VẼ BIỂU ĐỒ DENSITY PLOT (KDE)
print("\nVẼ BIỂU ĐỒ DENSITY PLOT (KDE)")

plt.figure(figsize=(15, 4))
for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    # Vẽ KDE plot có fill màu
    sns.kdeplot(df[feature], label=feature, fill=True, color='orange', alpha=0.6)
    plt.title(f'KDE Plot: {feature}', fontsize=12, fontweight='bold')
    plt.xlabel(feature)
    plt.ylabel('Mật độ xác suất')
    plt.legend()
    plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5. VẼ BIỂU ĐỒ BOXPLOT
print("\nVẼ BIỂU ĐỒ BOXPLOT")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[features], palette='Set2', width=0.6)
plt.title('Boxplot của 3 thuộc tính: total_bill, tip, size', fontsize=14, fontweight='bold')
plt.ylabel('Giá trị')
plt.grid(axis='y', alpha=0.3)
plt.show()

# 6. PHÂN TÍCH VÀ NHẬN XÉT
print("\n" + "="*60)
print("PHÂN TÍCH ĐẶC ĐIỂM PHÂN PHỐI DỮ LIỆU")
print("="*60)

print("\n1. TOTAL_BILL (Tổng hóa đơn):")
print("   - Phân phối: Lệch phải (Right-skewed) - Skewness = {:.3f}".format(df['total_bill'].skew()))
print("   - Đặc điểm: Hầu hết giá trị tập trung ở phía trái, có một vài giá trị cao")
print("   - Ý nghĩa: Phần lớn hóa đơn có giá trị thấp, ít người chi tiêu cao")

print("\n2. TIP (Tiền tip):")
print("   - Phân phối: Lệch phải (Right-skewed) - Skewness = {:.3f}".format(df['tip'].skew()))
print("   - Đặc điểm: Tập trung quanh các giá trị thấp, có outliers phía trên")
print("   - Ý nghĩa: Tiền tip thường nhỏ, ít người cho tip lớn")

print("\n3. SIZE (Số lượng người):")
print("   - Phân phối: Lệch phải (Right-skewed) - Skewness = {:.3f}".format(df['size'].skew()))
print("   - Đặc điểm: Phần lớn bàn có 2-3 người, ít bàn có nhiều người")
print("   - Ý nghĩa: Khách hàng thường đến ăn theo nhóm nhỏ")

print("\n" + "="*60)
print("KẾT LUẬN: Cả ba thuộc tính đều có phân phối lệch phải,")
print("cho thấy các giá trị cao là ngoại lệ, không phổ biến.")
print("="*60)