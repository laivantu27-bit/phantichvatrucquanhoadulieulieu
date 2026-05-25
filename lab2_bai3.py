import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# BÀI 3: PHÁT HIỆN OUTLIERS BẰNG BOXPLOT
# ============================================

print("="*70)
print("PHÁT HIỆN OUTLIERS BẰNG BOXPLOT")
print("="*70)

# 1. CHỌN VÀ TẢI DATASET
print("\n1. TẢI DỮ LIỆU")
print("-" * 70)

# Sử dụng dataset 'iris' từ seaborn - bộ dữ liệu nổi tiếng trong ML
df = sns.load_dataset('iris')

print(f"Dataset: Iris (Hoa Iris)")
print(f"Kích thước: {df.shape[0]} quan sát, {df.shape[1]} thuộc tính")
print(f"\nDữ liệu 5 hàng đầu tiên:")
print(df.head())

print(f"\nThông tin về dataset:")
print(df.info())

print(f"\nThống kê mô tả:")
print(df.describe())

# 2. CHỌN CÁC THUỘC TÍNH SỐ
print("\n" + "="*70)
print("2. CHỌN CÁC THUỘC TÍNH SỐ")
print("-" * 70)

# Lấy tất cả cột số
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Các thuộc tính số được chọn: {numeric_columns}")

# 3. PHÁT HIỆN OUTLIERS BẰNG PHƯƠNG PHÁP IQR
print("\n" + "="*70)
print("3. PHÁT HIỆN OUTLIERS BẰNG PHƯƠNG PHÁP IQR")
print("-" * 70)
print("\nPhương pháp IQR (Interquartile Range):")
print("- Q1 = 25th percentile (tứ phân vị thứ 1)")
print("- Q3 = 75th percentile (tứ phân vị thứ 3)")
print("- IQR = Q3 - Q1")
print("- Lower Bound = Q1 - 1.5 * IQR")
print("- Upper Bound = Q3 + 1.5 * IQR")
print("- Outliers: giá trị < Lower Bound hoặc > Upper Bound\n")

outlier_info = {}

for col in numeric_columns:
    print(f"\n{col.upper()}:")
    print("-" * 50)
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Tìm outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"  Min: {df[col].min():.3f}")
    print(f"  Q1 (25%): {Q1:.3f}")
    print(f"  Median (50%): {df[col].median():.3f}")
    print(f"  Q3 (75%): {Q3:.3f}")
    print(f"  Max: {df[col].max():.3f}")
    print(f"  IQR: {IQR:.3f}")
    print(f"  Lower Bound: {lower_bound:.3f}")
    print(f"  Upper Bound: {upper_bound:.3f}")
    print(f"  Số lượng outliers: {len(outliers)}")
    
    if len(outliers) > 0:
        print(f"  Giá trị outliers:")
        for idx, val in outliers[col].items():
            print(f"    - Row {idx}: {val:.3f}")
    else:
        print(f"  [OK] Không có outliers trong thuộc tính này")
    
    outlier_info[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers,
        'count': len(outliers)
    }

# 4. VẼ BOXPLOTS
print("\n" + "="*70)
print("4. VẼ BOXPLOTS")
print("-" * 70)

# Boxplot 1: Từng thuộc tính riêng biệt
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_columns):
    ax = axes[idx]
    
    # Vẽ boxplot
    bp = ax.boxplot(df[col], vert=True, patch_artist=True, widths=0.5)
    
    # Tô màu cho boxplot
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    # Tô màu cho outliers
    if len(outlier_info[col]['outliers']) > 0:
        for outlier in outlier_info[col]['outliers'][col]:
            ax.plot(1, outlier, 'ro', markersize=8, label='Outlier' if outlier == outlier_info[col]['outliers'][col].iloc[0] else '')
    
    ax.set_ylabel('Giá trị', fontsize=11)
    ax.set_title(f'Boxplot: {col}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Thêm thông tin lên biểu đồ
    info_text = f"Q1={outlier_info[col]['Q1']:.2f}\nQ3={outlier_info[col]['Q3']:.2f}\nIQR={outlier_info[col]['IQR']:.2f}"
    ax.text(1.15, df[col].mean(), info_text, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.suptitle('Boxplot cho từng thuộc tính', fontsize=14, fontweight='bold', y=1.00)
plt.show()

# Boxplot 2: Tất cả thuộc tính trên một hình
fig, ax = plt.subplots(figsize=(12, 7))

# Chuẩn hóa dữ liệu để so sánh
df_normalized = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()

bp = ax.boxplot([df_normalized[col] for col in numeric_columns], 
                  labels=numeric_columns,
                  patch_artist=True,
                  widths=0.6)

# Tô màu cho các boxes
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Làm nổi bật median lines
for median in bp['medians']:
    median.set(color='red', linewidth=2)

ax.set_ylabel('Giá trị chuẩn hóa (Standardized)', fontsize=12)
ax.set_title('So sánh Boxplot - Tất cả thuộc tính (dữ liệu chuẩn hóa)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 5. VIZ - VIOLIN PLOT (để so sánh phân phối)
print("\nVẽ Violin Plot (để so sánh phân phối)")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, col in enumerate(numeric_columns):
    ax = axes[idx]
    
    sns.violinplot(data=df, y=col, ax=ax, color='skyblue')
    sns.stripplot(data=df, y=col, ax=ax, color='red', alpha=0.5, size=6)
    
    ax.set_title(f'Violin Plot: {col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá trị', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.suptitle('Violin Plot - Phân phối dữ liệu cho từng thuộc tính', fontsize=14, fontweight='bold', y=1.00)
plt.show()

# 6. PHÂN TÍCH VÀ GIẢI THÍCH OUTLIERS
print("\n" + "="*70)
print("5. PHÂN TÍCH VÀ GIẢI THÍCH OUTLIERS")
print("-" * 70)

print(f"""
PHÂN TÍCH KẾT QUẢ:

1. TỔNG QUAN VỀ OUTLIERS:
   - Outliers là các giá trị khác biệt đáng kể so với phần còn lại của dữ liệu
   - Chúng có thể là lỗi đo lường hoặc các giá trị thực sự bất thường
   
2. PHƯƠNG PHÁP PHÁT HIỆN:
   - Sử dụng phương pháp IQR (Interquartile Range)
   - Định nghĩa: Giá trị nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
   - Phương pháp này dựa trên thống kê và phổ biến trong thực tế
   
3. CHI TIẾT OUTLIERS:
""")

total_outliers = 0
for col in numeric_columns:
    count = outlier_info[col]['count']
    total_outliers += count
    percentage = (count / len(df)) * 100
    
    if count > 0:
        print(f"\n   {col}:")
        print(f"   - Số lượng: {count} ({percentage:.1f}%)")
        print(f"   - Phạm vi hợp lệ: [{outlier_info[col]['lower_bound']:.3f}, {outlier_info[col]['upper_bound']:.3f}]")
        print(f"   - Giá trị outliers: {outlier_info[col]['outliers'][col].values.round(3).tolist()}")
    else:
        print(f"\n   {col}: [OK] Không có outliers")

print(f"\n4. TỔNG CỘNG:")
print(f"   - Tổng số outliers: {total_outliers}")
print(f"   - Tỷ lệ: {(total_outliers / (len(df) * len(numeric_columns))) * 100:.2f}%")

print(f"\n5. GỢI Ý XỬ LÝ OUTLIERS:")
print(f"""
   a) Kiểm tra dữ liệu:
      - Xác minh outliers có phải là lỗi hay không
      - Kiểm tra nguồn dữ liệu gốc
      
   b) Các phương pháp xử lý:
      - Loại bỏ: Nếu là lỗi đo lường
      - Giữ lại: Nếu là dữ liệu hợp lệ
      - Chuyển đổi: Sử dụng log/sqrt để giảm ảnh hưởng
      - Chuẩn hóa: Sử dụng robust scaling
      
   c) Tác động:
      - Outliers có thể ảnh hưởng đến trung bình
      - Ít ảnh hưởng đến trung vị (median)
      - Có thể làm tăng độ lệch chuẩn
""")

print("\n" + "="*70)
print("[DONE] Phan tich hoan tat!")
print("="*70)

# 7. THÊM CHỈ SỐ THỐNG KÊ
print("\n" + "="*70)
print("6. CHỈ SỐ THỐNG KÊ BỔ SUNG")
print("-" * 70)

for col in numeric_columns:
    print(f"\n{col}:")
    print(f"  Skewness (Độ lệch): {df[col].skew():.3f}")
    print(f"  Kurtosis (Độ nhọn): {df[col].kurtosis():.3f}")
    
    skew_interp = "Cân đối" if abs(df[col].skew()) < 0.5 else ("Lệch trái" if df[col].skew() < 0 else "Lệch phải")
    kurt_interp = "Bình thường" if abs(df[col].kurtosis()) < 3 else ("Nhọn hơn" if df[col].kurtosis() > 3 else "Dẹt hơn")
    
    print(f"  Giải thích: {skew_interp}, {kurt_interp}")
