import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ============================================
# BAI 4: PHAN TICH MOI QUAN HE GIUA CAC THUOC TINH
# ============================================

print("="*70)
print("PHAN TICH MOI QUAN HE GIUA CAC THUOC TINH")
print("="*70)

# 1. TAI DU LIEU
print("\n1. TAI DU LIEU")
print("-" * 70)

# Su dung dataset 'tips' tu seaborn
df = sns.load_dataset('tips')

print("Dataset: Tips (Tip data)")
print(f"Kich thuoc: {df.shape[0]} quan sat, {df.shape[1]} thuoc tinh")
print(f"\nDu lieu 5 hang dau tien:")
print(df.head())

# Chon cac cot so
numeric_df = df.select_dtypes(include=[np.number])
print(f"\nCac thuoc tinh so: {list(numeric_df.columns)}")

# 2. TINH MA TRAN TUONG QUAN (PEARSON)
print("\n" + "="*70)
print("2. TINH MA TRAN TUONG QUAN (PEARSON CORRELATION)")
print("-" * 70)

# Tinh ma tran tuong quan
corr_matrix = numeric_df.corr()

print("\nMa tran tuong quan Pearson:")
print(corr_matrix)

# 3. PHAN TICH TUONG QUAN
print("\n" + "="*70)
print("3. PHAN TICH TUONG QUAN")
print("-" * 70)

print("\nGiai thich do tuong quan:")
print("  [+1.0 to +0.7]: Tuong quan duong rat manh")
print("  [+0.7 to +0.3]: Tuong quan duong manh vua")
print("  [+0.3 to +0.0]: Tuong quan duong yeu")
print("  [+0.0 to -0.3]: Tuong quan am yeu")
print("  [-0.3 to -0.7]: Tuong quan am manh vua")
print("  [-0.7 to -1.0]: Tuong quan am rat manh")

# Tim cac cap bien co tuong quan cao
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > 0.3:  # Nguong tương quan > 0.3
            high_corr_pairs.append({
                'Var1': corr_matrix.columns[i],
                'Var2': corr_matrix.columns[j],
                'Correlation': corr_value
            })

# Sap xep theo gia tri tuong quan giam dan
high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x['Correlation']), reverse=True)

print("\nCac cap bien co tuong quan cao (|r| > 0.3):")
print("-" * 70)
for idx, pair in enumerate(high_corr_pairs, 1):
    var1 = pair['Var1']
    var2 = pair['Var2']
    corr_val = pair['Correlation']
    
    # Phan loai tuong quan
    if corr_val > 0:
        type_str = "Tuong quan DUONG"
    else:
        type_str = "Tuong quan AM"
    
    strength = "Rat manh" if abs(corr_val) > 0.7 else ("Manh vua" if abs(corr_val) > 0.5 else "Yeu vua")
    
    print(f"\n{idx}. {var1} <-> {var2}")
    print(f"    Gia tri: {corr_val:.4f}")
    print(f"    Loai: {type_str} {strength}")
    print(f"    Y nghia: ", end="")
    
    if var1 == 'total_bill' and var2 == 'tip':
        print("Hoa don cao thi tien tip cao - quan tron")
    elif var1 == 'total_bill' and var2 == 'size':
        print("Hoa don cao thi nhom nguoi da - dung logich")
    elif var1 == 'tip' and var2 == 'size':
        print("Nhom da thi tien tip cao - tuong ung voi so luong")
    else:
        print("Hai bien co moi lien quan")

# 4. TINH MA TRAN TUONG QUAN SPEARMAN (RANK-BASED)
print("\n" + "="*70)
print("4. MA TRAN TUONG QUAN SPEARMAN (RANK-BASED)")
print("-" * 70)

spearman_corr = numeric_df.corr(method='spearman')
print("\nMa tran tuong quan Spearman:")
print(spearman_corr)

# 5. VE HEATMAP
print("\n" + "="*70)
print("5. VE HEATMAP - TUONG QUAN PEARSON")
print("-" * 70)

# Heatmap 1: Pearson Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    vmin=-1,
    vmax=1
)
plt.title('Ma tran tuong quan Pearson', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Heatmap 2: Spearman Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(
    spearman_corr,
    annot=True,
    fmt='.3f',
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    vmin=-1,
    vmax=1
)
plt.title('Ma tran tuong quan Spearman', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 6. VE SCATTER PLOT CHO CAC CAP TUONG QUAN CAO
print("\nVe Scatter Plot cho cac cap tuong quan cao...")

if len(high_corr_pairs) > 0:
    n_pairs = min(len(high_corr_pairs), 4)  # Toi da 4 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx in range(n_pairs):
        pair = high_corr_pairs[idx]
        var1 = pair['Var1']
        var2 = pair['Var2']
        corr_val = pair['Correlation']
        
        ax = axes[idx]
        
        # Scatter plot voi trend line
        ax.scatter(df[var1], df[var2], alpha=0.6, s=50, color='blue')
        
        # Them duong hoi quy
        z = np.polyfit(df[var1], df[var2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[var1].min(), df[var1].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend line')
        
        ax.set_xlabel(var1, fontsize=11)
        ax.set_ylabel(var2, fontsize=11)
        ax.set_title(f'{var1} vs {var2}\nr = {corr_val:.4f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # An nhung subplot khong su dung
    for idx in range(n_pairs, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# 7. THONG KE CHI TIET
print("\n" + "="*70)
print("6. THONG KE CHI TIET VE MOI QUAN HE")
print("-" * 70)

for col_i in numeric_df.columns:
    for col_j in numeric_df.columns:
        if col_i < col_j:  # Chi xet cac cap duy nhat
            corr_pearson = corr_matrix.loc[col_i, col_j]
            corr_spearman = spearman_corr.loc[col_i, col_j]
            
            # Tinh p-value cho Pearson
            _, p_value = pearsonr(df[col_i], df[col_j])
            
            if abs(corr_pearson) > 0.3:
                print(f"\n{col_i} <--> {col_j}:")
                print(f"  Pearson:  {corr_pearson:.4f}")
                print(f"  Spearman: {corr_spearman:.4f}")
                print(f"  P-value:  {p_value:.6f}")
                
                if p_value < 0.05:
                    print(f"  [SIGNIFICANT] Co y nghia thong ke o muc 0.05")
                else:
                    print(f"  [NOT SIGNIFICANT] Khong co y nghia thong ke")

# 8. KET LUAN
print("\n" + "="*70)
print("7. KET LUAN")
print("-" * 70)

print(f"""
Phan tich moi quan he giua cac thuoc tinh:

1. TUONG QUAN MANH:
   - Cac cap bien co tuong quan cao (|r| > 0.7): {len([p for p in high_corr_pairs if abs(p['Correlation']) > 0.7])} cap
   - Cac cap bien co tuong quan vua (|r| > 0.5): {len([p for p in high_corr_pairs if 0.5 < abs(p['Correlation']) <= 0.7])} cap
   - Cac cap bien co tuong quan yeu (0.3 < |r| <= 0.5): {len([p for p in high_corr_pairs if 0.3 < abs(p['Correlation']) <= 0.5])} cap

2. Y NGHIA:
   - Tuong quan duong: Hai bien tang (hoac giam) cung luc
   - Tuong quan am: Mot bien tang thi bien kia giam
   - Tuong quan 0: Hai bien khong co moi lien quan

3. CANH BAO:
   - Tuong quan khong co nghia la nhan qua
   - Can xem xet them cac yeu to khac
   - Nen su dung Spearman cho du lieu khong phuong thuong

4. AP DUNG:
   - Loai bo cac bien co tuong quan rat cao trong mo hinh ML
   - Dung thong tin tuong quan de chon feature
   - Thu tuc lai du lieu de kiem chung
""")

print("="*70)
print("[DONE] Phan tich hoan tat!")
print("="*70)