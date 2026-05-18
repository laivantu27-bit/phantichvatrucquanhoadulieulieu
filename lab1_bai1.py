import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("--- 10 dòng đầu tiên ---")
print(df.head(10)) # [cite: 7]

print(f"\nSố lượng bản ghi (rows): {df.shape[0]}") # [cite: 7]
print(f"Số lượng thuộc tính (columns): {df.shape[1]}") # [cite: 7]

print("\n--- Kiểu dữ liệu ---")
print(df.dtypes) # [cite: 8]

print("\n--- Thống kê cơ bản ---")
print(df.describe()) 

print("\n--- Mean/Min/Max/Std cho các cột số ---")
num_cols = df.select_dtypes(include=[np.number]).columns
print(df[num_cols].agg(['mean', 'min', 'max', 'std']).T)

import os
os.makedirs('plots', exist_ok=True)

print("\n--- Lưu histogram cho các cột số ---")
for col in num_cols:
	try:
		plt.figure(figsize=(8, 4))
		sns.histplot(df[col].dropna(), kde=True)
		plt.title(f'Histogram phân bố {col}')
		plt.tight_layout()
		out = f'plots/hist_{col}.png'
		plt.savefig(out)
		plt.close()
		print(f'Saved {out}')
	except Exception as e:
		print(f'Không thể vẽ histogram cho {col}: {e}')

print("\n--- Lưu bar chart cho các cột phân loại ---")
cat_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].nunique() <= 10]
for col in cat_cols:
	try:
		plt.figure(figsize=(8, 4))
		sns.countplot(data=df, x=col)
		plt.title(f'Bar chart phân phối {col}')
		plt.xticks(rotation=45)
		plt.tight_layout()
		out = f'plots/bar_{col}.png'
		plt.savefig(out)
		plt.close()
		print(f'Saved {out}')
	except Exception as e:
		print(f'Không thể vẽ bar chart cho {col}: {e}')