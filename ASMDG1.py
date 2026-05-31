import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

sns.set(style='whitegrid', palette='muted')
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.unicode_minus': False,
})


def generate_demo_data(n_users=500, random_state=42):
    np.random.seed(random_state)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 5, 1)
    signup_dates = pd.to_datetime(
        np.random.choice(pd.date_range(start_date, end_date, freq='D'), size=n_users)
    )

    sessions_per_week = np.maximum(0, np.round(np.random.normal(3.3, 2.5, size=n_users)).astype(int))
    avg_session_minutes = np.clip(np.random.normal(35, 15, size=n_users), 5, 120)
    weeks_active = np.clip(np.random.poisson(8, size=n_users), 1, 40)
    total_learning_minutes = np.round(sessions_per_week * avg_session_minutes * weeks_active).astype(int)

    courses_enrolled = np.clip(
        np.round(np.random.normal(3.1, 1.6, size=n_users)).astype(int), 1, 12
    )
    courses_completed = np.minimum(
        courses_enrolled,
        np.maximum(0, np.round(np.random.normal(courses_enrolled * 0.7, 1.1, size=n_users)).astype(int)),
    )
    completion_rate = np.round(courses_completed / courses_enrolled, 3)

    advanced_course_purchase = np.where(
        (completion_rate > 0.7) & (sessions_per_week > 2),
        np.random.binomial(1, 0.55, size=n_users),
        np.random.binomial(1, 0.15, size=n_users),
    )
    purchase_amount = advanced_course_purchase * np.round(np.clip(np.random.normal(175, 65, size=n_users), 40, 400), 2)

    days_since_signup = (datetime(2024, 5, 1) - signup_dates).days
    days_since_signup = np.maximum(days_since_signup, 1)
    churned = np.where(
        (sessions_per_week == 0) | (weeks_active < 3) | (completion_rate < 0.3),
        np.random.binomial(1, 0.45, size=n_users),
        np.random.binomial(1, 0.12, size=n_users),
    )
    churned = np.where(days_since_signup < 21, 0, churned)

    last_activity_date = signup_dates + pd.to_timedelta(np.clip(weeks_active * 7 - np.random.randint(0, 7, size=n_users), 1, 280), unit='D')
    last_activity_date = last_activity_date.where(last_activity_date <= datetime(2024, 5, 1), datetime(2024, 5, 1))

    recommendations_used = np.clip(np.round(np.random.normal(5, 3, size=n_users)).astype(int), 0, 20)
    tests_completed = np.clip(np.round(np.random.normal(6, 4, size=n_users)).astype(int), 0, 30)
    avg_quiz_score = np.round(np.clip(np.random.normal(78, 12, size=n_users), 45, 100), 1)

    df = pd.DataFrame(
        {
            'user_id': np.arange(1, n_users + 1),
            'signup_date': signup_dates,
            'last_activity_date': last_activity_date,
            'weeks_active': weeks_active,
            'sessions_per_week': sessions_per_week,
            'avg_session_minutes': np.round(avg_session_minutes, 1),
            'total_learning_minutes': total_learning_minutes,
            'courses_enrolled': courses_enrolled,
            'courses_completed': courses_completed,
            'completion_rate': completion_rate,
            'advanced_course_purchase': advanced_course_purchase,
            'purchase_amount': purchase_amount,
            'churned': churned,
            'recommendations_used': recommendations_used,
            'tests_completed': tests_completed,
            'avg_quiz_score': avg_quiz_score,
        }
    )
    df['learning_days'] = np.clip(np.round(df['total_learning_minutes'] / df['avg_session_minutes']).astype(int), 0, 500)
    df['is_power_user'] = np.where((df['sessions_per_week'] >= 6) & (df['total_learning_minutes'] >= 1500), 1, 0)
    return df


def load_data(path, allow_fallback=False):
    if not os.path.exists(path):
        if allow_fallback:
            print(f"Không tìm thấy file dữ liệu: {path}. Sẽ sử dụng dữ liệu demo thay thế.")
            return None
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    df = pd.read_csv(path)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def summarize_data(df, data_path=None):
    print("\n--- TỔNG QUAN DỮ LIỆU ---")
    if data_path is not None:
        print(f"Nguồn dữ liệu: {data_path}")
    print(f"Số bản ghi: {len(df):,}")
    print(f"Số thuộc tính: {len(df.columns)}")
    print("\nCác thuộc tính chính:")
    print(df.columns.tolist())

    print("\nThông tin kiểu dữ liệu:")
    print(df.dtypes)
    print("\nCác giá trị thiếu theo cột:")
    print(df.isna().sum().sort_values(ascending=False).head(15))
    print("\nCác giá trị trùng lặp:")
    print(f"  Số lượng bản ghi trùng lặp: {df.duplicated().sum()}")

    print("\nThống kê mô tả cho các thuộc tính số:")
    desc = df.select_dtypes(include=[np.number]).describe().T
    # Format numeric columns for clearer printing
    pd.options.display.float_format = '{:,.3f}'.format
    print(desc)
    pd.reset_option('display.float_format')


def clean_data(df):
    df = df.drop_duplicates().copy()

    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
    if 'last_activity_date' in df.columns:
        df['last_activity_date'] = pd.to_datetime(df['last_activity_date'], errors='coerce')

    if 'completion_rate' not in df.columns and {'courses_completed', 'courses_enrolled'}.issubset(df.columns):
        df['completion_rate'] = df['courses_completed'] / df['courses_enrolled'].replace({0: np.nan})

    if 'purchase_amount' not in df.columns and 'advanced_course_purchase' in df.columns:
        df['purchase_amount'] = np.where(df['advanced_course_purchase'] == 1, np.nan, 0.0)

    if 'churned' not in df.columns and 'last_activity_date' in df.columns and 'signup_date' in df.columns:
        df['inactive_days'] = (pd.to_datetime('2024-05-01') - df['last_activity_date']).dt.days
        df['churned'] = np.where(df['inactive_days'] > 30, 1, 0)

    if 'days_since_signup' not in df.columns and 'signup_date' in df.columns:
        df['days_since_signup'] = (pd.to_datetime('2024-05-01') - df['signup_date']).dt.days

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_distribution(df, column, title, xlabel, bins=30, output_dir=None):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, color='steelblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Số lượng người dùng')
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"dist_{column}.png"))
    plt.show()


def plot_time_series(df, date_col, value_col, freq='W', title=None, output_dir=None):
    if date_col not in df.columns:
        return

    ts = df.set_index(date_col).resample(freq)[value_col].mean().rename(value_col)
    plt.figure(figsize=(12, 5))
    ts.plot(marker='o', linewidth=2)
    plt.title(title or f"Xu hướng trung bình {value_col} theo thời gian")
    plt.xlabel('Thời gian')
    plt.ylabel(value_col)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"ts_{value_col}.png"))
    plt.show()


def detect_outliers(df, columns):
    outlier_report = {}
    for column in columns:
        if column not in df.columns:
            continue
        series = df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        outlier_report[column] = {
            'count': len(outliers),
            'lower_bound': lower,
            'upper_bound': upper,
            'examples': outliers.head(10),
        }
        print(f"\nOutliers cho {column}:")
        print(f"  Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")
        print(f"  Số lượng outliers: {len(outliers)}")
    return outlier_report


def report_behavioral_groups(df):
    if {'sessions_per_week', 'completion_rate', 'purchase_amount'}.issubset(df.columns):
        features = df[['sessions_per_week', 'completion_rate', 'purchase_amount']].fillna(0)
        score = (
            features['sessions_per_week'] * 0.4
            + features['completion_rate'] * 0.4
            + (features['purchase_amount'] / (features['purchase_amount'].max() + 1)) * 0.2
        )
        df['engagement_score'] = score
        df['behavior_segment'] = pd.qcut(score, q=4, labels=['Thấp', 'Trung bình', 'Cao', 'Rất cao'])
        print('\nPhân nhóm hành vi người dùng bằng engagement_score:')
        print(df.groupby('behavior_segment').agg(
            users=('user_id', 'count'),
            avg_sessions=('sessions_per_week', 'mean'),
            avg_completion=('completion_rate', 'mean'),
            purchase_rate=('advanced_course_purchase', 'mean') if 'advanced_course_purchase' in df.columns else ('purchase_amount', 'mean')
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EDA LearnX - khám phá hành vi người dùng và phát hiện outliers'
    )
    parser.add_argument(
        '--data-path',
        default='learnx_user_behavior.csv',
        help='Đường dẫn tới file dữ liệu người dùng LearnX (CSV).',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Sử dụng dữ liệu mẫu giả lập nếu không có file thực tế.',
    )
    parser.add_argument(
        '--output-dir',
        default='eda_outputs',
        help='Thư mục lưu ảnh biểu đồ và báo cáo.',
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.demo:
        print('Sử dụng dữ liệu giả lập để minh họa phân tích.\n')
        df = generate_demo_data(500)
    else:
        df = load_data(args.data_path, allow_fallback=True)
        if df is None:
            print('\nDữ liệu mặc định không tồn tại. Sử dụng dataset demo thay thế.\n')
            df = generate_demo_data(500)

    df = clean_data(df)
    summarize_data(df, args.data_path if not args.demo else 'Demo dataset')

    print('\n--- KIỂM TRA DỮ LIỆU THIẾU ---')
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0])

    print('\n--- PHÂN TÍCH BẢN GHI TRÙNG ---')
    print(f"Số bản ghi trùng lặp sau khi loại bỏ: {df.duplicated().sum()}")

    numeric_cols = [
        'total_learning_minutes',
        'sessions_per_week',
        'avg_session_minutes',
        'courses_enrolled',
        'courses_completed',
        'completion_rate',
        'purchase_amount',
        'recommendations_used',
        'tests_completed',
        'avg_quiz_score',
    ]
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    print('\n--- VẼ BIỂU ĐỒ PHÂN PHỐI ---')
    for col in ['total_learning_minutes', 'sessions_per_week', 'completion_rate', 'purchase_amount']:
        if col in df.columns:
            plot_distribution(
                df,
                col,
                title=f'Phân phối {col.replace("_", " ").capitalize()}',
                xlabel=col.replace('_', ' ').capitalize(),
                output_dir=args.output_dir,
            )

    print('\n--- PHÂN TÍCH XU HƯỚNG THỜI GIAN ---')
    if 'signup_date' in df.columns:
        plot_time_series(
            df,
            date_col='signup_date',
            value_col='sessions_per_week',
            freq='W',
            title='Xu hướng Sessions per Week theo thời gian đăng ký',
            output_dir=args.output_dir,
        )
    if 'last_activity_date' in df.columns and 'total_learning_minutes' in df.columns:
        plot_time_series(
            df,
            date_col='last_activity_date',
            value_col='total_learning_minutes',
            freq='W',
            title='Xu hướng tăng thời gian học theo tuần hoạt động cuối',
            output_dir=args.output_dir,
        )

    print('\n--- PHÁT HIỆN OUTLIERS ---')
    detect_outliers(df, numeric_cols)

    print('\n--- HÀNH VI NGƯỜI DÙNG ĐẶC BIỆT ---')
    if 'total_learning_minutes' in df.columns:
        super_learners = df.nlargest(10, 'total_learning_minutes')
        print('\nNgười dùng học rất tích cực (Top 10 theo tổng phút học):')
        print(super_learners[['user_id', 'total_learning_minutes', 'sessions_per_week', 'completion_rate', 'advanced_course_purchase']].to_string(index=False))

    if {'courses_enrolled', 'total_learning_minutes'}.issubset(df.columns):
        dormant_learners = df[(df['courses_enrolled'] >= 3) & (df['total_learning_minutes'] == 0)]
        print(f"\nNgười dùng đăng ký nhiều khóa nhưng không học: {len(dormant_learners)} người")
        if len(dormant_learners) > 0:
            print(dormant_learners[['user_id', 'courses_enrolled', 'courses_completed', 'completion_rate']].head(10).to_string(index=False))

    if 'purchase_amount' in df.columns:
        big_spenders = df.nlargest(10, 'purchase_amount')
        print('\nNgười dùng chi tiêu bất thường (Top 10 theo purchase_amount):')
        print(big_spenders[['user_id', 'purchase_amount', 'courses_completed', 'total_learning_minutes']].to_string(index=False))

    report_behavioral_groups(df)

    print('\n--- K?T LU?N NHANH ---')
    if 'advanced_course_purchase' in df.columns:
        purchase_rate = df['advanced_course_purchase'].mean()
        print(f"Tỷ lệ mua khóa nâng cao trong dữ liệu: {purchase_rate:.2%}")
    if 'churned' in df.columns:
        churn_rate = df['churned'].mean()
        print(f"Tỷ lệ rời bỏ (churn) trong dữ liệu: {churn_rate:.2%}")

    print('\nPhân tích hoàn tất. Ảnh biểu đồ và báo cáo nếu có được lưu vào:', args.output_dir)
