###----------{coding: utf-8}----------###

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
from imblearn.over_sampling import RandomOverSampler

# Bỏ qua các cảnh báo về việc tên feature không khớp khi dùng NumPy array
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN MODEL (PHIÊN BẢN CÂN BẰNG TỐI ƯU) ---")

# --- Bước 1: Tải dữ liệu ---
# Lưu ý: Cập nhật đường dẫn này đến file dữ liệu thực tế của bạn
data_path = r"/home/jetson/Desktop/Predictive_maintenance/data_randomforest/datatrain_randomforest.csv"
print(f"Đang tải dữ liệu từ: {data_path}")
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy tệp dữ liệu tại '{data_path}'. Hãy cập nhật đường dẫn chính xác.")
    exit()

X = df.drop('label', axis=1)
y = df['label']

# SỬA LỖI CÚ PHÁP: X.column -> X.columns
feature_names = X.columns.tolist()
num_features = len(feature_names)

print(f"Số lượng feature đầu vào : {num_features}")
if num_features != 18: 
    print(f'Warn: số lượng đặc trưng thực tế là {num_features}, không phải 18.')
print(f"Dữ liệu đầu vào (X) có shape: {X.shape}")
print(f"Dữ liệu đầu ra (y) có shape: {y.shape}")
print(f"Số mẫu: {len(df)} | Số class: {y.nunique()}")
print(f"Class distribution:\n{y.value_counts()}\n")
if X.shape[1] != 18:
    raise ValueError(f"ERROR: File chỉ có {X.shape[1]} feature, cần đúng 18!")
print("-" * 50)

# Khử nhiễu: Loại bỏ các dòng có giá trị quá dị biệt (giống bài Autoencoder)
scaler_check = StandardScaler()
X_temp = scaler_check.fit_transform(X)
mask = (np.abs(X_temp) < 5).all(axis=1) # Giữ lại các mẫu trong khoảng 5 std
X = X[mask]
y = y[mask]
print(f"Số mẫu sau khi khử nhiễu: {len(X)}")

# --- Bước 2: Chia dữ liệu ---
print("Đang chia dữ liệu train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("-" * 50)

# Cân bằng dữ liệu sử dụng RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train) 

print(f"Số mẫu sau khi Augment (Oversampling): {len(X_train)}")

# --- Bước 3: Chuẩn hóa dữ liệu ---
print("Đang huấn luyện Scaler và chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)
print("Chuẩn hóa hoàn tất!")
print("-" * 50)

# --- Bước 4: Xây dựng Model với tham số Cân bằng Tối ưu ---
print("Đang cấu hình và huấn luyện model Random Forest với tham số cân bằng...")
model = RandomForestClassifier(
    n_estimators=200,      
    max_depth=5,   
    min_samples_leaf=10,    
    max_features='sqrt', 
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1,
)

print(f" - n_estimators: {model.n_estimators}")
print(f" - max_depth: {model.max_depth}")
print(f" - min_samples_leaf: {model.min_samples_leaf}")

model.fit(X_train_scaled, y_train)
print("\nHuấn luyện hoàn tất!")
print("-" * 50)

# --- Bước 5: Đánh giá Model ---
print("Đang đánh giá model trên tập test...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=> Độ chính xác tổng thể (Accuracy): {accuracy:.4f}\n")

print("=> Báo cáo phân loại chi tiết (Classification Report):")
target_names = ["Low speed(0)", 'High Speed (1)']
print(classification_report(y_test, y_pred, target_names=target_names))
print("-" * 50)

# --- Bước 6: Lưu Model và Scaler (Tự động lưu vào cấu trúc thư mục mới) ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "scalers").mkdir(parents=True, exist_ok=True)
print(f"Đang lưu model và scaler vào thư mục: '{MODEL_DIR}'")

scaler_filename = "rf_scaler.pkl"
scaler_save_path = MODEL_DIR / "scalers" / scaler_filename
joblib.dump(scaler, scaler_save_path)
print(f"-> Scaler đã lưu vào: '{scaler_save_path}'")

model_filename = "rf_classifier.joblib"
model_save_path = MODEL_DIR / model_filename
joblib.dump(model, model_save_path)
print(f"-> Model đã lưu vào: '{model_save_path}'")
print("-" * 50)
print("--- QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT ---")
