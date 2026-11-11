import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Input, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib

# =============================
# 1️ Đường dẫn dữ liệu
# =============================
# Lưu ý: Điều chỉnh đường dẫn này đến file dữ liệu huấn luyện thực tế của bạn
data_files = [
    r"/home/jetson/Desktop/Predictive_maintenance/data_autoencoder/datatrain_autoencoder.csv", 
]

# =============================
# 2️ Đọc & gộp dữ liệu
# =============================
dfs = []
for file in data_files:
    try:
        df = pd.read_csv(file)
        print(f"Đọc file: {Path(file).name} - Shape gốc: {df.shape}")
        
        # Bỏ cột đầu (timestamp/index), giữ 18 cột feature
        df = df.iloc[:, 1:]  
        if df.shape[1] != 18:
            raise ValueError(f"File {file} có {df.shape[1]} cột feature, mong đợi 18!")
        
        dfs.append(df)
    except Exception as e:
        print(f"Lỗi đọc file {file}: {e}")
        continue

if not dfs:
    raise ValueError("Không đọc được file dữ liệu nào! Hãy cập nhật đường dẫn chính xác trong data_files.")

df_all = pd.concat(dfs, ignore_index=True)
print(f" Tổng số mẫu: {len(df_all)}")
if df_all.isnull().values.any() or np.isinf(df_all.values).any():
    print("Cảnh báo: Có NaN hoặc inf trong dữ liệu → Xử lý bằng fillna(0)")
    df_all = df_all.fillna(0)

# =============================
# 3️ Chuẩn hóa dữ liệu & Khử nhiễu
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_all.values).astype(np.float32)

# KHỬ NHIỄU: Loại bỏ các mẫu có giá trị quá cực đoan (ngoài khoảng 5 std)
mask = (np.abs(X_scaled) < 5).all(axis=1)
X_scaled = X_scaled[mask]
print(f"Shape sau khi khử nhiễu: {X_scaled.shape} (Đã loại bỏ các mẫu cực đoan)")

# =============================
# 4️ Chia dữ liệu
# =============================
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42, shuffle=True)
print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

# =============================
# 5️ Xây dựng mô hình
# =============================
feature_dim = X_scaled.shape[1] # 18 features

model = Sequential([
    Input(shape=(feature_dim,)),
    GaussianNoise(0.05), 
    Dense(12, kernel_regularizer=l2(1e-4)), 
    BatchNormalization(),
    ReLU(),
    Dense(8, kernel_regularizer=l2(1e-4)), 
    BatchNormalization(),
    ReLU(),
    Dense(12, kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    ReLU(),
    Dense(feature_dim) 
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=0.00001, 
    verbose=1
)

print("\nBắt đầu huấn luyện mô hình...")
history = model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=200,          
    batch_size=64,      
    shuffle=True,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================
# 7️ Tính Threshold 
# =============================
X_all_scaled = np.vstack([X_train, X_val])
reconstructions = model.predict(X_all_scaled, verbose=0)
mse_errors = np.mean(np.power(X_all_scaled - reconstructions, 2), axis=1)

suggested_threshold = np.percentile(mse_errors, 99) 

print(f"\n--- ĐÁNH GIÁ KẾT QUẢ TÁI TẠO ---")
print(f"Mean MSE: {np.mean(mse_errors):.5f}")
print(f"Gợi ý Ngưỡng (99th percentile): {suggested_threshold:.5f} <-- Hãy sao chép số này nạp vào config.json")

# =============================
# 8️ Lưu mô hình (Tự động lưu vào cấu trúc thư mục mới)
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "scalers").mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODEL_DIR / "scalers" / "ae_scaler.pkl"
MODEL_PATH = MODEL_DIR / "autoencoder_model.keras"

model.save(MODEL_PATH) 
joblib.dump(scaler, SCALER_PATH)

print(f"\nĐã lưu model và scaler mới cho 18 features:")
print(f" - Model: {MODEL_PATH}")
print(f" - Scaler: {SCALER_PATH}")
print("Cập nhật config.json để dùng các file này nếu chưa khớp!")
