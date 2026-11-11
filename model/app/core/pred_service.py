import joblib
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
import cupy as cp

# Cấu hình GPU 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        logging.error(f"GPU Config Error: {e}")

class AIService:
    def __init__(self, config):
        self.config = config
        m_cfg = config.get("model", {})
        self.ae_threshold = m_cfg.get("ae_threshold", 2.0)
        self.ae_model = None
        self.rf_model = None

    def load_models(self):
        m_cfg = self.config.get("model", {})
        try:
            # Load models
            self.ae_model = tf.keras.models.load_model(m_cfg["autoencoder_path"])
            ae_scaler = joblib.load(m_cfg["ae_scaler_path"])
            # Lưu hằng số scaler dạng numpy để chuẩn hóa trên CPU cho nhanh và an toàn
            self.ae_mean = ae_scaler.mean_
            self.ae_std = ae_scaler.scale_
            
            self.rf_model = joblib.load(m_cfg["random_forest_path"])
            self.rf_scaler = joblib.load(m_cfg["rf_scaler_path"])

            # Warm-up TensorFlow GPU model to avoid 30s-60s first-inference freeze on Jetson
            try:
                dummy_x = np.zeros((1, len(self.ae_mean)), dtype=np.float32)
                with tf.device('/GPU:0'):
                    _ = self.ae_model(dummy_x, training=False)
                logging.info("TensorFlow GPU Model Warmed Up.")
            except Exception as w_err:
                logging.warning(f"TF Warmup Warning: {w_err}")

            logging.info("All AI Models Loaded Successfully.")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def predict_ae(self, features_gpu):
        try:
            # 1. Xuống CPU 
            x_cpu = cp.asnumpy(features_gpu).reshape(1, -1).astype(np.float32)
            
            # 2. Chuẩn hóa nhanh trên CPU bằng numpy
            x_scaled = (x_cpu - self.ae_mean) / self.ae_std
            
            # 3. Ép chạy Inference trên GPU của Jetson
            with tf.device('/GPU:0'):
                # Dùng __call__ thay vì .predict() để giảm trễ cho 1 mẫu
                y_hat = self.ae_model(x_scaled, training=False).numpy()
            
            # 4. Tính MSE
            error = np.mean(np.square(x_scaled - y_hat))
            prediction = 1 if error > self.ae_threshold else 0
            return prediction, float(error)
        except Exception as e:
            logging.error(f"AI AE Error: {e}")
            return 0, 0.0

    def classify_rf(self, features_gpu):
        try:
            x_cpu = cp.asnumpy(features_gpu).reshape(1, -1)
            feat_names = self.config["plot"].get("feature_names", [])
            X_df = pd.DataFrame(x_cpu, columns=feat_names)
            X_scaled = self.rf_scaler.transform(X_df.values)
            probs = self.rf_model.predict_proba(X_scaled)[0]
            
            # Lấy nhãn lớp thực tế (1 hoặc 2) từ thuộc tính classes_ của model thay vì chỉ mục index (0 hoặc 1)
            pred_id = int(self.rf_model.classes_[np.argmax(probs)])
            
            # Cấu trúc lại mảng xác suất đầy đủ (bao gồm cả lớp Normal=0 ở đầu)
            # để các widget giao diện vẽ biểu đồ chẩn đoán nhận đúng giá trị của từng loại lỗi
            num_classes = len(self.config["model"].get("fault_library", []))
            full_probs = np.zeros(num_classes)
            for cls_label, p in zip(self.rf_model.classes_, probs):
                if int(cls_label) < num_classes:
                    full_probs[int(cls_label)] = p
            
            return pred_id, full_probs
        except Exception as e:
            logging.error(f"AI RF Error: {e}")
            # Trả về mặc định Normal nếu có lỗi xảy ra
            num_classes = len(self.config["model"].get("fault_library", []))
            default_probs = np.zeros(num_classes)
            default_probs[0] = 1.0
            return 0, default_probs
