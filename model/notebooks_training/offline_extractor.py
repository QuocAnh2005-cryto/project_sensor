import pandas as pd
import cupy as cp
import numpy as np
import random
import os
import time
import logging
import csv 
import sys
from pathlib import Path 



# Cấu hình log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đưa thư mục model/app/ vào sys.path để import bộ tính đặc trưng dùng chung
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from core.feature_calc import FeatureCalculator

# ------------Config Đường Dẫn File----------
# Cấu hình tương đối từ thư mục của script
INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "final_output.csv"))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datatrain.csv"))
        
def run_extraction():
    # LABEL = "0"
    LABEL = "1"
    SAMPLE_RATE = 4000.0
    WINDOW_SIZE = 80000
    MIN_STEP = 2000
    MAX_STEP = 20001
    READ_CHUNK_SIZE = 1_000_000

    if not os.path.exists(INPUT_FILE):
        logging.error(f"File nguồn không tồn tại tại: {INPUT_FILE}. Hãy cập nhật đường dẫn chính xác.")
        return

    start_time = time.time()
    
    # Sử dụng bộ tính toán đặc trưng dùng chung để tránh trùng lặp code
    processor = FeatureCalculator(SAMPLE_RATE)

    # Định nghĩa các cột header
    feature_cols = []
    for ch in ["ch1", "ch2", "ch3"]:
        for feat in ["RMS", "Kurtosis", "CrestFactor", "Std", "PeakToPeak", "ShapeFactor"]:
            feature_cols.append(f"{ch}/{feat}")
    header = ['label'] + feature_cols

    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Mở file ghi kết quả
    with open(OUTPUT_FILE, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        logging.info(f"----- Bắt đầu xử lý trích xuất đặc trưng offline -----")
        logging.info(f"File nguồn: {INPUT_FILE}")
        logging.info(f"File kết quả: {OUTPUT_FILE}")

        # Buffer xử lý cửa sổ trượt giữa các chunk
        left_data = np.empty((0, 3), dtype=np.float32)
        total = 0
        processed_rows = 0

        try:
            for df_chunk in pd.read_csv(INPUT_FILE, usecols=['x_value', 'y_value', 'z_value'],
                                        chunksize=READ_CHUNK_SIZE, dtype=np.float32):
                chunk_data = np.vstack([left_data, df_chunk.values])
                processed_rows += len(df_chunk)

                idx = 0
                while idx + WINDOW_SIZE <= len(chunk_data):
                    # Cắt cửa sổ 20 giây (80.000 mẫu)
                    window_b = chunk_data[idx : idx + WINDOW_SIZE]

                    # Khử mean cục bộ trước khi tính toán
                    normalized_window = window_b - np.mean(window_b, axis=0)

                    try:
                        # Tính đặc trưng trên GPU và chuyển về CPU
                        features_gpu = processor.diagnostic_features(normalized_window)
                        features_cpu = cp.asnumpy(features_gpu)
                        
                        writer.writerow([LABEL] + features_cpu.tolist())
                        total += 1
                    except Exception as e:
                        logging.error(f"Lỗi tính toán tại dòng {processed_rows + idx} : {e}")

                    # Nhảy bước ngẫu nhiên tránh overfitting
                    idx += random.randint(MIN_STEP, MAX_STEP)

                # Giữ lại phần dữ liệu thừa cho chunk tiếp theo
                left_data = chunk_data[idx:]

                # Giải phóng bộ nhớ GPU của CuPy sau mỗi big chunk
                cp.get_default_memory_pool().free_all_blocks()
                logging.info(f"Đã duyệt: {processed_rows} dòng. Trích xuất: {total} mẫu đặc trưng.")
                
        except Exception as e:
            logging.error(f"Lỗi đọc file nguồn: {e}")

    duration = time.time() - start_time
    logging.info(f"----- Xử lý hoàn tất -----")
    logging.info(f"Thời gian chạy: {duration:.2f}s")
    logging.info(f"Tổng số mẫu đặc trưng trích xuất: {total}")

if __name__ == "__main__":
    run_extraction()
