import threading
import time
import numpy as np
import logging
from collections import deque
import queue
import cupy as cp

class DataProcessor:
    def __init__(self, config, state, data, ai_service, gpu_calc):
        self.config = config
        self.state = state
        self.data = data
        self.ai_service = ai_service
        self.gpu_calc = gpu_calc

        proc_cfg = config.get("processing", {})
        self.batch_size = proc_cfg.get("batch_size", 80000)
        self.stride = proc_cfg.get("stride_size", 20000)
        self.sample_rate = proc_cfg.get("sample_rate", 4000)

        self.data_buffer = deque(maxlen=self.batch_size)
        self.new_points = 0 

        # Cấu hình bộ lọc Bias cho UI trên CPU (nhẹ cho mảng nhỏ)
        self.moving_bias_cpu = np.zeros(3, dtype=np.float32)
        
        logging.info(f"Processor initialized: {self.sample_rate}Hz, batch={self.batch_size}, stride={self.stride}")

    def start_threads(self):
        threading.Thread(target=self._realtime_loop, daemon=True).start()
        threading.Thread(target=self._feature_loop, daemon=True).start()

    def _realtime_loop(self):
        """Luồng hiển thị: Phóng đại dữ liệu (visual_boost) để hiển thị nhạy bén trên UI bằng NumPy CPU"""
        alpha = self.config["processing"].get("alpha", 0.9997)
        v_boost = self.config["plot"].get("visual_boost", 1000.0)
        ds = self.config["plot"].get("downsampling", 2)

        while self.state.running:
            if not self.state.system_active:
                time.sleep(0.1)
                continue
            try:
                draw_batch = []
                while not self.data.plot_queue.empty():
                    pts = self.data.plot_queue.get_nowait()
                    pts_np = np.array(pts, dtype=np.float32)
                    
                    # Trừ nhiễu tĩnh để cân bằng về 0 trên CPU
                    curr_mean = np.mean(pts_np, axis=0)
                    self.moving_bias_cpu = alpha * self.moving_bias_cpu + (1 - alpha) * curr_mean
                    
                    # PHÓNG ĐẠI CHỈ Ở NHÁNH NÀY
                    visual_np = (pts_np - self.moving_bias_cpu) * v_boost
                    draw_batch.extend(visual_np.tolist())
                    self.data.plot_queue.task_done()
                
                if draw_batch:
                    # Gửi batch điểm xuống Dashboard
                    self.data.realtime_queue.put(draw_batch[::ds])
                time.sleep(0.02)
            except Exception as e:
                logging.error(f"UI Proc Error: {e}")

    def _feature_loop(self):
        """Luồng AI: Dùng dữ liệu nguyên bản trên GPU để tính toán đặc trưng và chạy mô hình"""
        while self.state.running:
            if not self.state.system_active:
                time.sleep(0.1)
                continue
            try:
                while not self.data.raw_data_queue.empty():
                    batch = self.data.raw_data_queue.get_nowait()
                    self.data_buffer.extend(batch)
                    self.new_points += len(batch)
                    self.data.raw_data_queue.task_done()

                if self.new_points >= self.stride and len(self.data_buffer) == self.batch_size:
                    raw_window = np.array(list(self.data_buffer), dtype=np.float32)
                    
                    # Chuyển lên CuPy GPU và trừ Mean 100% trên GPU
                    raw_gpu = cp.asarray(raw_window, dtype=cp.float32)
                    normalized_gpu = raw_gpu - cp.mean(raw_gpu, axis=0)
                    
                    # Tính đặc trưng bằng CuPy trên GPU
                    features_gpu = self.gpu_calc.diagnostic_features(normalized_gpu)
                    is_abnormal, score = self.ai_service.predict_ae(features_gpu)
                    
                    rf_pred = 0
                    if is_abnormal:
                        rf_pred, probs = self.ai_service.classify_rf(features_gpu)
                        self.data.last_classification_proba = probs
                        logging.warning(f"--- DETECTED ABNORMAL --- Score: {score:.2f} | ID: {rf_pred}")
                    else:
                        logging.info(f"AI Eval Normal - Score: {score:.2f}")

                    feat_cpu = cp.asnumpy(features_gpu)
                    self.data.feature_queue.put((feat_cpu, score, rf_pred))
                    self.new_points = 0
                time.sleep(0.01)
            except Exception as e:
                logging.error(f"AI Proc Error: {e}")
