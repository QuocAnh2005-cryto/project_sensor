from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple
from collections import deque
import queue
import numpy as np

@dataclass
class SystemState:
    running: bool = True
    system_active: bool = False
    is_logging: bool = False
    user_is_interacting: bool = False
    is_currently_abnormal: bool = False
    offsets_calculated: bool = False

    # dung 1 mang de chua offset cho cac truc
    axes_offsets: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    previous_button_state: str = "lightgrey"

@dataclass
class DataBuffers:
    # maxsize nen de config nhung de so lon de k bi ket queue
    raw_data_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=300000))
    plot_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=200000))
    realtime_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=500))
    feature_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=200))
    
    raw_data_array: Optional[np.ndarray] = field(default=None)

    # danh sach cac deque luu lich su cho tung dac trung
    feature_history: List[deque] = field(default_factory=list)
    # lich su diem so loi autoencoder mse
    score_history: deque = field(default_factory=lambda: deque(maxlen=500))
    # xac suat cua lan du doan abnormal gan nhat
    last_classification_proba: Optional[np.ndarray] = field(default=None, metadata={'unit': 'probability_array'})

    def initialize_buffers(self, config: dict):
        """Khởi tạo kích thước của các buffer và lịch sử vẽ từ cấu hình hệ thống"""
        axis_key = config.get("ui", {}).get("raw_labels", ["X", "Y", "Z"])
        num_axes = len(axis_key)
        raw_window = config.get("plot", {}).get("raw_window", 2000)
        self.raw_data_array = np.zeros((raw_window, num_axes), dtype=np.float32)

        num_features = len(config.get("plot", {}).get("feature_names", []))
        history_len = config.get("plot", {}).get("feature_history", 50)
        self.feature_history = [deque(maxlen=history_len) for _ in range(num_features)]
