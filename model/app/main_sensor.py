"""
main.py – Entry point for Predictive Maintenance Application
Stack: PyQt5 + pyqtgraph + QSS  (matplotlib removed)
"""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

# Config loader
JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "config", "config.json"))

def get_config():
    if not os.path.exists(JSON_PATH):
        print(f"fatal error: khong tim thay file {JSON_PATH}")
        sys.exit(1)
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"fatal error: loi cu phap json - {e}")
        sys.exit(1)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from core.states import SystemState, DataBuffers
from core.pred_service import AIService
from core.processor import DataProcessor
from ui.sensor_dashboard import Dashboard
from core.feature_calc import FeatureCalculator
from core.serial_handler import SerialHandler


class App:
    def __init__(self):
        self.config = get_config()
        self.state  = SystemState()
        self.data   = DataBuffers()
        self.data.initialize_buffers(self.config)

        # Services
        self.ai       = AIService(self.config)
        self.ai.load_models()
        self.gpu_calc = FeatureCalculator(self.config)
        self.processor = DataProcessor(self.config, self.state, self.data, self.ai, self.gpu_calc)
        self.data_source = SerialHandler(self.config, self.state, self.processor)

        # Qt Application
        self.qapp = QApplication(sys.argv)

        # Dashboard
        self.dashboard = Dashboard(
            self.config,
            self.state,
            self.data,
            self.ai,
            self.data_source
        )

    def run(self):
        self.dashboard.init_ui()
        self.processor.start_threads()

        self.timer = QTimer()
        self.timer.timeout.connect(self.dashboard.update_gui_loop)
        self.timer.start(50)  # 20 FPS refresh

        self.qapp.aboutToQuit.connect(self._shutdown)
        sys.exit(self.qapp.exec_())

    def _shutdown(self):
        logging.info("shutting down...")
        self.state.system_active = False
        self.state.running = False
        self.data_source.stop()

        # Drain queues
        for q in [self.data.realtime_queue, self.data.plot_queue,
                  self.data.raw_data_queue, self.data.feature_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    try:
        app_instance = App()
        app_instance.run()
    except Exception as e:
        logging.critical(f"application crash: {e}", exc_info=True)
