"""
Dashboard - Predictive Maintenance Real-time Monitoring
Design: TwinXcelerator-V30 | DX-DMVN 2026
Stack: PyQt5 + pyqtgraph + QSS
"""
import os
import time
import logging
import subprocess
import numpy as np
import pyqtgraph as pg

from datetime import datetime, timedelta
from collections import deque

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QAbstractItemView
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QColor, QFont, QPainter, QBrush, QPen, QIcon

# ── pyqtgraph global config (light theme matching the reference) ──────────────
pg.setConfigOptions(antialias=False, foreground='#334155', background='#FFFFFF')


# ─────────────────────────────────────────────────────────────────────────────
#  Custom time axis: shows Unix timestamps as HH:MM:SS
# ─────────────────────────────────────────────────────────────────────────────
class TimeAxisItem(pg.AxisItem):
    """Formats X-axis ticks as wall-clock time (HH:MM:SS) from Unix timestamps."""
    def tickStrings(self, values, scale, spacing):
        result = []
        for v in values:
            try:
                result.append(datetime.fromtimestamp(v).strftime('%H:%M:%S'))
            except Exception:
                result.append('')
        return result


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_qss(path: str) -> str:
    """Load a .qss file and return its content as string."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.warning(f"Could not load QSS file: {e}")
        return ""


def _make_divider(horizontal: bool = True) -> QFrame:
    line = QFrame()
    line.setObjectName("divider")
    if horizontal:
        line.setFrameShape(QFrame.HLine)
    else:
        line.setFrameShape(QFrame.VLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def _make_circle_indicator(color: str, size: int = 10) -> QLabel:
    """Create a small colored circle QLabel."""
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    lbl.setStyleSheet(
        f"background-color: {color}; border-radius: {size//2}px; border: none;"
    )
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
#  System Stats Worker (CPU / GPU / FPS) – lightweight polling
# ─────────────────────────────────────────────────────────────────────────────
class StatsWorker(QThread):
    stats_ready = pyqtSignal(float, float, float, str)   # cpu, gpu, fps, uptime

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True
        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()

    def increment_frame(self):
        self._frame_count += 1

    def run(self):
        while self._running:
            try:
                cpu = self._get_cpu()
                gpu = self._get_gpu()

                # FPS estimation
                now = time.time()
                dt = now - self._last_fps_time
                if dt >= 1.0:
                    self._fps = self._frame_count / dt
                    self._frame_count = 0
                    self._last_fps_time = now

                elapsed = int(time.time() - self._start_time)
                h, rem = divmod(elapsed, 3600)
                m, s = divmod(rem, 60)
                uptime_str = f"{h:02d}:{m:02d}:{s:02d}"

                self.stats_ready.emit(cpu, gpu, self._fps, uptime_str)
            except Exception:
                pass
            time.sleep(2)

    def stop(self):
        self._running = False

    @staticmethod
    def _get_cpu() -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0

    @staticmethod
    def _get_gpu() -> float:
        # Method 1: Jetson sysfs – values 0-1000 → divide by 10 for %
        # Tries multiple known paths across Jetson Nano / Xavier / Orin
        for sysfs_path in [
            '/sys/devices/gpu.0/load',
            '/sys/class/devfreq/57000000.gpu/load_count',
            '/sys/class/devfreq/gpu/load',
        ]:
            try:
                with open(sysfs_path, 'r') as f:
                    raw = float(f.read().strip())
                    # Jetson values are 0-1000; clamp to 0-100
                    return min(raw / 10.0, 100.0)
            except Exception:
                pass

        # Method 2: tegrastats (Jetson Xavier / Orin / Nano B01)
        # Spawns one tegrastats sample, reads first line, then kills it
        try:
            import re
            proc = subprocess.Popen(
                ['tegrastats', '--interval', '500'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            line = proc.stdout.readline()
            proc.kill()
            proc.wait(timeout=1)
            m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
            if m:
                return float(m.group(1))
        except Exception:
            pass

        # Method 3: NVML (desktop NVIDIA GPU)
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass

        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard (Main Window)
# ─────────────────────────────────────────────────────────────────────────────
class Dashboard(QMainWindow):

    # ── Status constants ──────────────────────────────────────────────────────
    STATUS_IDLE     = "idle"
    STATUS_NORMAL   = "normal"
    STATUS_WARNING  = "warning"
    STATUS_ABNORMAL = "abnormal"

    STATUS_META = {
        STATUS_IDLE:     ("IDLE",     "Waiting for data",          "#94A3B8", "status_text_idle",     "status_display_idle"),
        STATUS_NORMAL:   ("NORMAL",   "System operating normally", "#15803D", "status_text_normal",   "status_display_normal"),
        STATUS_WARNING:  ("WARNING",  "Monitor closely",           "#B45309", "status_text_warning",  "status_display_warning"),
        STATUS_ABNORMAL: ("ABNORMAL", "Action required",           "#DC2626", "status_text_abnormal", "status_display_abnormal"),
    }

    def __init__(self, config, state, data, ai_service, serial):
        super().__init__()
        self.config    = config
        self.state     = state
        self.data      = data
        self.ai        = ai_service
        self.serial    = serial

        # ── UI config shortcuts ──────────────────────────────────────────────
        ui_cfg               = self.config.get("ui", {})
        self.labels_raw      = ui_cfg.get("raw_labels",  ["X", "Y", "Z"])
        self.colors_raw      = ui_cfg.get("raw_colors",  ["#B91C1C", "#D97706", "#1D4ED8"])
        self.offsets         = self.config["plot"].get("channel_offsets", [0] * 3)
        self.feature_names   = self.config["plot"].get("feature_names", [])
        self.num_axes        = len(self.labels_raw)
        # self.logo_path       = ui_cfg.get("logo_path", "")
        self.log_path        = ui_cfg.get("log_path", "")
        self.ae_threshold    = self.config["model"].get("ae_threshold", 3200)
        self.raw_window_ms   = self.config["plot"].get("raw_window", 2000)
        self.score_hist_len  = self.config["plot"].get("score_history", 60)

        # ── Runtime state ────────────────────────────────────────────────────
        self.curves_raw      = []
        self.fault_counter   = 0
        self.current_status  = self.STATUS_IDLE
        self.current_score   = 0.0
        self.score_x         = []
        self.score_y         = []
        # x_counter removed – MSE X axis now uses real time.time() timestamps
        self._blink_state    = False   # for fault row blinking
        self._current_fps    = 0.0
        self._start_time     = time.time()
        self.log_entries     = deque(maxlen=100)
        self.diag_window     = None
        self._last_check_str = "--:--:--"

        # ── Stats worker ─────────────────────────────────────────────────────
        self.stats_worker = StatsWorker()
        self.stats_worker.stats_ready.connect(self._on_stats_ready)
        self.stats_worker.start()

    # ─────────────────────────────────────────────────────────────────────────
    #  init_ui – called once from App.run()
    # ─────────────────────────────────────────────────────────────────────────
    def init_ui(self):
        self.setWindowTitle("Predictive Maintenance Real-time Monitoring")
        self.setMinimumSize(1280, 800)

        # ── Load QSS ─────────────────────────────────────────────────────────
        qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
        self.setStyleSheet(_load_qss(qss_path))

        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 1. Header ─────────────────────────────────────────────────────────
        root.addWidget(self._build_header())

        # ── 2. Body (charts + right panel) ───────────────────────────────────
        body_wrapper = QWidget()
        body_wrapper.setObjectName("central_widget")
        body_layout = QHBoxLayout(body_wrapper)
        body_layout.setContentsMargins(8, 8, 8, 4)
        body_layout.setSpacing(8)

        # Left column: vibration + MSE chart

        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        left_col.addWidget(self._build_vibration_card(), stretch=2)
        left_col.addWidget(self._build_mse_card(), stretch=1)   
        body_layout.addLayout(left_col, stretch=3)
        body_layout.addWidget(self._build_right_panel(), stretch=1)

        root.addWidget(body_wrapper, stretch=1)

        # ── 3. Log panel ─────────────────────────────────────────────────────
        root.addWidget(self._build_log_panel())

        # ── Blink timer for Fault Type row (500ms toggle) ────────────────────
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._blink_fault)
        self.blink_timer.start(500)

        self.showFullScreen()
        self._push_log("INFO", "Dashboard", "UI initialized successfully")

    # ─────────────────────────────────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("header_bar")
        bar.setFixedHeight(75)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 16, 0)
        h.setSpacing(14)

        # Vertical divider
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setStyleSheet("color: rgba(255,255,255,0.2); max-width: 1px;")
        h.addWidget(vline)

        # Title block
        title_block = QVBoxLayout()
        title_block.setSpacing(0)
        t1 = QLabel("Sensor Live Monitor")
        t1.setObjectName("header_title")
        t2 = QLabel("© 2026 Designed by DX-DMVN")
        t2.setObjectName("header_subtitle")
        title_block.addWidget(t1)
        title_block.addWidget(t2)
        h.addLayout(title_block)

        h.addStretch()

        # Stats badges
        self.lbl_cpu = self._make_stat_badge("CPU", "–%")
        self.lbl_gpu = self._make_stat_badge("GPU", "–%")
        self.lbl_fps = self._make_stat_badge("FPS", "–")

        for badge_widget in [self.lbl_cpu, self.lbl_gpu, self.lbl_fps]:
            h.addWidget(badge_widget)

        # Uptime
        self.lbl_uptime = QLabel("⏱  00:00:00")
        self.lbl_uptime.setObjectName("uptime_badge")
        h.addWidget(self.lbl_uptime)

        # Connection status
        self.lbl_connected = QLabel("⚡  CONNECTED")
        self.lbl_connected.setObjectName("disconnected_badge")
        h.addWidget(self.lbl_connected)

        return bar

    def _make_stat_badge(self, prefix: str, value: str) -> QWidget:
        """A small horizontal pill: '● CPU 35%'"""
        w = QWidget()
        w.setObjectName("stat_label")
        hl = QHBoxLayout(w)
        hl.setContentsMargins(8, 2, 8, 2)
        hl.setSpacing(4)
        dot = _make_circle_indicator("#4ADE80", 8)
        lbl_p = QLabel(prefix)
        lbl_p.setObjectName("stat_label")
        lbl_v = QLabel(value)
        lbl_v.setObjectName("stat_value")
        # store ref so we can update
        w._val_label = lbl_v
        w._prefix     = prefix
        hl.addWidget(dot)
        hl.addWidget(lbl_p)
        hl.addWidget(lbl_v)
        return w


    def _build_mse_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("chart_card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 8, 10, 6)
        cl.setSpacing(4)

        # Title row
        tr = QHBoxLayout()
        ico = QLabel("📈")
        ico.setObjectName("chart_icon")
        title = QLabel("Anomaly Score (MSE) - Last 5 minutes")
        title.setObjectName("chart_title")
        self.lbl_mse_cur = QLabel("Current: –")
        self.lbl_mse_cur.setObjectName("mse_current_badge")
        self.lbl_mse_thr = QLabel(f"Threshold: {self.ae_threshold}")
        self.lbl_mse_thr.setObjectName("mse_threshold_badge")
        tr.addWidget(ico)
        tr.addWidget(title)
        tr.addStretch()
        tr.addWidget(self.lbl_mse_cur)
        tr.addSpacing(6)
        tr.addWidget(self.lbl_mse_thr)
        cl.addLayout(tr)

        # pyqtgraph PlotWidget – custom TimeAxisItem so X shows HH:MM:SS
        self.win_score = pg.PlotWidget(
            axisItems={'bottom': TimeAxisItem(orientation='bottom')}
        )
        self.win_score.setBackground('#FFFFFF')
        self.win_score.showGrid(x=True, y=True, alpha=0.15)
        self.win_score.getAxis('left').setLabel('MSE Score', color='#64748B')
        self.win_score.getAxis('bottom').setLabel('Time', color='#64748B')
        self.win_score.getAxis('left').setPen(pg.mkPen('#CBD5E1'))
        self.win_score.getAxis('bottom').setPen(pg.mkPen('#CBD5E1'))
        self.win_score.setMouseEnabled(x=False, y=False)

        # Threshold line
        self.t_line = pg.InfiniteLine(
            pos=self.ae_threshold, angle=0,
            pen=pg.mkPen('#EF4444', width=1.5, style=Qt.DashLine)
        )
        self.win_score.addItem(self.t_line)

        # Score curve
        self.curve_score = self.win_score.plot(
            pen=pg.mkPen('#3B82F6', width=2),
            fillLevel=0,
            brush=pg.mkBrush(59, 130, 246, 30)
        )
        cl.addWidget(self.win_score)
        return card

    # ─────────────────────────────────────────────────────────────────────────
    #  VIBRATION CHART CARD
    # ─────────────────────────────────────────────────────────────────────────
    def _build_vibration_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("chart_card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 6)
        card_layout.setSpacing(4)

        # Title row
        title_row = QHBoxLayout()
        ico = QLabel("〰")
        ico.setObjectName("chart_icon")
        title = QLabel(f"{self.num_axes}-Axis Vibration Data")
        title.setObjectName("chart_title")
        badge = QLabel(f"⏱  Window: {self.raw_window_ms}ms")
        badge.setObjectName("window_badge")
        title_row.addWidget(ico)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(badge)
        card_layout.addLayout(title_row)

        # pyqtgraph PlotWidget
        self.win_raw = pg.PlotWidget()
        self.win_raw.setBackground('#FFFFFF')
        self.win_raw.showGrid(x=True, y=True, alpha=0.15)
        self.win_raw.getAxis('left').setLabel('Amplitude (mg)', color='#64748B')
        self.win_raw.getAxis('bottom').setLabel('Time (ms)', color='#64748B')
        self.win_raw.getAxis('left').setPen(pg.mkPen('#CBD5E1'))
        self.win_raw.getAxis('bottom').setPen(pg.mkPen('#CBD5E1'))
        self.win_raw.setMouseEnabled(x=False, y=False)

        y_lim = self.config["plot"].get("raw_y_limit", [-1000, 1000])
        self.win_raw.setYRange(y_lim[0], y_lim[1])

        # Build legend items + curves
        legend = self.win_raw.addLegend(offset=(10, 10))
        legend.setLabelTextColor('#334155')
        for i in range(self.num_axes):
            color = self.colors_raw[i] if i < len(self.colors_raw) else '#000000'
            label = self.labels_raw[i] + "-axis"
            c = self.win_raw.plot(
                pen=pg.mkPen(color, width=1.8),
                name=label
            )
            self.curves_raw.append(c)

        card_layout.addWidget(self.win_raw)
        return card

    # ─────────────────────────────────────────────────────────────────────────
    #  RIGHT PANEL
    # ─────────────────────────────────────────────────────────────────────────
    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("right_panel")
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(420)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(14, 14, 14, 14)
        pl.setSpacing(12)

        # ── STATUS section label ──────────────────────────────────────────────
        sec = QLabel("STATUS")
        sec.setObjectName("section_label")
        pl.addWidget(sec)

        # ── Status display box ────────────────────────────────────────────────
        self.status_frame = QFrame()
        self.status_frame.setObjectName("status_display_idle")
        sf_layout = QVBoxLayout(self.status_frame)
        sf_layout.setContentsMargins(12, 12, 12, 12)
        sf_layout.setSpacing(4)

        # Icon + text row
        icon_text_row = QHBoxLayout()
        self.status_icon = QLabel("●")
        self.status_icon.setStyleSheet("font-size: 34pt; color: #94A3B8; background: transparent;")
        self.status_text = QLabel("IDLE")
        self.status_text.setObjectName("status_text_idle")
        icon_text_row.addWidget(self.status_icon)
        icon_text_row.addWidget(self.status_text)
        icon_text_row.addStretch()
        sf_layout.addLayout(icon_text_row)

        self.status_sub = QLabel("Waiting for data")
        self.status_sub.setObjectName("status_subtext")
        sf_layout.addWidget(self.status_sub)

        self.status_time_lbl = QLabel("⏱  Last check: --:--:--")
        self.status_time_lbl.setObjectName("status_time")
        sf_layout.addWidget(self.status_time_lbl)

        pl.addWidget(self.status_frame)

        pl.addWidget(_make_divider())


        # ── Info rows: Fault / Confidence / Threshold ─────────────────────────
        # Fault row is built specially so we can blink it on Abnormal
        self.fault_row_frame = QFrame()
        self.fault_row_frame.setObjectName("info_row")
        fault_h = QHBoxLayout(self.fault_row_frame)
        fault_h.setContentsMargins(10, 8, 10, 8)
        fault_lbl = QLabel("🛡  Fault Type:")
        fault_lbl.setObjectName("info_label")
        self.lbl_fault_val = QLabel("None")
        self.lbl_fault_val.setObjectName("info_value_normal")
        self.lbl_fault_val.setStyleSheet(
            "color: #15803D; font-weight: 700; font-size: 11pt; background: transparent;"
        )
        fault_h.addWidget(fault_lbl)
        fault_h.addStretch()
        fault_h.addWidget(self.lbl_fault_val)
        pl.addWidget(self.fault_row_frame)

        self.lbl_conf_val  = self._make_info_row(pl, "🎯  Confidence:", "–%",  "#15803D")
        self.lbl_thr_val   = self._make_info_row(pl, "⚡  Threshold:", str(self.ae_threshold), "#1E293B")

        pl.addWidget(_make_divider())
        

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_start = QPushButton("▶  START")
        self.btn_start.setObjectName("btn_start")
        self.btn_stop  = QPushButton("⏹  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_start.clicked.connect(self._start_sys)
        self.btn_stop.clicked.connect(self._stop_sys)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        pl.addLayout(btn_row)

        pl.addWidget(_make_divider())

        # ── Status Legend ─────────────────────────────────────────────────────
        leg_title = QLabel("STATUS LEGEND")
        leg_title.setObjectName("legend_title")
        pl.addWidget(leg_title)

        for color, text, desc in [
            ("#4ADE80", "Normal",   "System operating normally"),
            ("#FACC15", "Warning",  "Monitor closely"),
            ("#EF4444", "Abnormal", "Action required"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = _make_circle_indicator(color, 12)
            t = QLabel(text)
            t.setObjectName("legend_text")
            t.setFixedWidth(80)
            d = QLabel(desc)
            d.setObjectName("legend_desc")
            row.addWidget(dot)
            row.addWidget(t)
            row.addWidget(d)
            pl.addLayout(row)

        pl.addStretch()
        return panel

    def _make_info_row(self, parent_layout, label_str: str, value_str: str, value_color: str) -> QLabel:
        """Creates a styled info row and adds it to parent_layout. Returns the value label."""
        row_frame = QFrame()
        row_frame.setObjectName("info_row")
        row_h = QHBoxLayout(row_frame)
        row_h.setContentsMargins(10, 6, 10, 6)
        lbl = QLabel(label_str)
        lbl.setObjectName("info_label")
        val = QLabel(value_str)
        val.setObjectName("info_value_normal")
        val.setStyleSheet(f"color: {value_color}; font-weight: 700; font-size: 9pt; background: transparent;")
        row_h.addWidget(lbl)
        row_h.addStretch()
        row_h.addWidget(val)
        parent_layout.addWidget(row_frame)
        return val

    # ─────────────────────────────────────────────────────────────────────────
    #  SYSTEM LOG PANEL
    # ─────────────────────────────────────────────────────────────────────────
    def _build_log_panel(self) -> QFrame:
        card = QFrame()
        card.setObjectName("log_card")
        card.setFixedHeight(170)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 6, 10, 6)
        cl.setSpacing(4)

        # Title row
        tr = QHBoxLayout()
        title = QLabel("📋  SYSTEM LOG - Latest 5 entries")
        title.setObjectName("log_title")
        self.btn_open_log = QPushButton("📂  Open Log Folder")
        self.btn_open_log.setObjectName("btn_open_log")
        self.btn_open_log.clicked.connect(self._open_log_folder)
        tr.addWidget(title)
        tr.addStretch()
        tr.addWidget(self.btn_open_log)
        cl.addLayout(tr)

        # Table
        self.log_table = QTableWidget()
        self.log_table.setColumnCount(4)
        self.log_table.setHorizontalHeaderLabels(["TIME", "LEVEL", "SOURCE", "MESSAGE"])
        self.log_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.log_table.setColumnWidth(0, 165)
        self.log_table.setColumnWidth(1, 70)
        self.log_table.setColumnWidth(2, 130)
        self.log_table.setRowCount(0)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setShowGrid(False)
        cl.addWidget(self.log_table)
        return card

    # ─────────────────────────────────────────────────────────────────────────
    #  GUI UPDATE LOOP (called by QTimer every 50ms from App)
    # ─────────────────────────────────────────────────────────────────────────
    def update_gui_loop(self):
        if not self.state.system_active:
            return
        try:
            self.stats_worker.increment_frame()

            # 1. Raw waveform from realtime_queue
            all_pts = []
            while not self.data.realtime_queue.empty():
                all_pts.extend(self.data.realtime_queue.get_nowait())
                self.data.realtime_queue.task_done()
            if all_pts:
                self._update_raw_plots(all_pts)

            # 2. AI results from feature_queue
            while not self.data.feature_queue.empty():
                feats, score, rf_pred = self.data.feature_queue.get_nowait()
                self.data.feature_queue.task_done()
                self._update_score_plot(score)
                self._run_alert_logic(score, rf_pred)
                self.current_score = score

        except Exception as e:
            logging.error(f"Render Error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Private update helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _update_raw_plots(self, batch_points):
        pts = np.array(batch_points)
        num_new = pts.shape[0]
        max_buffer = self.data.raw_data_array.shape[0]
        if num_new > max_buffer:
            pts = pts[-max_buffer:]
            num_new = max_buffer

        self.data.raw_data_array = np.roll(self.data.raw_data_array, -num_new, axis=0)
        self.data.raw_data_array[-num_new:, :] = pts

        actual_axes = min(pts.shape[1], len(self.curves_raw))
        for i in range(actual_axes):
            off = self.offsets[i] if i < len(self.offsets) else 0
            self.curves_raw[i].setData(self.data.raw_data_array[:, i] + off)

    def _update_score_plot(self, score: float):
        now = time.time()
        self.score_x.append(now)
        self.score_y.append(score)

        # Keep only last score_hist_len points
        if len(self.score_x) > self.score_hist_len:
            self.score_x.pop(0)
            self.score_y.pop(0)

        self.curve_score.setData(self.score_x, self.score_y)

        # X range: always show last 5 minutes ending at now
        window_secs = 5 * 60
        self.win_score.setXRange(now - window_secs, now + 5, padding=0)

        max_y = max(max(self.score_y), self.ae_threshold) if self.score_y else self.ae_threshold
        self.win_score.setYRange(0, max_y * 1.3)

        # Update badge labels
        self.lbl_mse_cur.setText(f"Current: {score:.4f}")

    def _run_alert_logic(self, score: float, rf_pred: int):
        threshold = self.ae_threshold
        fault_lib = self.config["model"].get("fault_library", [])

        if score > threshold:
            self.fault_counter += 1
        else:
            self.fault_counter = 0

        self.state.is_currently_abnormal = (self.fault_counter >= 3)
        now_str = datetime.now().strftime("%H:%M:%S")
        self._last_check_str = now_str

        if self.fault_counter >= 3:
            new_status = self.STATUS_ABNORMAL
            # Find RF prediction label
            fault_label = "Unknown"
            for f in fault_lib:
                if f["id"] == rf_pred:
                    fault_label = f["label"]
                    break
            # Show fault as a prominent badge (⚠ prefix + large bold red text)
            self.lbl_fault_val.setText(f"⚠  {fault_label}")
            self.lbl_fault_val.setStyleSheet(
                "color: #FFFFFF; font-weight: 800; font-size: 12pt;"
                "background: #DC2626; border-radius: 5px; padding: 3px 10px;"
            )
            # Blink the fault row frame (handled by _blink_fault timer)
            self._blink_active = True
            self._push_log("WARN", "AI Pipeline", f"Abnormal detected: score={score:.4f} | fault={fault_label}")
        elif self.fault_counter >= 1:
            new_status = self.STATUS_WARNING
            self.lbl_fault_val.setText("⚠  Approaching")
            self.lbl_fault_val.setStyleSheet(
                "color: #FFFFFF; font-weight: 700; font-size: 11pt;"
                "background: #B45309; border-radius: 5px; padding: 3px 10px;"
            )
            self._blink_active = False
            self.fault_row_frame.setStyleSheet("")
        else:
            new_status = self.STATUS_NORMAL
            self.lbl_fault_val.setText("None")
            self.lbl_fault_val.setStyleSheet(
                "color: #15803D; font-weight: 700; font-size: 11pt; background: transparent;"
            )
            self._blink_active = False
            self.fault_row_frame.setStyleSheet("")

        # Confidence (uses score vs threshold as a proxy if no proba)
        proba = self.data.last_classification_proba
        if proba is not None and len(proba) > 0:
            conf = float(np.max(proba)) * 100.0
            self.lbl_conf_val.setText(f"{conf:.1f}%")
        else:
            # Invert MSE-based pseudo-confidence: closer to 0 = more confident normal
            if score < threshold:
                pct = max(0.0, (1.0 - score / threshold)) * 100.0
            else:
                pct = 0.0
            self.lbl_conf_val.setText(f"{pct:.1f}%")

        self._set_status(new_status)

    def _set_status(self, status: str):
        if status == self.current_status and status != self.STATUS_IDLE:
            # Still update time
            self.status_time_lbl.setText(f"⏱  Last check: {self._last_check_str}")
            return

        self.current_status = status
        meta = self.STATUS_META[status]
        text, sub, color, text_obj, frame_obj = meta

        self.status_frame.setObjectName(frame_obj)
        # Force QSS repaint
        self.status_frame.style().unpolish(self.status_frame)
        self.status_frame.style().polish(self.status_frame)

        self.status_icon.setStyleSheet(f"font-size: 28pt; color: {color}; background: transparent;")
        self.status_text.setText(text)
        self.status_text.setObjectName(text_obj)
        self.status_text.style().unpolish(self.status_text)
        self.status_text.style().polish(self.status_text)

        self.status_sub.setText(sub)
        self.status_time_lbl.setText(f"⏱  Last check: {self._last_check_str}")

        # Update connection badge while we're at it
        if self.state.system_active:
            self.lbl_connected.setText("⚡  CONNECTED")
            self.lbl_connected.setObjectName("connected_badge")
        else:
            self.lbl_connected.setText("⚡  DISCONNECTED")
            self.lbl_connected.setObjectName("disconnected_badge")
        self.lbl_connected.style().unpolish(self.lbl_connected)
        self.lbl_connected.style().polish(self.lbl_connected)

    # ─────────────────────────────────────────────────────────────────────────
    #  System Log helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _push_log(self, level: str, source: str, message: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_entries.appendleft((now, level, source, message))
        self._refresh_log_table()

    def _refresh_log_table(self):
        entries = list(self.log_entries)[:5]
        self.log_table.setRowCount(len(entries))
        level_colors = {
            "INFO":  "#3B82F6",
            "WARN":  "#FACC15",
            "WARNING": "#FACC15",
            "ERROR": "#EF4444",
            "CRIT":  "#7C3AED",
        }
        dot_colors = {
            "INFO": "#4ADE80",
            "WARN": "#FACC15",
            "WARNING": "#FACC15",
            "ERROR": "#EF4444",
        }
        for row, (ts, lvl, src, msg) in enumerate(entries):
            # Time with dot
            time_widget = QWidget()
            tw_layout = QHBoxLayout(time_widget)
            tw_layout.setContentsMargins(6, 0, 0, 0)
            tw_layout.setSpacing(4)
            dot = _make_circle_indicator(dot_colors.get(lvl, "#94A3B8"), 8)
            ts_lbl = QLabel(ts)
            ts_lbl.setStyleSheet("font-size: 8pt; color: #334155;")
            tw_layout.addWidget(dot)
            tw_layout.addWidget(ts_lbl)
            tw_layout.addStretch()
            self.log_table.setCellWidget(row, 0, time_widget)

            # Level
            lv_item = QTableWidgetItem(lvl)
            lv_item.setForeground(QColor(level_colors.get(lvl, "#64748B")))
            lv_item.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lv_item.setTextAlignment(Qt.AlignCenter)
            self.log_table.setItem(row, 1, lv_item)

            # Source
            self.log_table.setItem(row, 2, QTableWidgetItem(src))

            # Message
            self.log_table.setItem(row, 3, QTableWidgetItem(msg))

        self.log_table.setRowHeight(0, 24)

    # ─────────────────────────────────────────────────────────────────────────
    #  Button handlers
    # ─────────────────────────────────────────────────────────────────────────
    def _start_sys(self):
        if not self.state.system_active:
            if self.serial.connect():
                self.state.system_active = True
                self.btn_start.setEnabled(False)
                self._set_status(self.STATUS_NORMAL)
                self.lbl_connected.setText("⚡  CONNECTED")
                self.lbl_connected.setObjectName("connected_badge")
                self.lbl_connected.style().unpolish(self.lbl_connected)
                self.lbl_connected.style().polish(self.lbl_connected)
                self._push_log("INFO", "System", "Monitoring started")

    def _stop_sys(self):
        self.state.system_active = False
        self.serial.stop()
        self.btn_start.setEnabled(True)
        self._set_status(self.STATUS_IDLE)
        self.lbl_connected.setText("⚡  DISCONNECTED")
        self.lbl_connected.setObjectName("disconnected_badge")
        self.lbl_connected.style().unpolish(self.lbl_connected)
        self.lbl_connected.style().polish(self.lbl_connected)
        self._push_log("INFO", "System", "Monitoring stopped by user")

    def _open_log_folder(self):
        log_dir = os.path.dirname(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", self.log_path)
        ))
        if not os.path.exists(log_dir):
            log_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            if os.name == 'nt':
                os.startfile(log_dir)
            else:
                subprocess.Popen(['xdg-open', log_dir])
        except Exception as e:
            logging.error(f"Open log folder error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Fault row blink (called every 500ms by blink_timer)
    # ─────────────────────────────────────────────────────────────────────────
    def _blink_fault(self):
        """Toggle fault_row_frame background between bright red and muted red."""
        if not hasattr(self, '_blink_active') or not self._blink_active:
            return
        self._blink_state = not self._blink_state
        if self._blink_state:
            # Bright pulse: vivid red border
            self.fault_row_frame.setStyleSheet(
                "background: #FEE2E2; border: 2px solid #DC2626; border-radius: 8px;"
            )
        else:
            # Dim pulse: lighter tint
            self.fault_row_frame.setStyleSheet(
                "background: #FFF5F5; border: 1px solid #FCA5A5; border-radius: 8px;"
            )

    # ─────────────────────────────────────────────────────────────────────────
    #  Stats callback
    # ─────────────────────────────────────────────────────────────────────────
    def _on_stats_ready(self, cpu: float, gpu: float, fps: float, uptime: str):
        self.lbl_cpu._val_label.setText(f"{cpu:.0f}%")
        self.lbl_gpu._val_label.setText(f"{gpu:.0f}%")
        self.lbl_fps._val_label.setText(f"{fps:.0f}")
        self.lbl_uptime.setText(f"⏱  {uptime}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Keyboard shortcuts
    # ─────────────────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.close()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()   

    def changeEvent(self, event):
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() & Qt.WindowMaximized:
                self.showFullScreen()
        super().changeEvent(event)

    def closeEvent(self, event):
        self.stats_worker.stop()
        self.stats_worker.wait(2000)
        super().closeEvent(event)
