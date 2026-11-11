"""
widget_diag.py – AI Diagnostic Detail Window
Uses pyqtgraph BarGraphItem instead of matplotlib pie chart.
"""
import numpy as np
import pyqtgraph as pg
import logging
import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont


class ProbabilityBarWindow(QWidget):
    """
    Pop-up window showing AI fault classification probabilities as a bar chart.
    Replaces the old matplotlib pie chart with pyqtgraph for consistency and speed.
    """

    def __init__(self, config, data_ref):
        super().__init__()
        self.config   = config
        self.data_ref = data_ref

        fault_lib = self.config["model"].get("fault_library", [])
        # Only fault types (id > 0)
        self.fault_info = [f for f in fault_lib if f["id"] > 0]
        self.labels     = [f["label"] for f in self.fault_info]
        self.colors_hex = [f["color"]  for f in self.fault_info]
        self.n_faults   = len(self.labels)

        self._init_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_chart)
        self.timer.start(3000)

    # ─────────────────────────────────────────────────────────────────────────
    def _init_ui(self):
        self.setWindowTitle("AI Diagnostic Detail – Fault Probability")
        self.setFixedSize(560, 420)

        # QSS for this window
        self.setStyleSheet("""
            QWidget {
                background: #F8FAFC;
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #1E293B;
            }
            QLabel#diag_title {
                font-size: 13pt;
                font-weight: 700;
                color: #1E293B;
            }
            QLabel#diag_sub {
                font-size: 9pt;
                color: #64748B;
            }
            QLabel#fault_label {
                font-size: 9pt;
                color: #334155;
            }
            QLabel#fault_pct {
                font-size: 9pt;
                font-weight: 700;
                color: #1D4ED8;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        # Title
        title = QLabel("Live Probability Distribution")
        title.setObjectName("diag_title")
        sub   = QLabel("AI fault classification (Random Forest) – updates every 3s")
        sub.setObjectName("diag_sub")
        root.addWidget(title)
        root.addWidget(sub)

        # Bar chart
        pg.setConfigOptions(antialias=True, foreground='#334155', background='#FFFFFF')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#FFFFFF')
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        self.plot_widget.getAxis('bottom').setLabel('Fault Type')
        self.plot_widget.getAxis('left').setLabel('Probability')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setYRange(0, 1.1)

        # X ticks with label names
        ticks = [[(i, lbl) for i, lbl in enumerate(self.labels)]]
        self.plot_widget.getAxis('bottom').setTicks(ticks)

        # Bar item placeholder
        bar_colors = [QColor(c) for c in self.colors_hex]
        self.bars = pg.BarGraphItem(
            x=list(range(self.n_faults)),
            height=[0.0] * self.n_faults,
            width=0.6,
            brushes=bar_colors
        )
        self.plot_widget.addItem(self.bars)
        root.addWidget(self.plot_widget, stretch=1)

        # Legend rows
        legend_frame = QFrame()
        legend_frame.setStyleSheet("background: #F1F5F9; border-radius: 6px;")
        lf_layout = QHBoxLayout(legend_frame)
        lf_layout.setContentsMargins(10, 6, 10, 6)
        lf_layout.setSpacing(16)

        self.pct_labels = {}
        for i, (lbl, color) in enumerate(zip(self.labels, self.colors_hex)):
            item_layout = QVBoxLayout()
            dot_row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 12pt; background: transparent;")
            name = QLabel(lbl)
            name.setObjectName("fault_label")
            dot_row.addWidget(dot)
            dot_row.addWidget(name)
            pct = QLabel("0.0%")
            pct.setObjectName("fault_pct")
            pct.setAlignment(Qt.AlignCenter)
            item_layout.addLayout(dot_row)
            item_layout.addWidget(pct)
            lf_layout.addLayout(item_layout)
            self.pct_labels[i] = pct

        root.addWidget(legend_frame)

        # Note
        self.note_lbl = QLabel("* Waiting for next AI inference (3s)")
        self.note_lbl.setObjectName("diag_sub")
        root.addWidget(self.note_lbl)

        self.refresh_chart()

    # ─────────────────────────────────────────────────────────────────────────
    def refresh_chart(self):
        try:
            proba = self.data_ref.last_classification_proba
            display = np.zeros(self.n_faults)

            if proba is not None and len(proba) > 1:
                actual = min(len(proba) - 1, self.n_faults)
                raw = proba[1: actual + 1]
                display[:actual] = raw
                total = display.sum()
                if total > 0:
                    display /= total

            self.bars.setOpts(height=display.tolist())

            for i, pct in self.pct_labels.items():
                val = display[i] if i < len(display) else 0.0
                pct.setText(f"{val * 100:.1f}%")

            if proba is None:
                self.note_lbl.setText("* Waiting for next AI inference (3s)")
            else:
                self.note_lbl.setText(f"* Last updated: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            logging.error(f"Diag Widget Error: {e}")


# Keep backward-compatible alias
ProbabilityPieWindow = ProbabilityBarWindow
