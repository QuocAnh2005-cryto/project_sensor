import os
import time
import queue
import logging
import threading
from datetime import datetime

import numpy as np

FRAME = 9


class BinSource:
    def __init__(self, config, state, processor):
        self.config = config
        self.state = state
        self.processor = processor
        self.scale = config.get("serial", {}).get("scale_factor", 0.0000195)
        cfg = config.get("bin_source", {})
        self.bin_dir = cfg.get("dir", ".")
        self.start_from_end = cfg.get("start_from_end", True)
        self.poll_interval = cfg.get("poll_interval", 0.02)

        self.running = False
        self._thread = None

    # ── giải mã ───────────────────────────────────────────────────────────────
    def _decode_20bit(self, b0, b1, b2):
        val = ((b0 << 16) | (b1 << 8) | b2) >> 4
        if val & 0x80000:
            val -= 0x100000
        return val * self.scale

    def _parse(self, buf):
        """Trả về (list tuple xyz, số byte đã tiêu thụ). Chống trượt byte bằng marker bit."""
        out = []
        idx = 0
        n = len(buf)
        while idx <= n - FRAME:
            if (buf[idx + 2] & 1) == 1 and (buf[idx + 5] & 1) == 0 and (buf[idx + 8] & 1) == 0:
                out.append((
                    self._decode_20bit(buf[idx], buf[idx + 1], buf[idx + 2]),
                    self._decode_20bit(buf[idx + 3], buf[idx + 4], buf[idx + 5]),
                    self._decode_20bit(buf[idx + 6], buf[idx + 7], buf[idx + 8]),
                ))
                idx += FRAME
            else:
                idx += 1
        return out, idx

    # ── luồng đọc file ────────────────────────────────────────────────────────
    def _run(self):
        current_date = None
        f = None
        remainder = b""
        first_open = True

        try:
            while self.running:
                date_str = datetime.now().strftime("%Y_%m_%d")

                if date_str != current_date:
                    path = os.path.join(self.bin_dir, f"log__{date_str}.bin")
                    if not os.path.exists(path):
                        time.sleep(0.5)
                        continue
                    if f:
                        f.close()
                    f = open(path, "rb")
                    if self.start_from_end and first_open:
                        size = os.fstat(f.fileno()).st_size
                        f.seek(size - (size % FRAME))
                    first_open = False
                    current_date = date_str
                    remainder = b""
                    logging.info(f"[BinSource] Đang đọc {path}")

                chunk = f.read()
                if not chunk:
                    time.sleep(self.poll_interval)
                    continue

                buf = remainder + chunk
                batch, used = self._parse(buf)
                remainder = buf[used:]

                if batch:
                    arr = np.array(batch, dtype=np.float32)
                    try:
                        self.processor.data.raw_data_queue.put_nowait(arr)
                        self.processor.data.plot_queue.put_nowait(batch)
                    except queue.Full:
                        pass
        except Exception as e:
            logging.error(f"[BinSource] Lỗi: {e}")
        finally:
            if f:
                f.close()

    # ── API giống SerialHandler ───────────────────────────────────────────────
    def connect(self):
        if self._thread and self._thread.is_alive():
            return True
        self.state.offsets_calculated = False
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("BinSource online. Đọc dữ liệu từ file .bin...")
        return True

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logging.info("Data source stopped.")