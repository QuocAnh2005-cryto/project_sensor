import serial
import time
import threading
import queue
import logging
import numpy as np

class SerialHandler:
    def __init__(self, config, state, processor):
        self.config = config
        self.state = state
        self.processor = processor
        
        self.serial_cfg = config.get("serial", {})
        self.port = self.serial_cfg.get("port", "/dev/ttyACM0")
        self.baud = self.serial_cfg.get("baud", 1000000)
        self.scale = self.serial_cfg.get("scale_factor", 0.0000195)
        
        # Hàng đợi trung chuyển nội bộ giữa 2 nhân CPU (Core)
        self.internal_byte_queue = queue.Queue(maxsize=1000000)
        self.running = False
        self.ser = None

    def _decode_20bit(self, b):
        """Giải mã 20-bit dữ liệu từ 3 byte sensor (ADXL357)"""
        val = (b[0] << 16) | (b[1] << 8) | b[2]
        val >>= 4 
        # Xử lý số âm (2's complement 20-bit)
        if val & 0x80000:
            val -= 0x100000
        return val * self.scale

    def _reader_worker(self):
        """Luồng 1 (Nhân CPU 1): Chuyên đọc dữ liệu thô từ cổng Serial phần cứng"""
        logging.info("Core Reader Started.")
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    chunk = self.ser.read(self.ser.in_waiting)
                    self.internal_byte_queue.put(chunk)
                else:
                    time.sleep(0.0001) # Tránh ăn 100% CPU
            except Exception as e:
                logging.error(f"Hardware Reading Crash: {e}")
                break

    def _sync_processor_worker(self):
        """Luồng 2 (Nhân CPU 2): Đồng bộ và giải mã byte thành mẫu XYZ vật lý (Đã tối ưu index)"""
        buffer = bytearray()
        
        while self.running:
            try:
                data = self.internal_byte_queue.get(timeout=1.0)
                buffer.extend(data)
                
                batch_for_system = []
                idx = 0
                buf_len = len(buffer)
                
                while idx <= buf_len - 9:
                    # Đọc 9 byte, kiểm tra Marker Bit để đồng bộ
                    is_x = (buffer[idx + 2] & 0x01) == 1
                    is_y_z = (buffer[idx + 5] & 0x01) == 0 and (buffer[idx + 8] & 0x01) == 0
                    
                    if is_x and is_y_z:
                        b = buffer[idx:idx + 9]
                        vals = (
                            self._decode_20bit(b[0:3]),
                            self._decode_20bit(b[3:6]),
                            self._decode_20bit(b[6:9])
                        )
                        batch_for_system.append(vals)
                        idx += 9
                    else:
                        # Trượt 1 byte tìm lại mốc đồng bộ
                        idx += 1
                
                # Cắt phần byte đã xử lý xong trong 1 thao tác duy nhất
                if idx > 0:
                    del buffer[:idx]
                
                if batch_for_system:
                    # Gửi dữ liệu thô sang Processor
                    arr_batch = np.array(batch_for_system, dtype=np.float32)
                    try:
                        self.processor.data.raw_data_queue.put_nowait(arr_batch)
                        self.processor.data.plot_queue.put_nowait(batch_for_system)
                    except queue.Full:
                        pass

            except queue.Empty:
                continue

    def connect(self):
        """Khởi động luồng thu thập dữ liệu và kết nối Pico"""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            # --- HANDSHAKE ---
            logging.info("Starting Handshake with Pico...")
            for _ in range(5):
                self.ser.write(b'E') # Stop Pico
                time.sleep(0.1)
                if self.ser.in_waiting > 0:
                    self.ser.read(self.ser.in_waiting)
            
            ready_confirmed = False
            start_t = time.time()
            self.ser.reset_input_buffer()
            while time.time() - start_t < 3:
                self.ser.write(b'E')
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip().upper()
                    if "READY" in line:
                        ready_confirmed = True
                        break
                time.sleep(0.1)

            if not ready_confirmed:
                logging.error("Pico Connection Timeout!")
                return False

            # --- START DATA STREAM ---
            self.state.offsets_calculated = False
            self.ser.write(b'S') # Gửi START
            self.running = True
            
            # Khởi chạy luồng đọc và đồng bộ song song
            threading.Thread(target=self._reader_worker, daemon=True).start()
            threading.Thread(target=self._sync_processor_worker, daemon=True).start()
            
            logging.info("Pico and ADXL357 Online. Real-time data flowing...")
            return True

        except Exception as e:
            logging.error(f"Serial Connection Failed: {e}")
            return False

    def stop(self):
        """Dừng thu thập cảm biến an toàn"""
        self.running = False
        if self.ser:
            try:
                self.ser.write(b'E') # Dừng Pico
                time.sleep(0.1)
                self.ser.close()
            except:
                pass
        logging.info("Data source stopped.")
