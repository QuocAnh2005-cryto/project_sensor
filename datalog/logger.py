import serial
import time
import threading
import queue
from datetime import datetime

# --- CẤU HÌNH ---
PORT = '/dev/ttyACM0' 
BAUD = 1000000
RAW_QUEUE = queue.Queue(maxsize=2000)
CLEAN_QUEUE_FILE = queue.Queue(maxsize=2000)
CLEAN_QUEUE_AI = queue.Queue(maxsize=2000)

# --- THREAD 1: ĐỌC DỮ LIỆU THÔ TỪ SERIAL ---
def serial_reader_thread(stop_event):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        ser.write(b'E')
        time.sleep(0.5)
        ser.reset_input_buffer()

        while not stop_event.is_set():
            ser.write(b'E')
            line = ser.readline().decode(errors='ignore').strip()
            if "READY" in line: break
            time.sleep(0.1)

        print("--- ĐÃ KẾT NỐI PICO - BẮT ĐẦU ĐO ---")
        ser.write(b'S')
        
        while not stop_event.is_set():
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                RAW_QUEUE.put(chunk)
            else:
                time.sleep(0.001)
        ser.write(b'E')
        ser.close()
    except Exception as e:
        print(f"Lỗi Serial: {e}")
        stop_event.set()

# --- THREAD 2: ĐỒNG BỘ HÓA (CHỐNG SLIP) DỰA TRÊN DATASHEET ---
def synchronizer_thread(stop_event):
    buffer = bytearray()
    print("[Sync] Đang kiểm tra Marker Bit để đồng bộ...")
    
    while not stop_event.is_set() or not RAW_QUEUE.empty():
        try:
            data = RAW_QUEUE.get(timeout=0.5)
            buffer.extend(data)
            
            # Mỗi mẫu XYZ là 9 byte
            while len(buffer) >= 9:
                # Kiểm tra Marker Bit theo Datasheet ADXL357
                # Byte 2 (cuối trục X) bit 0 phải = 1
                # Byte 5 (cuối trục Y) bit 0 phải = 0
                # Byte 8 (cuối trục Z) bit 0 phải = 0
                is_x = (buffer[2] & 0x01) == 1
                is_y_z = (buffer[5] & 0x01) == 0 and (buffer[8] & 0x01) == 0
                
                if is_x and is_y_z:
                    # Khớp frame 9 byte chuẩn
                    sample = buffer[:9]
                    CLEAN_QUEUE_FILE.put(sample)
                    CLEAN_QUEUE_AI.put(sample)
                    del buffer[:9]
                else:
                    # Nếu sai mốc, bỏ 1 byte và tìm lại (Trượt byte)
                    del buffer[0]
                    
            RAW_QUEUE.task_done()
        except queue.Empty:
            continue

# --- THREAD 3: GHI FILE THEO NGÀY & ĐO TẦN SỐ ---
def file_logger_thread(stop_event):
    current_date = ""
    f = None
    count = 0
    start_time = time.time()
    
    try:
        while not stop_event.is_set() or not CLEAN_QUEUE_FILE.empty():
            try:
                sample = CLEAN_QUEUE_FILE.get(timeout=0.5)
                
                # Tính Hz
                count += 1
                now_time = time.time()
                if now_time - start_time >= 1.0:
                    print(f"--- Tốc độ: {count} Hz ---")
                    count = 0
                    start_time = now_time

                # Ghi file
                date_str = datetime.now().strftime("%Y_%m_%d")
                if date_str != current_date:
                    if f: f.close()
                    current_date = date_str
                    f = open(f"log__{current_date}.bin", "ab")
                
                f.write(sample)
                CLEAN_QUEUE_FILE.task_done()
            except queue.Empty:
                continue
    finally:
        if f: f.close()
        print("[Logger] File đã lưu.")

# --- THREAD 4: ĐẨY VÀO AI (DỰ ĐOÁN REALTIME) ---
def ai_processor_thread(stop_event):
    print("[AI] Luồng AI đang chạy...")
    input_buffer = []
    
    while not stop_event.is_set():
        try:
            sample = CLEAN_QUEUE_AI.get(timeout=0.5)
            
            # Giải mã 20-bit từ 3 byte (Datasheet trang 34-36)
            # Hàm này chuyển 3 byte thô thành 1 giá trị số thực
            def bytes_to_accel(b):
                val = (b[0] << 12) | (b[1] << 4) | (b[2] >> 4)
                if val & 0x80000: # Xử lý số âm (2's complement 20-bit)
                    val -= 0x100000
                return val * 0.0000195 # Scale cho dải 10g (tùy chọn)

            x = bytes_to_accel(sample[0:3])
            y = bytes_to_accel(sample[3:6])
            z = bytes_to_accel(sample[6:9])
            
            input_buffer.append([x, y, z])
            
            # Ví dụ: Gom đủ 50 mẫu thì đưa vào Model AI dự đoán
            if len(input_buffer) >= 50:
                # result = model.predict(input_buffer) 
                # print(f"AI Predict: {result}")
                input_buffer = [] 

            CLEAN_QUEUE_AI.task_done()
        except queue.Empty:
            continue

# --- MAIN ---
def main():
    stop_event = threading.Event()
    t_reader = threading.Thread(target=serial_reader_thread, args=(stop_event,))
    t_sync = threading.Thread(target=synchronizer_thread, args=(stop_event,))
    t_file = threading.Thread(target=file_logger_thread, args=(stop_event,))
    t_ai = threading.Thread(target=ai_processor_thread, args=(stop_event,))

    threads = [t_reader, t_sync, t_file, t_ai]
    for t in threads: t.start()

    print("Bấm Ctrl + C để dừng...")
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    for t in threads: t.join()
    print("Hoàn tất.")

if __name__ == "__main__":
    main()