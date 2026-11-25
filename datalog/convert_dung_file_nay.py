import csv
import os

# ===================== CẤU HÌNH =====================
INPUT_FILE = r"../log__2026_11_02.bin"         # File .bin từ Pico
OUTPUT_FILE = "log__2026_11_02.csv" # File kết quả cuối cùng
SENSITIVITY = 51200.0            # 51200 LSB/g (Dành cho dải 10g)
# ===================================================

def parse_to_g(b1, b2, b3):
    """Giải mã chuẩn 20-bit Signed từ 3 byte SPI"""
    # 1. Ghép 3 byte thành 24-bit
    val_24 = (b1 << 16) | (b2 << 8) | b3
    # 2. Dịch phải 4 bit (bỏ Marker/Empty/Virtual) lấy 20-bit dữ liệu
    accel_raw = val_24 >> 4
    # 3. Xử lý số bù 2 cho 20-bit
    if accel_raw >= 0x80000: # 2^19
        accel_raw -= 0x100000 # 2^20
    # 4. Chia 51200 để ra đơn vị g
    return round(accel_raw / SENSITIVITY, 6)

def run_conversion():
    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: Không tìm thấy file {INPUT_FILE}")
        return

    with open(INPUT_FILE, "rb") as f:
        raw_bytes = f.read()

    total_len = len(raw_bytes)
    if total_len < 9: return

    # --- BƯỚC 1: TỰ ĐỘNG ĐỒNG BỘ (ANTI-SLIP) ---
    # Tìm điểm bắt đầu chuẩn: X(Marker=1), Y(Marker=0), Z(Marker=0)
    start_idx = -1
    for i in range(total_len - 8):
        if (raw_bytes[i+2] & 0x01) == 1 and \
           (raw_bytes[i+5] & 0x01) == 0 and \
           (raw_bytes[i+8] & 0x01) == 0:
            start_idx = i
            break

    if start_idx == -1:
        print("Không tìm thấy chu kỳ X-Y-Z chuẩn!")
        return

    print(f"Đã đồng bộ tại byte: {start_idx}")
    data = raw_bytes[start_idx:]
    num_samples = len(data) // 9

    # --- BƯỚC 2: TÍNH TOÁN VÀ GHI CSV ---
    with open(OUTPUT_FILE, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["x_value", "y_value", "z_value"]) # Chỉ lưu 3 cột decimal

        for i in range(num_samples):
            chunk = data[i*9 : (i+1)*9]
            if len(chunk) < 9: break
            
            # Tính toán trực tiếp
            gx = parse_to_g(chunk[0], chunk[1], chunk[2])
            gy = parse_to_g(chunk[3], chunk[4], chunk[5])
            gz = parse_to_g(chunk[6], chunk[7], chunk[8])
            
            writer.writerow([gx, gy, gz])

    print(f"Thành công! Đã lưu {num_samples} mẫu vào {OUTPUT_FILE}")

if __name__ == "__main__":
    run_conversion()

