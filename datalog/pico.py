import machine, time, sys, uselect

# --- CẤU HÌNH ---
SPI_SPEED = 10_000_000 # Tăng lên 10MHz để Burst Read nhanh hơn
WATERMARK = 20         
PORT_SPEED = 1_000_000 # Baudrate USB

spi = machine.SPI(0, baudrate=SPI_SPEED, polarity=0, phase=0,
                  sck=machine.Pin(2), mosi=machine.Pin(3), miso=machine.Pin(4))
cs = machine.Pin(5, machine.Pin.OUT, value=1)
int1 = machine.Pin(15, machine.Pin.IN) 
led = machine.Pin('LED', machine.Pin.OUT)

def write_reg(reg, val):
    cs.low()
    spi.write(bytearray([(reg << 1), val]))
    cs.high()

def init_sensor():
    write_reg(0x2F, 0x52); time.sleep_ms(150) # Reset
    write_reg(0x28, 0x00) # 4000Hz / LPF 1000Hz
    write_reg(0x2C, 0x41) # Range 10g / Int Pol High
    write_reg(0x29, WATERMARK * 3) # Watermark logic
    write_reg(0x2A, 0x02) # Map Watermark to INT1
    write_reg(0x2D, 0x01) # Bắt đầu ở chế độ STANDBY

poll_obj = uselect.poll()
poll_obj.register(sys.stdin, uselect.POLLIN)

while True:
    init_sensor()
    # Xả sạch rác trong stdin
    while poll_obj.poll(0): sys.stdin.read(1)
    
    # --- CHẾ ĐỘ CHỜ ---
    while True:
        sys.stdout.write("READY\n")
        led.toggle()
        if poll_obj.poll(100): 
            cmd = sys.stdin.read(1)
            if cmd == 'S': break
            if cmd == 'E': init_sensor()

    # --- CHUẨN BỊ ĐO (Rất quan trọng) ---
    led.value(1)
    # Xả sạch FIFO cũ còn tồn đọng trước khi vào mode đo chính thức
    # Đọc tối đa dung lượng FIFO (96 samples * 3 axes * 3 bytes)
    cs.low(); spi.write(b'\x23'); spi.read(864); cs.high() 
    
    write_reg(0x2D, 0x00) # Vào MEASUREMENT MODE
    time.sleep_ms(10)     # Đợi cảm biến ổn định
    
    streaming = True
    # Cache lại các hàm để chạy nhanh hơn trong loop
    write_buffer = sys.stdout.buffer.write
    read_spi = spi.read
    burst_cmd = b'\x23'
    
    while streaming:
        # Sử dụng ngắt vật lý INT1
        if int1.value():
            cs.low()
            spi.write(burst_cmd)
            # Đọc đúng Watermark samples (180 bytes)
            data = read_spi(WATERMARK * 9) 
            cs.high()
            
            # Gửi thẳng vào pipe Serial USB của Jetson
            write_buffer(data)
        
        # Kiểm tra lệnh dừng 'E' từ Jetson (chạy AI xong hoặc user bấm Stop)
        if poll_obj.poll(0):
            if sys.stdin.read(1) == 'E':
                write_reg(0x2D, 0x01) # Quay về STANDBY
                led.value(0)
                streaming = False