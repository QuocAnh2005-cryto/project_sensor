import os
from datetime import datetime


def extract_bin_segment(input_file, output_file, file_start_str, file_end_str, cut_start_str, cut_end_str):
    time_format = "%Y-%m-%d %H:%M:%S"

    

    t_file_start = datetime.strptime(file_start_str, time_format).timestamp()
    t_file_end = datetime.strptime(file_end_str, time_format).timestamp()
    t_cut_start = datetime.strptime(cut_start_str, time_format).timestamp()
    t_cut_end = datetime.strptime(cut_end_str, time_format).timestamp()

    if t_cut_start < t_file_start or t_cut_end > t_file_end:
        print(f"Canh bao: Khung gio cat {cut_start_str} --> {cut_end_str}")
        print(f"Vuot qua pham vi file goc {file_start_str} --> {file_end_str}")

        if t_cut_start < t_file_start:
            t_cut_start = t_file_start
            cut_start_str = file_start_str
            print(f"Vi vuot qua nen can doi time bat dau {t_cut_start} --> {t_file_start}")
        
        if t_cut_end < t_file_end:
            t_cut_end = t_file_end
            cut_end_str = file_end_str
            print(f"Vi vuot qua nen can doi time bat dau {t_cut_end} --> {t_file_end}")
        print()

    
    file_size_bytes = os.path.getsize(input_file)
    total_duration_secs = t_file_end - t_file_start

    if total_duration_secs <= 0:
        print("Loi: Thoi gian file goc ko hop le")
        return
    
    bytes_per_sec = file_size_bytes / total_duration_secs

    byte_start = int((t_cut_start - t_file_start) * bytes_per_sec)
    byte_end = int((t_cut_end - t_file_start) * bytes_per_sec)
    byte_length = byte_end - byte_start


    byte_start = max(0, min(byte_start, file_size_bytes))
    byte_end = max(0, min(byte_length, file_size_bytes - byte_start))

    if byte_length == 0:
        print(f"Khung gio khong co du lieu hoac nam ngoai file")
        return
    
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(input_file, 'rb') as f_in:
        f_in.seek(byte_start)
        data = f_in.read(byte_length)


    with open(output_file, 'wb') as f_out:
        f_out.write(data)

    
    print(f"Successful: {os.path.basename(output_file)}")
    print(f"Time: {cut_start_str} -> {cut_end_str}")
    print(f"Dimension: {len(data)} byte\n")


if __name__=="__main__":
    INPUT_FILE = "log__2026_11_02.bin"
    FILE_START = "2026-11-02 14:10:00"
    FILE_END = "2026-11-02 23:59:59"
    OUTPUT_DIR = "output_bin"

    time_ranges = [
        {
            "name": "OK",
            "start": "2026-11-02 14:10:00",
            "end" : "2026-11-02 20:00:00"
        },

        {
            "name": "Time for Eating",
            "start": "2026-11-02 20:00:00",
            "end" : "2026-11-02 22:00:00"
        },

        {
            "name": "Time for Error",
            "start": "2026-11-02 22:00:00",
            "end" : "2026-11-02 23:59:59"
        },
    ]
    
    print(f"-------Starting process file:{INPUT_FILE}--------\n")

    for i, range in enumerate(time_ranges, 1):
        out_file = os.path.join(OUTPUT_DIR, f"{i}_{range['name']}.bin")

        extract_bin_segment(
            input_file=INPUT_FILE,
            output_file=out_file,
            file_start_str=FILE_START,
            file_end_str=FILE_END,
            cut_start_str=range['start'],
            cut_end_str=range['end']
        )
    
    print(f"-------Done ------\n")