"""
Tách file .bin của cảm biến ADXL357 thành 2 loại dữ liệu để TRAIN:
  - merged_GOOD.bin  : các đoạn máy chạy bình thường
  - merged_FAULT.bin : các đoạn máy bị lỗi (đột biến)
Mỗi đoạn còn được lưu thành 1 file riêng, kèm segments.csv ghi ranh giới từng đoạn
(quan trọng: khi train, chỉ cắt cửa sổ BÊN TRONG từng đoạn, không cắt qua chỗ nối).
Vùng LỖI lấy từ file fault_list.csv (giờ lỗi thật do bạn cung cấp, mỗi dòng "giờ_đầu, giờ_cuối").
Công cụ hiển thị và kiểm tra xem dữ liệu lúc đó có thật sự lệch so với bình thường không.
Vùng TỐT = toàn file trừ vùng lỗi và GUARD_SEC (mặc định 5 phút) đệm TRƯỚC và SAU mỗi vùng lỗi.

Cách làm:
  1. Tính RMS + đỉnh (peak) mỗi OVERVIEW_SEC giây cho 3 trục.
  2. Tự xác định "đường đi bình thường" bằng median trượt, đo độ lệch bằng MAD
     (bền vững với chính các điểm lỗi). Điểm nào lệch quá K_SIGMA => nghi lỗi.
     Ngoài ra, đoạn mất đồng bộ frame (marker X=1,Y=0,Z=0 sai) cũng bị đánh dấu.
  3. Gộp các điểm nghi lỗi gần nhau thành vùng, nới thêm PAD_SEC mỗi phía.
  4. Hiển thị lên đồ thị: bạn CLICK PHẢI để bỏ vùng báo nhầm, CLICK TRÁI để
     soi chi tiết, và thêm vùng lỗi thủ công trong cửa sổ chi tiết.
  5. Bấm 'S': xuất file TỐT, file LỖI, từng đoạn riêng và segments.csv.

Phím: S = xuất file | U = hoàn tác | P = in danh sách vùng lỗi
      G = nhận các đoạn "lạ nhất" do stumpy gợi ý (màu cam) vào vùng lỗi
      R = tinh chỉnh ranh giới các vùng lỗi tới điểm đổi trạng thái gần nhất (ruptures)
"""
import os
import csv
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== CẤU HÌNH =====================
INPUT_BIN = "log__2026_11_02.bin"
START_TIME_STR = "08:00:00"    # Giờ bắt đầu thu của file
SCALE_FACTOR = 0.0000195       # Hệ số ADXL357 (±10g)
SAMPLE_RATE = 4000             # Hz
FRAME_BYTES = 9                # 3 trục x 3 byte

OVERVIEW_SEC = 10              # Mỗi điểm trên bản đồ tổng quan = 10 giây
ZOOM_MINUTES = 10              # Cửa sổ chi tiết rộng 10 phút
ZOOM_STEP = 40                 # Cửa sổ chi tiết: 40 mẫu (0.01 s) -> 1 cột min/max

# ---- Danh sách khoảng giờ lỗi THẬT (nhãn đáng tin) ----
FAULT_LIST_FILE = "fault_list.csv"   # mỗi dòng: giờ_đầu, giờ_cuối  (VD: 10:15:00, 10:42:30)
USE_AUTO_DETECT = False              # True: gộp thêm cả vùng do công cụ tự phát hiện
START_DATE = ""                      # Ngày bắt đầu thu của file, VD "2026-11-05". Để trống = bỏ qua cột ngày
POINT_BEFORE_SEC = 60                # Sự kiện chỉ có 1 thời điểm: lấy trước đó bao nhiêu giây
POINT_AFTER_SEC = 60                 #   ... và sau đó bao nhiêu giây (HÃY ĐẶT THEO THỜI GIAN LỖI THỰC TẾ)

# ---- Danh sách khoảng DỪNG máy (không phải lỗi, cũng không phải "tốt") ----
STOP_LIST_FILE = "stop_list.csv"     # cùng định dạng fault_list.csv; có thể dán thẳng các dòng từ bảng log dừng
STOP_PAD_SEC = 30                    # dừng ghi 1 thời điểm: lấy trước/sau ngần này giây
STOP_GUARD_SEC = 60                  # chừa thêm quanh khoảng dừng khi lấy dữ liệu TỐT (máy chưa ổn định lại)
DROP_ANOMALY_NEAR_STOP = True        # Khi tự phát hiện lỗi: bỏ vùng bất thường sát khoảng dừng (do dừng/chạy lại)
FILE_TIME_TOLERANCE_SEC = 120        # Kiểm tra giờ theo Properties của file: lệch quá ngần này giây thì cảnh báo

# ---- Gợi ý đoạn "lạ nhất" bằng stumpy (matrix profile) ----   (cài: pip install stumpy)
USE_STUMPY = True
STUMPY_WINDOW_SEC = 60               # độ dài mỗi đoạn đem so sánh
STUMPY_TOP_K = 10                    # số đoạn gợi ý tối đa (màu CAM)
STUMPY_MIN_LOG_DIFF = 0.2            # chỉ gợi ý đoạn mà trung bình mỗi điểm lệch >= ~20% RMS so với đoạn giống nhất

# ---- Tinh chỉnh ranh giới vùng lỗi bằng ruptures (bấm R) ----   (cài: pip install ruptures)
REFINE_PEN = 100                     # độ "khó tính": tăng => ít điểm đổi hơn (100-300 là hợp lý ở dữ liệu thử)
REFINE_MARGIN_SEC = 120              # lấy thêm ngần này giây (hoặc bằng độ dài vùng) mỗi phía để so sánh
REFINE_SNAP_SEC = 60                 # chỉ dịch ranh giới tối đa ngần này giây
REFINE_MAX_WINDOW_MIN = 20           # vùng dài quá mức này thì không tinh chỉnh (ruptures chậm)

# ---- Tham số phát hiện lỗi tự động (dùng để KIỂM TRA / gợi ý, chỉnh nếu báo thừa/thiếu) ----
K_SIGMA = 6.0                  # Lệch quá bao nhiêu "sigma" thì là bất thường (tăng => ít báo hơn)
BASELINE_WINDOW_MIN = 30       # Cửa sổ tính "đường đi bình thường" (phút)
MIN_REL_SIGMA = 0.05           # Sigma tối thiểu = 5% giá trị bình thường (tránh quá nhạy khi tín hiệu rất phẳng)
MERGE_GAP_SEC = 60             # Hai điểm lỗi cách nhau <= 60 s thì gộp thành 1 vùng
PAD_SEC = 30                   # Nới mỗi vùng lỗi thêm 30 s mỗi phía
SYNC_BAD_FRACTION = 0.01       # >1% frame sai marker trong 1 khoảng => lỗi mất đồng bộ

# ---- Vùng TỐT (dùng làm nhãn "bình thường" khi train) ----
GUARD_SEC = 300                # Chừa đệm 5 phút TRƯỚC và SAU mỗi khoảng lỗi, KHÔNG lấy làm "tốt"
                               # (đoạn chuyển tiếp quanh lỗi có thể chưa sạch)
MIN_GOOD_SEC = 60              # Đoạn tốt ngắn hơn 60 s thì bỏ
# ====================================================

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------
def find_sync_offset(bin_path):
    """Tìm điểm đồng bộ chuẩn X=1, Y=0, Z=0, lặp đúng ở 8 frame liên tiếp."""
    with open(bin_path, "rb") as f:
        header = f.read(10240)
    n_check = 8
    last_start = len(header) - FRAME_BYTES * n_check
    for i in range(max(0, last_start)):
        ok = True
        for k in range(n_check):
            b = i + k * FRAME_BYTES
            if not ((header[b + 2] & 1) == 1 and
                    (header[b + 5] & 1) == 0 and
                    (header[b + 8] & 1) == 0):
                ok = False
                break
        if ok:
            return i
    raise ValueError("Không tìm thấy điểm đồng bộ Marker chuẩn!")


def decode_axis(raw, axis):
    """Giải mã 20-bit có dấu của 1 trục từ mảng uint8 (N, 9) -> g (float32)."""
    o = axis * 3
    val = ((raw[:, o].astype(np.int32) << 12) |
           (raw[:, o + 1].astype(np.int32) << 4) |
           (raw[:, o + 2].astype(np.int32) >> 4))
    val = np.where(val & 0x80000, val - 0x100000, val)
    return (val * SCALE_FACTOR).astype(np.float32)


def build_overview(bin_path, start_offset):
    """Trả về RMS, peak (N,3), tỉ lệ frame sai marker (N,) và memmap toàn file."""
    file_size = os.path.getsize(bin_path) - start_offset
    sps = SAMPLE_RATE * OVERVIEW_SEC
    total_steps = file_size // (sps * FRAME_BYTES)
    if total_steps < 1:
        raise ValueError("File quá ngắn so với OVERVIEW_SEC.")

    mm = np.memmap(bin_path, dtype=np.uint8, mode="r",
                   offset=start_offset, shape=(total_steps * sps, FRAME_BYTES))

    rms = np.zeros((total_steps, 3), dtype=np.float32)
    peak = np.zeros((total_steps, 3), dtype=np.float32)
    bad_frac = np.zeros(total_steps, dtype=np.float32)

    steps_per_batch = 30
    for s0 in range(0, total_steps, steps_per_batch):
        s1 = min(s0 + steps_per_batch, total_steps)
        raw = np.asarray(mm[s0 * sps: s1 * sps])

        bad = (((raw[:, 2] & 1) != 1) | ((raw[:, 5] & 1) != 0) | ((raw[:, 8] & 1) != 0))
        bad_frac[s0:s1] = bad.reshape(-1, sps).mean(axis=1)

        for axis in range(3):
            blocks = decode_axis(raw, axis).reshape(-1, sps)
            ac = blocks - blocks.mean(axis=1, keepdims=True)
            rms[s0:s1, axis] = np.sqrt(np.mean(ac ** 2, axis=1))
            peak[s0:s1, axis] = np.abs(ac).max(axis=1)
    return rms, peak, bad_frac, mm


# ---------------------------------------------------------------
# Phát hiện lỗi tự động
# ---------------------------------------------------------------
def robust_score(x, window):
    """Độ lệch (đơn vị sigma bền vững) của x so với median trượt."""
    window = max(5, int(window) | 1)  # số lẻ
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, window)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1)
    sigma = np.maximum.reduce([1.4826 * mad,
                               MIN_REL_SIGMA * np.abs(med),
                               np.full_like(med, 1e-6)])
    return np.abs(x - med) / sigma


def detect_regions(rms_xyz, peak_xyz, bad_frac):
    """Trả về (danh sách vùng lỗi [{s,e,src,score}] tính theo mẫu, mảng score)."""
    n = len(rms_xyz)
    rms_tot = np.sqrt((rms_xyz.astype(np.float64) ** 2).sum(axis=1))
    peak_tot = peak_xyz.max(axis=1).astype(np.float64)

    window = BASELINE_WINDOW_MIN * 60 / OVERVIEW_SEC
    score = np.maximum(robust_score(rms_tot, window), robust_score(peak_tot, window))

    flag_amp = score > K_SIGMA
    flag_sync = bad_frac > SYNC_BAD_FRACTION
    flags = flag_amp | flag_sync
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return [], score

    gap_steps = int(round(MERGE_GAP_SEC / OVERVIEW_SEC))
    pad_steps = int(np.ceil(PAD_SEC / OVERVIEW_SEC))
    sps = SAMPLE_RATE * OVERVIEW_SEC

    groups = np.split(idx, np.flatnonzero(np.diff(idx) > gap_steps + 1) + 1)
    regions = []
    for g in groups:
        a = max(0, g[0] - pad_steps)
        b = min(n, g[-1] + 1 + pad_steps)
        src = "mat_dong_bo" if flag_sync[g].any() else "dot_bien"
        regions.append({"s": int(a * sps), "e": int(b * sps),
                        "src": src, "score": float(score[g].max())})
    return merge_regions(regions), score


def check_file_time(path, total_samples, base_dt):
    """Kiểm tra giờ bằng ngày giờ của file (Properties trong Windows).
    - Giờ SỬA LẦN CUỐI (mtime) ~ lúc logger ghi mẫu cuối cùng.
    - Giờ bắt đầu suy ra = mtime - (số mẫu / 4000). So với START_TIME_STR bạn khai.
    - Trên Windows còn so cả 'Ngày tạo' để phát hiện logger tắt/mất mẫu giữa chừng.
    Lưu ý: sao chép file có thể đổi 'Ngày tạo' (thành lúc sao chép); mtime thường được giữ nguyên."""
    dur = total_samples / SAMPLE_RATE
    st = os.stat(path)
    mtime = datetime.fromtimestamp(st.st_mtime)
    implied = mtime - timedelta(seconds=dur)
    d = ((implied.hour * 3600 + implied.minute * 60 + implied.second)
         - (base_dt.hour * 3600 + base_dt.minute * 60 + base_dt.second))
    d = (d + 43200) % 86400 - 43200            # chênh lệch giờ trong ngày, về khoảng [-12h, 12h)
    print(f"[Kiểm tra giờ theo file] sửa lần cuối {mtime:%d/%m/%Y %H:%M:%S}; dữ liệu dài {dur / 3600:.2f} giờ "
          f"=> giờ bắt đầu suy ra {implied:%H:%M:%S} (bạn khai {base_dt:%H:%M:%S}).")
    if os.name == "nt":
        created0 = datetime.fromtimestamp(st.st_ctime)
        span0 = (mtime - created0).total_seconds()
        if 0 <= span0 < 0.5 * dur:
            print(f"  CẢNH BÁO: từ 'Ngày tạo' ({created0:%H:%M:%S}) đến 'Ngày sửa' chỉ {span0 / 60:.1f} phút, "
                  f"nhưng file chứa {dur / 3600:.2f} giờ dữ liệu.")
            print("  => File này KHÔNG được thu trực tiếp theo thời gian thực (nhiều khả năng do script sinh ra / "
                  "giả lập, hoặc ghi dồn một lần). Giờ bắt đầu suy ra từ Properties KHÔNG đáng tin; "
                  "đừng dùng nó để đối chiếu log.")
            return
    if abs(d) > FILE_TIME_TOLERANCE_SEC:
        print(f"  CẢNH BÁO: lệch {d:+.0f} giây ({d / 60:+.1f} phút). Có thể START_TIME_STR sai, hoặc logger đã "
              f"mất mẫu / dừng giữa chừng, hoặc file đã bị ghi/sao chép làm đổi giờ sửa đổi.")
    else:
        print(f"  OK: khớp trong ±{FILE_TIME_TOLERANCE_SEC} giây (chưa chứng minh không mất mẫu giữa chừng).")
    if os.name == "nt":
        created = datetime.fromtimestamp(st.st_ctime)
        span = (mtime - created).total_seconds()
        gap = span - dur
        print(f"  Ngày tạo {created:%d/%m/%Y %H:%M:%S}: khoảng tạo→sửa {span / 3600:.2f} giờ, "
              f"dữ liệu {dur / 3600:.2f} giờ.")
        if gap > FILE_TIME_TOLERANCE_SEC:
            print(f"  Thời gian dài hơn dữ liệu ~{gap / 60:.1f} phút: logger có thể đã tắt hoặc mất mẫu "
                  f"(hoặc 'Ngày tạo' là lúc sao chép file).")
        elif gap < -FILE_TIME_TOLERANCE_SEC:
            print(f"  Dữ liệu dài hơn khoảng tạo→sửa {-gap / 60:.1f} phút: 'Ngày tạo' có vẻ bị đổi (sao chép) "
                  f"hoặc tần số mẫu thật khác {SAMPLE_RATE} Hz.")


def stumpy_suggest(rms_tot, avoid, total_samples):
    """Xếp hạng các đoạn LẠ NHẤT bằng matrix profile (không chuẩn hoá biên độ, trên log RMS).
    avoid: các khoảng (có 's','e') cần bỏ qua (khoảng dừng + đệm, vùng lỗi đã có).
    Trả về danh sách vùng {s,e,src,score}, score = khoảng cách matrix profile (càng cao càng lạ)."""
    try:
        import stumpy
    except ImportError:
        print("(chưa cài stumpy -> bỏ qua gợi ý. Cài bằng: pip install stumpy)")
        return []
    m = max(3, int(round(STUMPY_WINDOW_SEC / OVERVIEW_SEC)))
    if len(rms_tot) < 4 * m:
        return []
    series = np.log(np.maximum(np.asarray(rms_tot, dtype=np.float64), 1e-9))
    d = stumpy.stump(series, m, normalize=False)[:, 0].astype(float)
    d[~np.isfinite(d)] = -np.inf
    sps = SAMPLE_RATE * OVERVIEW_SEC
    out = []
    for _ in range(len(d)):
        if len(out) >= STUMPY_TOP_K:
            break
        i = int(np.argmax(d))
        if not np.isfinite(d[i]):
            break
        val = float(d[i])
        if val < STUMPY_MIN_LOG_DIFF * np.sqrt(m):      # còn lại chỉ là nhiễu, không gợi ý nữa
            break
        d[max(0, i - 2 * m): i + 2 * m] = -np.inf      # vùng loại trừ để không chọn trùng
        s, e = i * sps, min(total_samples, (i + m) * sps)
        if any(e > a["s"] and s < a["e"] for a in avoid):
            continue
        out.append({"s": s, "e": e, "src": "stumpy", "score": val})
    return out


def refine_boundaries(region, mm, total_samples):
    """Tinh chỉnh 2 ranh giới của một vùng bằng ruptures (Pelt) trên đặc trưng mỗi giây quanh vùng.
    Trả về (vùng_mới_hoặc_chính_vùng_cũ, thông_báo). Chỉ dịch ranh giới tới điểm đổi gần nhất
    trong REFINE_SNAP_SEC; nếu không có điểm đổi thì giữ nguyên."""
    try:
        import ruptures as rpt
    except ImportError:
        return region, "chưa cài ruptures (pip install ruptures)"
    sr = SAMPLE_RATE
    length = region["e"] - region["s"]
    margin = int(max(REFINE_MARGIN_SEC * sr, length))
    w0 = max(0, region["s"] - margin) // sr * sr
    w1 = min(total_samples, region["e"] + margin)
    n_sec = (w1 - w0) // sr
    if n_sec * sr > REFINE_MAX_WINDOW_MIN * 60 * sr:
        return region, f"vùng quá dài (> {REFINE_MAX_WINDOW_MIN} phút) nên bỏ qua"
    if n_sec < 20:
        return region, "đoạn quá ngắn để tinh chỉnh"
    raw = np.asarray(mm[w0:w0 + n_sec * sr])
    rms = np.zeros((n_sec, 3)); peak = np.zeros((n_sec, 3)); kurt = np.zeros((n_sec, 3))
    for a in range(3):
        x = decode_axis(raw, a).astype(np.float64).reshape(n_sec, sr)
        xc = x - x.mean(axis=1, keepdims=True)
        v = (xc ** 2).mean(axis=1)
        rms[:, a] = np.sqrt(v)
        peak[:, a] = np.abs(xc).max(axis=1)
        kurt[:, a] = (xc ** 4).mean(axis=1) / (v ** 2 + 1e-18) - 3.0

    def rstd(x, floor):
        med = np.median(x)
        return (x - med) / max(1.4826 * np.median(np.abs(x - med)), floor)

    feat = np.column_stack([
        rstd(np.log(np.sqrt((rms ** 2).sum(axis=1)) + 1e-9), 0.02),
        rstd(np.log(peak.max(axis=1) + 1e-9), 0.02),
        rstd(kurt.mean(axis=1), 0.1)])
    bk = rpt.Pelt(model="l2", min_size=2, jump=1).fit(feat).predict(pen=REFINE_PEN)[:-1]
    if not bk:
        return region, "không thấy điểm đổi nào, giữ nguyên"
    bk = np.array(bk, dtype=float)
    s_sec = (region["s"] - w0) / sr
    e_sec = (region["e"] - w0) / sr

    def snap(t):
        j = int(np.argmin(np.abs(bk - t)))
        return float(bk[j]) if abs(bk[j] - t) <= REFINE_SNAP_SEC else t

    ns, ne = snap(s_sec), snap(e_sec)
    if ne <= ns or (ns == s_sec and ne == e_sec):
        return region, "ranh giới đã khớp điểm đổi / không có điểm đổi gần"
    new = dict(region)
    new["s"] = int(w0 + ns * sr)
    new["e"] = int(w0 + ne * sr)
    if "ranh_gioi_ruptures" not in new["src"]:
        new["src"] = new["src"] + "+ranh_gioi_ruptures"
    return new, ""


def load_fault_list(path, base_dt, total_samples, src="danh_sach", before=None, after=None):
    """Đọc danh sách giờ lỗi thật. Mỗi dòng có thể là:
         10:15:00, 10:42:30                    (khoảng giờ)
         2026/11/06 02:38:13 02:38:13 ...      (có ngày; dòng dán thẳng từ bảng log, chỉ lấy 2 mốc giờ đầu)
         02:38:13                              (1 thời điểm)
         # dòng bắt đầu bằng # là chú thích
    Giờ đầu = giờ cuối (hoặc chỉ có 1 mốc) => mở rộng thành POINT_BEFORE_SEC trước, POINT_AFTER_SEC sau.
    Nếu khai START_DATE và dòng có ngày thì so khớp theo ngày; nếu không, giờ nhỏ hơn giờ bắt đầu
    file được hiểu là sang ngày hôm sau.
    Trả về (danh sách vùng lỗi tính theo mẫu, danh sách cảnh báo)."""
    import re
    start_tod = base_dt.hour * 3600 + base_dt.minute * 60 + base_dt.second
    file_sec = total_samples / SAMPLE_RATE

    def to_sec(t):
        p = [int(x) for x in t.split(":")]
        while len(p) < 3:
            p.append(0)
        return p[0] * 3600 + p[1] * 60 + p[2]

    regions, warns = [], []
    with open(path, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            times = re.findall(r"\d{1,2}:\d{2}(?::\d{2})?", line)
            if not times:
                if re.search(r"\d", line):   # dòng tiêu đề thuần chữ thì bỏ qua im lặng
                    warns.append(f"dòng {ln} không đọc được mốc giờ: '{line[:60]}'")
                continue
            ts = to_sec(times[0])
            te = to_sec(times[1]) if len(times) >= 2 else ts
            label = f"{times[0]}-{times[1] if len(times) >= 2 else times[0]}"
            dm = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", line)
            if dm and START_DATE:
                # Có ngày trong dòng và đã khai START_DATE: so khớp theo ngày + giờ tuyệt đối
                day = datetime(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
                midnight = base_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                s_off = (day - midnight).total_seconds() + ts - start_tod
            else:
                s_off = (ts - start_tod) % 86400
            e_off = s_off + ((te - ts) % 86400)
            if e_off == s_off:   # sự kiện tại 1 thời điểm (giờ đầu = giờ cuối) -> mở rộng thành cửa sổ
                s_off -= POINT_BEFORE_SEC if before is None else before
                e_off += POINT_AFTER_SEC if after is None else after
            if e_off <= 0 or s_off >= file_sec:
                warns.append(f"dòng {ln} ({label}{', ngày ' + dm.group(0) if dm else ''}) "
                             f"nằm NGOÀI thời gian của file, bỏ qua")
                continue
            if s_off < 0:
                warns.append(f"dòng {ln} ({label}) bắt đầu trước đầu file, đã cắt bớt")
                s_off = 0
            if e_off > file_sec:
                warns.append(f"dòng {ln} ({label}) vượt quá cuối file, đã cắt bớt")
                e_off = file_sec
            regions.append({"s": int(s_off * SAMPLE_RATE), "e": int(e_off * SAMPLE_RATE),
                            "src": src, "score": 0.0})
    return merge_regions(regions), warns


def compute_good(faults, total_samples, stops=()):
    """Vùng TỐT = toàn file trừ (vùng lỗi + GUARD_SEC đệm) và trừ (khoảng dừng + STOP_GUARD_SEC),
    bỏ đoạn quá ngắn."""
    guard = int(GUARD_SEC * SAMPLE_RATE)
    sguard = int(STOP_GUARD_SEC * SAMPLE_RATE)
    min_len = int(MIN_GOOD_SEC * SAMPLE_RATE)
    blocked = merge_regions(
        [{"s": max(0, r["s"] - guard), "e": min(total_samples, r["e"] + guard),
          "src": "", "score": 0.0} for r in faults] +
        [{"s": max(0, r["s"] - sguard), "e": min(total_samples, r["e"] + sguard),
          "src": "", "score": 0.0} for r in stops])
    good, pos = [], 0
    for r in blocked:
        if r["s"] - pos >= min_len:
            good.append({"s": pos, "e": r["s"], "src": "tu_dong", "score": 0.0})
        pos = max(pos, r["e"])
    if total_samples - pos >= min_len:
        good.append({"s": pos, "e": total_samples, "src": "tu_dong", "score": 0.0})
    return good


def merge_regions(regions):
    """Gộp các vùng chồng lấn / liền kề."""
    out = []
    for r in sorted(regions, key=lambda r: r["s"]):
        if out and r["s"] <= out[-1]["e"]:
            last = out[-1]
            last["e"] = max(last["e"], r["e"])
            last["score"] = max(last["score"], r["score"])
            if r["src"] not in last["src"].split("+"):
                last["src"] += "+" + r["src"]
        else:
            out.append(dict(r))
    return out


# ---------------------------------------------------------------
# Tiện ích giao diện
# ---------------------------------------------------------------
def toolbar_active(fig):
    tb = getattr(fig.canvas, "toolbar", None)
    if tb is None:
        return False
    return str(getattr(tb, "mode", "")) not in ("", "_Mode.NONE")


def attach_hover_tooltip(fig, axes_list, base_dt):
    """Đường gióng dọc + hộp giờ khi di chuột.
    Dùng blitting: mỗi lần chuột nhích chỉ vẽ lại vài đối tượng nhỏ, KHÔNG vẽ lại
    cả đồ thị (hàng chục nghìn điểm) nên không còn bị lag/xoay tròn."""
    base_num = mdates.date2num(base_dt)
    vlines = [ax.axvline(x=base_num, color="gray", linestyle=":", linewidth=1,
                         visible=False, animated=True)
              for ax in axes_list]
    annot = axes_list[0].annotate(
        "", xy=(0, 0), xytext=(12, -18), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="black", lw=0.8),
        fontsize=9, fontweight="bold", visible=False, animated=True)
    artists = vlines + [annot]
    use_blit = bool(getattr(fig.canvas, "supports_blit", False))
    state = {"bg": None}

    def on_draw(_event):
        # Mỗi lần đồ thị được vẽ đầy đủ (zoom, đổi cỡ cửa sổ, thêm vùng...) thì chụp lại nền
        if use_blit:
            state["bg"] = fig.canvas.copy_from_bbox(fig.bbox)

    def render():
        if use_blit and state["bg"] is not None:
            fig.canvas.restore_region(state["bg"])
            for a in artists:
                if a.get_visible():
                    fig.draw_artist(a)
            fig.canvas.blit(fig.bbox)
        else:
            fig.canvas.draw_idle()

    def on_move(event):
        inside = event.inaxes in axes_list and event.xdata is not None
        if not inside:
            if annot.get_visible():
                annot.set_visible(False)
                for vl in vlines:
                    vl.set_visible(False)
                render()
            return
        dt_cur = mdates.num2date(event.xdata).replace(tzinfo=None)
        elapsed = (event.xdata - base_num) * SECONDS_PER_DAY
        for vl in vlines:
            vl.set_xdata([event.xdata, event.xdata])
            vl.set_visible(True)
        annot.xy = (event.xdata, event.ydata)
        annot.set_text(f"Giờ: {dt_cur.strftime('%H:%M:%S.%f')[:-4]}\n(Mốc giây: {elapsed:,.1f}s)")
        annot.set_visible(True)
        render()

    fig.canvas.mpl_connect("draw_event", on_draw)
    fig.canvas.mpl_connect("motion_notify_event", on_move)


# ---------------------------------------------------------------
# Chương trình chính
# ---------------------------------------------------------------
def main():
    if not os.path.exists(INPUT_BIN):
        print(f"Không tìm thấy file {INPUT_BIN}")
        return

    day0 = START_DATE.replace("/", "-") if START_DATE else "2026-01-01"
    base_dt = datetime.strptime(f"{day0} {START_TIME_STR}", "%Y-%m-%d %H:%M:%S")
    base_num = mdates.date2num(base_dt)

    start_offset = find_sync_offset(INPUT_BIN)
    print(f"Điểm đồng bộ tại byte {start_offset}. Đang tính tổng quan (có thể mất vài phút)...")
    rms_xyz, peak_xyz, bad_frac, mm = build_overview(INPUT_BIN, start_offset)
    total_samples = len(mm)
    print(f"Xong: {len(rms_xyz)} điểm ({total_samples / SAMPLE_RATE / 3600:.2f} giờ dữ liệu).")
    check_file_time(INPUT_BIN, total_samples, base_dt)

    auto_regions, score = detect_regions(rms_xyz, peak_xyz, bad_frac)
    rms_tot = np.sqrt((rms_xyz.astype(np.float64) ** 2).sum(axis=1))

    # Vùng lỗi: ưu tiên danh sách giờ lỗi thật do bạn cung cấp
    list_path = next((p for p in (FAULT_LIST_FILE,
                                  os.path.join(os.path.dirname(os.path.abspath(INPUT_BIN)), FAULT_LIST_FILE))
                      if os.path.exists(p)), None)
    if list_path:
        faults, warns = load_fault_list(list_path, base_dt, total_samples)
        print(f"Đã đọc {len(faults)} khoảng giờ lỗi từ {list_path}")
        for w in warns:
            print("  CẢNH BÁO:", w)
        if USE_AUTO_DETECT:
            faults = merge_regions(faults + auto_regions)
    else:
        if USE_AUTO_DETECT:
            print(f"Không thấy file {FAULT_LIST_FILE} -> dùng vùng lỗi tự phát hiện (chỉ là GỢI Ý, chưa phải nhãn thật).")
            faults = list(auto_regions)
        else:
            print(f"Không thấy file {FAULT_LIST_FILE} và USE_AUTO_DETECT=False -> KHÔNG có vùng lỗi nào (không tự cắt).")
            faults = []
    stop_path = next((p for p in (STOP_LIST_FILE,
                                  os.path.join(os.path.dirname(os.path.abspath(INPUT_BIN)), STOP_LIST_FILE))
                      if os.path.exists(p)), None)
    stops = []
    if stop_path:
        stops, warns = load_fault_list(stop_path, base_dt, total_samples, src="dung_may",
                                       before=STOP_PAD_SEC, after=STOP_PAD_SEC)
        print(f"Đã đọc {len(stops)} khoảng DỪNG máy từ {stop_path} (sẽ loại khỏi dữ liệu TỐT)")
        for w in warns:
            print("  CẢNH BÁO:", w)
    if stops and not list_path and DROP_ANOMALY_NEAR_STOP:
        # Lúc máy dừng / chạy lại tín hiệu cũng nhảy -> bỏ các "bất thường" dính vào khoảng dừng,
        # nếu không phần lớn "lỗi" tìm được chỉ là chính việc dừng và khởi động.
        g = int(STOP_GUARD_SEC * SAMPLE_RATE)
        kept = [r for r in faults
                if not any(r["e"] > s["s"] - g and r["s"] < s["e"] + g for s in stops)]
        print(f"Bỏ {len(faults) - len(kept)} vùng bất thường nằm sát khoảng dừng máy "
              f"(giữ lại {len(kept)}).")
        faults = kept
    suggest = []                 # các đoạn lạ nhất do stumpy gợi ý (màu CAM trên bản đồ)
    if USE_STUMPY:
        gs = int(STOP_GUARD_SEC * SAMPLE_RATE)
        avoid = [{"s": s["s"] - gs, "e": s["e"] + gs} for s in stops] + list(faults)
        print("Đang xếp hạng các đoạn lạ nhất bằng stumpy (lần đầu có thể mất vài chục giây)...")
        suggest = stumpy_suggest(rms_tot, avoid, total_samples)
    history = []                 # để hoàn tác: ("add"/"del"/"replace", region)
    refreshers = []              # hàm vẽ lại của các cửa sổ đang mở

    def num_from_sample(s):
        return base_num + (s / SAMPLE_RATE) / SECONDS_PER_DAY

    def sample_from_num(x):
        sec = (x - base_num) * SECONDS_PER_DAY
        return max(0, min(total_samples, int(round(sec * SAMPLE_RATE))))

    def time_str(s):
        return (base_dt + timedelta(seconds=s / SAMPLE_RATE)).strftime("%H:%M:%S")

    def print_regions():
        print("\n--- DANH SÁCH VÙNG LỖI (kèm kiểm tra độ lệch so với bình thường) ---")
        good = compute_good(faults, total_samples, stops)
        sps = SAMPLE_RATE * OVERVIEW_SEC
        gmask = np.zeros(len(rms_tot), dtype=bool)
        for g in good:
            gmask[g["s"] // sps: g["e"] // sps] = True
        if gmask.any():
            base_rms = float(np.median(rms_tot[gmask]))
            print(f" Mức RMS bình thường (trung vị các đoạn tốt): {base_rms:.4f} g")
        else:
            base_rms = float(np.median(rms_tot))
            print(f" Mức RMS bình thường (trung vị TOÀN FILE, vì chưa có đoạn tốt nào): {base_rms:.4f} g")
        if not faults:
            print(" (trống)")
        for i, r in enumerate(sorted(faults, key=lambda r: r["s"]), 1):
            a = min(len(rms_tot) - 1, r["s"] // sps)
            b = min(len(rms_tot), max(a + 1, r["e"] // sps))
            seg_rms = rms_tot[a:b]
            seg_score = float(score[a:b].max())
            ratio = float(np.median(seg_rms)) / base_rms if base_rms > 0 else float("nan")
            peak_ratio = float(seg_rms.max()) / base_rms if base_rms > 0 else float("nan")
            verdict = "lệch rõ" if seg_score > K_SIGMA else "!! KHÔNG lệch rõ - kiểm tra lại giờ"
            print(f" {i:02d}. {time_str(r['s'])} -> {time_str(r['e'])}  "
                  f"({(r['e'] - r['s']) / SAMPLE_RATE:.0f}s)  RMS gấp {ratio:.1f}x (đỉnh {peak_ratio:.1f}x) "
                  f"bình thường, điểm bất thường {seg_score:.1f} -> {verdict}")
        tot = sum(r["e"] - r["s"] for r in merge_regions(faults)) / SAMPLE_RATE
        tot_g = sum(r["e"] - r["s"] for r in good) / SAMPLE_RATE
        print(f" Tổng thời gian lỗi: {tot:.0f}s | Tổng thời gian TỐT: {tot_g:.0f}s "
              f"({len(good)} đoạn) | Toàn file: {total_samples / SAMPLE_RATE:.0f}s")
        if suggest:
            print(" --- GỢI Ý đoạn lạ nhất (stumpy); bấm G để nhận vào vùng lỗi ---")
            for i, r in enumerate(suggest, 1):
                inside = any(r["e"] > f["s"] and r["s"] < f["e"] for f in faults)
                print(f"  #{i}: {time_str(r['s'])} -> {time_str(r['e'])}  độ lạ {r['score']:.2f}"
                      f"{'  (đã nằm trong vùng lỗi)' if inside else ''}")
        print()

    def refresh_all():
        for fn in list(refreshers):
            try:
                fn()
            except Exception:
                pass

    def add_region(r):
        faults.append(r)
        history.append(("add", r))
        refresh_all()

    def delete_region_at(sample):
        for r in faults:
            if r["s"] <= sample <= r["e"]:
                faults.remove(r)
                history.append(("del", r))
                refresh_all()
                return r
        return None

    def undo():
        if not history:
            print("Không còn thao tác để hoàn tác.")
            return
        act, r = history.pop()
        if act == "add" and r in faults:
            faults.remove(r)
        elif act == "del":
            faults.append(r)
        elif act == "replace":
            old, new = r
            if new in faults:
                faults.remove(new)
            faults.append(old)
        refresh_all()

    def adopt_suggestions():
        n = 0
        for r in suggest:
            if any(r["e"] > f["s"] and r["s"] < f["e"] for f in faults):
                continue
            add_region(dict(r))
            n += 1
        print(f"[+] Đã nhận {n} đoạn gợi ý vào vùng lỗi (bỏ qua đoạn đã nằm trong vùng lỗi). "
              f"Bấm chuột phải để bỏ đoạn không đúng, U để hoàn tác.")

    def refine_all():
        print("\n--- TINH CHỈNH RANH GIỚI bằng ruptures ---")
        changed = 0
        for r in sorted(list(faults), key=lambda r: r["s"]):
            new, msg = refine_boundaries(r, mm, total_samples)
            if new is not r:
                faults.remove(r)
                faults.append(new)
                history.append(("replace", (r, new)))
                changed += 1
                print(f" {time_str(r['s'])}->{time_str(r['e'])}  ==>  {time_str(new['s'])}->{time_str(new['e'])}")
            else:
                print(f" {time_str(r['s'])}->{time_str(r['e'])}: {msg}")
        refresh_all()
        print(f" Đã chỉnh {changed}/{len(faults)} vùng. Bấm U để hoàn tác (từng vùng).\n")

    # ---------------- Xuất file ----------------
    def write_range(fo, s, e):
        chunk = SAMPLE_RATE * 600
        for a in range(s, e, chunk):
            fo.write(np.asarray(mm[a:min(e, a + chunk)]).tobytes())

    def export_all():
        bad = merge_regions(faults)
        good = compute_good(faults, total_samples, stops)
        if not bad and not good and not stops:
            print("Không có đoạn nào để xuất.")
            return
        base = os.path.splitext(os.path.abspath(INPUT_BIN))[0]
        out_dir = base + "_cut"
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for label, regs, merged_name, sub in (("GOOD", good, "merged_GOOD.bin", "good_segments"),
                                              ("FAULT", bad, "merged_FAULT.bin", "fault_segments"),
                                              ("STOP", merge_regions(stops), "merged_STOP.bin", "stop_segments")):
            if not regs:
                continue
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
            offset = 0   # vị trí (tính bằng mẫu) của đoạn trong file merged
            with open(os.path.join(out_dir, merged_name), "wb") as fm:
                for i, r in enumerate(regs, 1):
                    fname = f"{label.lower()}_{i:03d}_{time_str(r['s']).replace(':', '')}.bin"
                    with open(os.path.join(out_dir, sub, fname), "wb") as fo:
                        write_range(fo, r["s"], r["e"])
                    write_range(fm, r["s"], r["e"])
                    n = r["e"] - r["s"]
                    rows.append([label, i, f"{sub}/{fname}", merged_name, offset, offset + n,
                                 r["s"], r["e"], time_str(r["s"]), time_str(r["e"]),
                                 f"{n / SAMPLE_RATE:.2f}", r["src"], f"{r['score']:.1f}"])
                    offset += n

        with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["nhan", "stt", "file_rieng", "file_gop", "mau_dau_trong_file_gop",
                        "mau_cuoi_trong_file_gop", "mau_dau_trong_file_goc", "mau_cuoi_trong_file_goc",
                        "gio_dau", "gio_cuoi", "thoi_luong_s", "nguon", "diem_bat_thuong"])
            w.writerows(rows)

        tg = sum(r["e"] - r["s"] for r in good) / SAMPLE_RATE
        tb = sum(r["e"] - r["s"] for r in bad) / SAMPLE_RATE
        print(f"\n--- ĐÃ XUẤT vào: {out_dir}")
        print(f" -> merged_GOOD.bin : {len(good)} đoạn, {tg:.0f} giây")
        print(f" -> merged_FAULT.bin: {len(bad)} đoạn, {tb:.0f} giây")
        if stops:
            ts_ = sum(r["e"] - r["s"] for r in merge_regions(stops)) / SAMPLE_RATE
            print(f" -> merged_STOP.bin : {len(merge_regions(stops))} đoạn, {ts_:.0f} giây (máy dừng, loại khỏi tốt/lỗi)")
        print(" -> good_segments/, fault_segments/ (mỗi đoạn 1 file) + segments.csv (ranh giới từng đoạn)")

    def on_key(event):
        k = (event.key or "").lower()
        if k == "s":
            export_all()
        elif k == "u":
            undo()
        elif k == "p":
            print_regions()
        elif k == "g":
            adopt_suggestions()
        elif k == "r":
            refine_all()

    # ---------------- Cửa sổ chi tiết ----------------
    def open_zoom_window(center_num):
        center_sample = sample_from_num(center_num)
        half = int(ZOOM_MINUTES * 60 * SAMPLE_RATE / 2)
        s0 = max(0, center_sample - half)
        e0 = min(total_samples, center_sample + half)
        n_blocks = (e0 - s0) // ZOOM_STEP
        if n_blocks < 1:
            print("Vùng chọn quá nhỏ để hiển thị.")
            return
        e0 = s0 + n_blocks * ZOOM_STEP
        sub = np.asarray(mm[s0:e0])

        xmin = np.zeros((n_blocks, 3), dtype=np.float32)
        xmax = np.zeros((n_blocks, 3), dtype=np.float32)
        for axis in range(3):
            g = decode_axis(sub, axis).reshape(-1, ZOOM_STEP)
            xmin[:, axis] = g.min(axis=1)
            xmax[:, axis] = g.max(axis=1)
        t2 = base_num + ((s0 + np.arange(n_blocks) * ZOOM_STEP) / SAMPLE_RATE) / SECONDS_PER_DAY

        fig2, axes2 = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        colors = ["#d62728", "#2ca02c", "#1f77b4"]
        names = ["Trục X (g)", "Trục Y (g)", "Trục Z (g)"]
        for i in range(3):
            axes2[i].fill_between(t2, xmin[:, i], xmax[:, i], color=colors[i], alpha=0.8,
                                  linewidth=0)
            axes2[i].set_ylabel(names[i])
            axes2[i].grid(True, alpha=0.3)
        axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes2[-1].set_xlabel("CHUỘT TRÁI (2 lần): thêm vùng LỖI | CHUỘT PHẢI (1 lần): xoá vùng lỗi "
                             "| U: hoàn tác | S: xuất file")
        fig2.suptitle(f"SOI CHI TIẾT QUANH {mdates.num2date(center_num).strftime('%H:%M:%S')}")
        attach_hover_tooltip(fig2, list(axes2), base_dt)

        spans = []
        pending = []   # [(x, sample, [vline artists])]

        def draw_spans():
            for p in spans:
                try:
                    p.remove()
                except Exception:
                    pass
            spans.clear()
            for r in compute_good(faults, total_samples, stops):
                if r["e"] < s0 or r["s"] > e0:
                    continue
                for ax in axes2:
                    spans.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                            color="green", alpha=0.10))
            for r in stops:
                if r["e"] < s0 or r["s"] > e0:
                    continue
                for ax in axes2:
                    spans.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                            color="gray", alpha=0.35))
            for r in faults:
                if r["e"] < s0 or r["s"] > e0:
                    continue
                for ax in axes2:
                    spans.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                            color="red", alpha=0.25))
            fig2.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is None or event.xdata is None or toolbar_active(fig2):
                return
            pos = sample_from_num(event.xdata)
            if event.button == 1:
                lines = [ax.axvline(event.xdata, color="red", linestyle="--", linewidth=1.2)
                         for ax in axes2]
                pending.append((pos, lines))
                if len(pending) == 2:
                    a, b = sorted(p[0] for p in pending)
                    for _, ls in pending:
                        for ln in ls:
                            ln.remove()
                    pending.clear()
                    if b > a:
                        add_region({"s": a, "e": b, "src": "thu_cong", "score": 0.0})
                        print(f"[+ VÙNG LỖI] {time_str(a)} -> {time_str(b)} ({(b - a) / SAMPLE_RATE:.2f}s)")
                fig2.canvas.draw_idle()
            elif event.button == 3:
                r = delete_region_at(pos)
                if r:
                    print(f"[- XOÁ] {time_str(r['s'])} -> {time_str(r['e'])}")
                else:
                    print("Không có vùng lỗi nào tại điểm vừa click.")

        def on_close(_):
            if draw_spans in refreshers:
                refreshers.remove(draw_spans)

        fig2.canvas.mpl_connect("button_press_event", on_click)
        fig2.canvas.mpl_connect("key_press_event", on_key)
        fig2.canvas.mpl_connect("close_event", on_close)
        refreshers.append(draw_spans)
        draw_spans()
        fig2.tight_layout(rect=(0, 0, 1, 0.96))
        fig2.show()

    # ---------------- Cửa sổ tổng quan ----------------
    t1 = base_num + (np.arange(len(rms_xyz)) * OVERVIEW_SEC) / SECONDS_PER_DAY
    fig1, (ax1, ax_s) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    for i, nm in enumerate(["RMS X", "RMS Y", "RMS Z"]):
        ax1.plot(t1, rms_xyz[:, i], label=nm, alpha=0.8)
    ax1.set_ylabel("RMS (g)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title(f"BẢN ĐỒ DỮ LIỆU (bắt đầu {START_TIME_STR}) — ĐỎ = lỗi, XÁM = dừng, XANH = tốt, CAM = gợi ý stumpy\n"
                  f"TRÁI: soi chi tiết | PHẢI: bỏ vùng | G: nhận gợi ý cam | R: tinh chỉnh ranh giới | "
                  f"U: hoàn tác | S: xuất | P: in danh sách", fontsize=10)

    ax_s.plot(t1, score, color="black", linewidth=0.8)
    ax_s.axhline(K_SIGMA, color="red", linestyle="--", linewidth=1, label=f"ngưỡng K={K_SIGMA:g}")
    ax_s.set_ylabel("Độ bất thường")
    ax_s.set_yscale("symlog", linthresh=K_SIGMA)
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(loc="upper right")
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_s.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_s.set_xlabel("Thời gian trong ngày (Giờ:Phút)")

    attach_hover_tooltip(fig1, [ax1, ax_s], base_dt)

    spans1 = []

    def draw_spans1():
        for p in spans1:
            try:
                p.remove()
            except Exception:
                pass
        spans1.clear()
        for r in compute_good(faults, total_samples, stops):
            for ax in (ax1, ax_s):
                spans1.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                         color="green", alpha=0.10))
        for r in stops:
            for ax in (ax1, ax_s):
                spans1.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                         color="gray", alpha=0.35))
        for i, r in enumerate(suggest, 1):
            if any(r["e"] > f["s"] and r["s"] < f["e"] for f in faults):
                continue
            for ax in (ax1, ax_s):
                spans1.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                         color="orange", alpha=0.35))
            spans1.append(ax1.annotate(f"#{i}", xy=(num_from_sample((r["s"] + r["e"]) // 2), 1.0),
                                       xycoords=("data", "axes fraction"), xytext=(0, -12),
                                       textcoords="offset points", ha="center", fontsize=8,
                                       color="darkorange", fontweight="bold"))
        for r in faults:
            for ax in (ax1, ax_s):
                spans1.append(ax.axvspan(num_from_sample(r["s"]), num_from_sample(r["e"]),
                                         color="red", alpha=0.25))
        fig1.canvas.draw_idle()

    def on_click1(event):
        if event.inaxes is None or event.xdata is None or toolbar_active(fig1):
            return
        if event.button == 1:
            open_zoom_window(event.xdata)
        elif event.button == 3:
            r = delete_region_at(sample_from_num(event.xdata))
            if r:
                print(f"[- XOÁ] {time_str(r['s'])} -> {time_str(r['e'])}")

    fig1.canvas.mpl_connect("button_press_event", on_click1)
    fig1.canvas.mpl_connect("key_press_event", on_key)
    refreshers.append(draw_spans1)
    draw_spans1()
    print_regions()
    fig1.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()