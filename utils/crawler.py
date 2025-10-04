import os
import pandas as pd
from lightkurve import search_lightcurve
from tqdm import tqdm

def download_lightcurves(input_file, mission, row, output_dir="datas/lightcurves"):
    # === 1️⃣ Cấu hình cơ bản ===
    INPUT_FILE = input_file
    OUTPUT_DIR = os.path.join(output_dir, mission)
    MISSION = mission

    # Tạo thư mục lưu file nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 2️⃣ Đọc dữ liệu CSV ===
    df = pd.read_csv(INPUT_FILE, comment='#')

    # Lọc danh sách ID hợp lệ
    ids = df[row].dropna().unique()

    print(f"Tổng cộng {len(ids)} đối tượng EPIC cần tải.")

    # === 3️⃣ Tải từng light curve ===
    for id in tqdm(ids, desc=f"Downloading {MISSION} light curves"):
        try:
            # Đảm bảo ID có dạng "EPIC 201111557"
            if mission.strip().lower() == "k2" and not str(id).startswith("EPIC"):
                id = f"EPIC {id}"
            if mission.strip().lower() == "tess" and not str(id).startswith("TIC"):
                id = f"TIC {id}"
            if mission.strip().lower() == "kepler" and not str(id).startswith("KIC"):
                id = f"KIC {id}"

            # Tìm kiếm light curve
            search_result = search_lightcurve(id, mission=MISSION)

            if len(search_result) == 0:
                print(f"[⚠️] Không tìm thấy light curve cho {id}")
                continue

            # Tải và lưu file FITS
            lc = search_result.download()[-1].PDCSAP_FLUX
            filename = os.path.join(OUTPUT_DIR, f"{id.replace(' ', '_')}.fits")
            lc.to_fits(filename, overwrite=True)

        except Exception as e:
            print(f"[❌] Lỗi với {id}: {e}")

    print(F"✅ Hoàn tất tải toàn bộ light curves của {MISSION}!")

if __name__ == "__main__":
    # Ví dụ sử dụng hàm
    download_lightcurves(
        input_file="datas\summary\k2.csv",
        mission="K2",
        row="epic_hostname",
        output_dir="datas/lightcurves"
    )

    download_lightcurves(
        input_file="datas\summary\kepler.csv",
        mission="Kepler",
        row="kepid",
        output_dir="datas/lightcurves"
    )

    download_lightcurves(
        input_file=r"datas\summary\tess.csv",
        mission="TESS",
        row="tid",
        output_dir="datas/lightcurves"
    )

