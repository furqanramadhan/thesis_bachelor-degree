import os
import re
import pandas as pd

def convert_ascii_to_csv(input_file, output_file=None):
    """
    Mengkonversi file ASCII dari data buoy RAMA menjadi format CSV.
    Jika ada informasi kedalaman, maka akan diproses seperti t0n90e_dy.ascii.
    Jika tidak ada informasi kedalaman, hanya menyimpan Timestamp.

    Parameters:
    input_file (str): Path ke file ASCII
    output_file (str, optional): Path untuk menyimpan file CSV hasil.

    Returns:
    str: Path ke file CSV yang dihasilkan
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.csv'
    
    # Baca seluruh isi file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Cek apakah ada informasi kedalaman
    depth_line = None
    for line in lines:
        if 'Depth(M):' in line:
            depth_line = line.strip()
            break
    
    # **Jika ada informasi kedalaman, proses seperti t0n90e_dy.ascii**
    if depth_line:
        print(f"✅ Ditemukan informasi kedalaman di {input_file}")
        
        # Ekstrak nilai kedalaman sebagai nama kolom
        depth_parts = depth_line.split(':')[1].strip().split()
        depth_values = [f"TEMP_{part}m" if part.isdigit() else "SST" for part in depth_parts]
        
        # Buat struktur untuk data
        data_rows = []
        
        for line in lines:
            if re.match(r'^\s*\d{8}\s+\d{4}', line):
                parts = line.strip().split()
                date, time = parts[0], parts[1]
                
                # Temukan awal QUALITY/SOURCE
                quality_start_idx = next((i for i, val in enumerate(parts[2:], 2) if re.match(r'^[1-5]+$', val) and len(val) > 8), -1)
                temp_values = parts[2:quality_start_idx] if quality_start_idx != -1 else parts[2:]
                
                # Pastikan jumlah suhu sesuai jumlah kedalaman
                temp_values += ['NaN'] * (len(depth_values) - len(temp_values))
                temp_values = temp_values[:len(depth_values)]
                
                # Simpan data dalam format dictionary
                row_data = {'YYYYMMDD': date, 'HHMM': time, **dict(zip(depth_values, temp_values))}
                data_rows.append(row_data)

        df = pd.DataFrame(data_rows)

    # **Jika tidak ada informasi kedalaman, hanya simpan Timestamp**
    else:
        print(f"⚠️ Tidak ditemukan informasi kedalaman di {input_file}, hanya menyimpan Timestamp")
        data_rows = []

        for line in lines:
            if re.match(r'^\s*\d{8}\s+\d{4}', line):
                parts = line.strip().split()
                date, time = parts[0], parts[1]
                data_rows.append({'YYYYMMDD': date, 'HHMM': time})
        
        df = pd.DataFrame(data_rows)

    # **Gabungkan tanggal & waktu menjadi Timestamp**
    df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')

    # Hapus kolom asli tanggal dan waktu
    df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)

    # Jika ada suhu, ubah nilai -9.999 jadi NaN
    if 'SST' in df.columns or any(col.startswith("TEMP_") for col in df.columns):
        df.replace('-9.999', 'NaN', inplace=True)
        for col in df.columns:
            if col != 'Timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")

    return output_file


if __name__ == "__main__":
    # Contoh penggunaan
    input_path = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/0N90E/ASCII/rad0n90e_dy.ascii"
    convert_ascii_to_csv(input_path)
