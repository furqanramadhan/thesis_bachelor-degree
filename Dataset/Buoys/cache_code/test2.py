import os
import re
import pandas as pd
import glob

def convert_ascii_to_csv(input_file, output_file=None):
    """
    Mengkonversi file ASCII dari data buoy RAMA menjadi format CSV.
    Menghapus kolom QUALITY dan SOURCE dari output.
    
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
    
    # Cari informasi kedalaman
    depth_line = None
    for line in lines:
        if 'Depth(M):' in line:
            depth_line = line.strip()
            break
    
    if not depth_line:
        print("❌ Tidak dapat menemukan informasi kedalaman")
        return None
    
    # Ekstrak nilai kedalaman untuk digunakan sebagai nama kolom
    depth_parts = depth_line.split(':')[1].strip().split()
    depth_values = []
    for part in depth_parts:
        try:
            depth = float(part)
            depth_values.append(f"TEMP_{depth}m")
        except ValueError:
            continue
    
    # Jika kedalaman pertama adalah 1, itu SST (Sea Surface Temperature)
    if depth_values and "TEMP_1.0m" in depth_values[0]:
        depth_values[0] = "SST"
    
    # Buat struktur untuk data
    data_rows = []
    
    # Proses baris data
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            # Ambil tanggal dan waktu
            date = parts[0]
            time = parts[1]
            
            quality_start_idx = -1
            for i, val in enumerate(parts[2:], 2):
                if re.match(r'^[1-5]+$', val) and len(val) > 8:
                    quality_start_idx = i
                    break
            
            # Jika tidak menemukan kolom QUALITY/SOURCE, gunakan semua nilai
            if quality_start_idx == -1:
                temp_values = parts[2:]
            else:
                temp_values = parts[2:quality_start_idx]
            
            # Pastikan jumlah nilai sesuai dengan jumlah kedalaman
            if len(temp_values) > len(depth_values):
                temp_values = temp_values[:len(depth_values)]
            elif len(temp_values) < len(depth_values):
                # Isi dengan NaN jika kurang
                temp_values.extend(['NaN'] * (len(depth_values) - len(temp_values)))
            
            # Gabungkan tanggal, waktu, dan nilai suhu
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            for i, depth_name in enumerate(depth_values):
                if i < len(temp_values):
                    row_data[depth_name] = temp_values[i]
                else:
                    row_data[depth_name] = 'NaN'
            
            data_rows.append(row_data)
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Debug: tampilkan kolom yang berhasil diproses
    print(f"Kolom yang berhasil diproses: {df.columns.tolist()}")
    print(f"Jumlah baris data: {len(df)}")
    
    # Gabungkan kolom tanggal dan waktu ke timestamp
    df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
    
    # Hapus kolom asli tanggal dan waktu
    df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Konversi nilai suhu ke numerik
    for col in df.columns:
        if col != 'Timestamp':
            df[col] = df[col].replace('-9.999', 'NaN')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    
    return output_file
if __name__ == "__main__":
    # Ubah path ini ke lokasi file Anda
    result = convert_ascii_to_csv("/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/0N90E/ASCII/w0n90e_dy.ascii")