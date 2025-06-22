import os
import re
import numpy as np
import pandas as pd
import glob

# KONFIGURASI QUALITY DAN SOURCE CODES
ACCEPTABLE_QUALITY_CODES = [1, 2, 3]  # 1 = Highest Quality, 2 = Default Quality, 3 = Data Adjusted
ACCEPTABLE_SOURCE_CODES = [1, 2, 3, 5, 6, 7, 8]  # Termasuk interpolated data
EXCLUDE_QUALITY_CODES = [0, 4, 5]  # Missing, Lower Quality, Sensor Failed
EXCLUDE_SOURCE_CODES = [0, 4]  # No Sensor, Inactive

def filter_by_quality_source(df, quality_col=None, source_col=None):
    """
    Filter DataFrame berdasarkan quality codes dan source codes yang dapat diterima.
    
    Parameters:
    df (pandas.DataFrame): DataFrame untuk difilter
    quality_col (str, optional): Nama kolom quality
    source_col (str, optional): Nama kolom source
    
    Returns:
    pandas.DataFrame: DataFrame yang sudah difilter
    """
    original_count = len(df)
    
    if quality_col and quality_col in df.columns:
        # Filter berdasarkan quality codes yang dapat diterima
        df = df[df[quality_col].isin(ACCEPTABLE_QUALITY_CODES)]
        print(f"📊 Setelah filter quality: {len(df)} baris (dikurangi {original_count - len(df)} baris)")
    
    if source_col and source_col in df.columns:
        # Filter berdasarkan source codes yang dapat diterima
        df = df[df[source_col].isin(ACCEPTABLE_SOURCE_CODES)]
        print(f"📊 Setelah filter source: {len(df)} baris")
    
    return df

def extract_quality_source_from_line(line_parts, expected_data_count):
    """
    Ekstrak quality dan source codes dari baris data.
    
    Parameters:
    line_parts (list): Parts dari baris yang sudah di-split
    expected_data_count (int): Jumlah kolom data yang diharapkan
    
    Returns:
    tuple: (data_values, quality_codes, source_codes)
    """
    data_values = []
    quality_codes = []
    source_codes = []
    
    # Cari posisi dimulainya quality codes (biasanya string panjang angka 1-5)
    quality_start_idx = -1
    for i, val in enumerate(line_parts[2:], 2):
        if re.match(r'^[1-5]+$', val) and len(val) >= expected_data_count:
            quality_start_idx = i
            break
    
    if quality_start_idx != -1:
        # Ada quality/source codes
        data_values = line_parts[2:quality_start_idx]
        
        # Ekstrak quality codes (karakter per karakter)
        quality_string = line_parts[quality_start_idx]
        quality_codes = [int(q) for q in quality_string[:expected_data_count]]
        
        # Ekstrak source codes jika ada
        if quality_start_idx + 1 < len(line_parts):
            source_string = line_parts[quality_start_idx + 1]
            source_codes = [int(s) for s in source_string[:expected_data_count]]
    else:
        # Tidak ada quality/source codes
        data_values = line_parts[2:]
    
    return data_values, quality_codes, source_codes

def convert_ascii_to_csv(input_file, output_file=None):
    """
    Mengkonversi file ASCII dari data buoy RAMA menjadi format CSV.
    Secara otomatis mendeteksi format file berdasarkan struktur data.
    
    Parameters:
    input_file (str): Path ke file ASCII
    output_file (str, optional): Path untuk menyimpan file CSV hasil.
    
    Returns:
    str: Path ke file CSV yang dihasilkan
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.csv'
    
    # Baca beberapa baris pertama untuk identifikasi format
    with open(input_file, 'r') as f:
        preview_lines = [f.readline() for _ in range(20)]
    
    # Deteksi tipe format file:
    # 1. Format Temperatur (multi-depth dengan pola Index:, multiple columns)
    # 2. Format Angin (multiple variables pada kedalaman yang sama)
    # 3. Format Umum (single variable dengan simple structure)
    
    is_temperature_format = False
    is_wind_format = False
    
    # Cek apakah ada baris Index:
    has_index_line = any('Index:' in line for line in preview_lines)
    
    # Cek pola depth pada file temperatur
    depth_pattern = False
    for line in preview_lines:
        if 'Depth(M):' in line and len(line.split()) > 6:
            # Jika ada banyak nilai kedalaman pada baris ini
            try:
                # Coba convert beberapa nilai ke float untuk konfirmasi multi-depth
                depth_parts = line.split(':')[1].strip().split()
                depth_count = sum(1 for part in depth_parts if re.match(r'^\d+\.?\d*$', part))
                if depth_count >= 3:  # Jika ada minimal 3 nilai depth
                    depth_pattern = True
                    break
            except (ValueError, IndexError):
                pass
    
    # Cek pola file angin (multiple variables dengan single depth)
    wind_pattern = False
    for i, line in enumerate(preview_lines):
        if 'Depth (M):' in line and 'WDIR' in ''.join(preview_lines[i:i+3]):
            wind_pattern = True
            break
    
    # Tentukan format berdasarkan pola yang terdeteksi
    if has_index_line and depth_pattern:
        is_temperature_format = True
    elif wind_pattern:
        is_wind_format = True
    
    # Debug output
    if is_temperature_format:
        print(f"📊 Terdeteksi format suhu (multi-kedalaman) dari {input_file}")
        return convert_temperature_ascii_to_csv(input_file, output_file)
    elif is_wind_format:
        print(f"🌬️ Terdeteksi format angin dari {input_file}")
        return convert_wind_ascii_to_csv(input_file, output_file)
    else:
        print(f"📋 Terdeteksi format umum dari {input_file}")
        return convert_general_ascii_to_csv(input_file, output_file)

def convert_temperature_ascii_to_csv(input_file, output_file):
    """
    Fungsi untuk konversi file format suhu multi-kedalaman dengan handling multiple time blocks
    dan skip TEMP_13m untuk konsistensi data.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # FASE 1: Deteksi semua time blocks dan kedalaman unik
    time_blocks = []
    all_depths = set()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Cari baris Time: yang menandakan time block baru
        if line.startswith('Time:') and 'index' in line.lower():
            # Parse informasi time block
            time_info = line
            
            # Cari baris Index: dan Depth(M): setelah Time:
            depth_line = None
            for j in range(i + 1, min(i + 5, len(lines))):  # Cek 4 baris ke depan
                if 'Depth(M):' in lines[j]:
                    depth_line = lines[j].strip()
                    break
            
            if depth_line:
                # Ekstrak kedalaman dari baris ini
                depth_parts = depth_line.split(':')[1].strip().split()
                block_depths = []
                
                for part in depth_parts:
                    if part in ['QUALITY', 'SOURCE']:
                        break
                    try:
                        depth = float(part)
                        block_depths.append(depth)
                        all_depths.add(depth)
                    except ValueError:
                        continue
                
                # Simpan informasi time block
                time_blocks.append({
                    'info': time_info,
                    'depths': block_depths,
                    'start_line': i
                })
                
                print(f"🔍 Time Block ditemukan: {len(block_depths)} kedalaman - {block_depths}")
        
        i += 1
    
    print(f"📊 Total time blocks ditemukan: {len(time_blocks)}")
    print(f"🌊 Semua kedalaman unik: {sorted(all_depths)}")
    
    # FASE 2: Buat struktur kolom target (skip kedalaman 13m)
    target_depths = sorted([d for d in all_depths if d != 13.0])  # Skip 13m
    
    # Buat nama kolom
    depth_columns = []
    for depth in target_depths:
        if depth == 1.0:
            depth_columns.append("SST")
        else:
            depth_columns.append(f"TEMP_{depth}m")
    
    print(f"🎯 Kolom target (skip 13m): {depth_columns}")
    
    # FASE 3: Buat mapping untuk setiap time block
    block_mappings = []
    for block in time_blocks:
        block_depths = block['depths']
        
        # Buat mapping dari index data ke kolom target
        mapping = {}
        for data_idx, depth in enumerate(block_depths):
            if depth == 13.0:
                # Skip 13m - tidak di-map ke kolom manapun
                continue
            
            # Cari posisi depth ini di target_depths
            try:
                target_idx = target_depths.index(depth)
                target_col = depth_columns[target_idx]
                mapping[data_idx] = target_col
            except ValueError:
                continue  # Depth tidak ada di target
        
        block_mappings.append({
            'depths': block_depths,
            'mapping': mapping,
            'info': block['info']
        })
        
        print(f"📋 Mapping untuk block {len(block_mappings)}: {mapping}")
    
    # FASE 4: Proses data dengan mapping yang tepat
    data_rows = []
    current_block_idx = 0
    current_mapping = block_mappings[0]['mapping'] if block_mappings else {}
    
    for line in lines:
        # Check apakah baris ini adalah awal time block baru
        if line.strip().startswith('Time:') and 'index' in line.lower():
            # Update mapping ke block berikutnya
            for i, block in enumerate(time_blocks):
                if block['info'].strip() == line.strip():
                    if i < len(block_mappings):
                        current_mapping = block_mappings[i]['mapping']
                        current_block_idx = i
                        print(f"🔄 Switch ke time block {i+1}")
                    break
            continue
        
        # Proses baris data
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            if len(parts) < 3:
                continue
            
            # Ambil tanggal dan waktu
            date = parts[0]
            time = parts[1]
            
            # Ekstrak data temperatur berdasarkan mapping current block
            temp_data = parts[2:]  # Semua data setelah tanggal/waktu
            
            # Inisialisasi row data dengan semua kolom target
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            for col in depth_columns:
                row_data[col] = 'NaN'  # Default value
            
            # Map data sesuai dengan mapping current block
            for data_idx, target_col in current_mapping.items():
                if data_idx < len(temp_data):
                    row_data[target_col] = temp_data[data_idx]
            
            data_rows.append(row_data)
    
    print(f"📈 Total baris data yang diproses: {len(data_rows)}")
    
    # FASE 5: Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Gabungkan kolom tanggal dan waktu ke timestamp
    df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
    df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Konversi missing values (-9.999) ke NaN untuk kolom data temperatur
    for col in depth_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if str(x) == '-9.999' else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Hapus baris dengan terlalu banyak missing values
    temp_cols = depth_columns
    max_allowed_missing = min(5, len(temp_cols) - 1)
    
    missing_counts = df[temp_cols].isnull().sum(axis=1)
    rows_to_drop = missing_counts > max_allowed_missing
    
    if rows_to_drop.sum() > 0:
        print(f"🗑️ Menghapus {rows_to_drop.sum()} baris dengan lebih dari {max_allowed_missing} missing values")
        df = df[~rows_to_drop].reset_index(drop=True)
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    
    # Tampilkan informasi missing values per kolom
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("\n📊 Informasi Missing Values:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                print(f"   {col}: {missing_count} missing values ({percentage:.1f}%)")
    
    # Tampilkan statistik per time block jika ada
    if len(time_blocks) > 1:
        print(f"\n📋 Statistik Time Blocks:")
        for i, block in enumerate(time_blocks):
            depths_str = ', '.join([f"{d}m" if d != 1.0 else "SST" for d in block['depths']])
            skip_info = " (13m skipped)" if 13.0 in block['depths'] else ""
            print(f"   Block {i+1}: {len(block['depths'])} depths [{depths_str}]{skip_info}")
    
    return output_file

def convert_wind_ascii_to_csv(input_file, output_file):
    """Fungsi khusus untuk konversi file format angin dengan improvement missing values handling"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Cari header data
    header_line = None
    depth_line = None
    
    for i, line in enumerate(lines):
        if 'Depth (M):' in line:
            depth_line = line
            # Header biasanya berada pada baris setelah Depth
            if i + 1 < len(lines):
                header_line = lines[i + 1]
            break
    
    if not header_line or not depth_line:
        print("❌ Tidak dapat menemukan header atau kedalaman untuk file angin")
        return None
    
    # Ekstrak header
    headers = header_line.strip().split()
    
    # Identifikasi header yang valid (YYYYMMDD, HHMM, UWND, VWND, WSPD, WDIR)
    valid_headers = []
    data_headers = []  # Header untuk data angin (bukan tanggal/waktu)
    
    for header in headers:
        if header in ['YYYYMMDD', 'HHMM', 'UWND', 'VWND', 'WSPD', 'WDIR']:
            valid_headers.append(header)
            if header in ['UWND', 'VWND', 'WSPD', 'WDIR']:
                data_headers.append(header)
    
    # Baca data
    data_rows = []
    removed_rows_count = 0  # Counter untuk baris yang dihapus karena terlalu banyak missing values
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            # Ekstrak data, quality, dan source menggunakan function yang sudah ada
            wind_values, quality_codes, source_codes = extract_quality_source_from_line(parts, len(data_headers))
            
            # 🔧 IMPROVEMENT 1: Hitung jumlah missing values (-99.9) dalam baris ini
            missing_count = 0
            for value in wind_values:
                if str(value) in ['-99.9', '-999.9', '-9.999', '-9']:
                    missing_count += 1
            
            # 🔧 IMPROVEMENT 2: Skip baris jika lebih dari 3 missing values
            if missing_count > 3:
                removed_rows_count += 1
                continue  # Skip baris ini, jangan tambahkan ke data_rows
            
            # Ambil tanggal dan waktu
            row_data = {}
            if len(parts) >= 2:
                row_data['YYYYMMDD'] = parts[0]
                row_data['HHMM'] = parts[1]
                
                # Tambahkan nilai angin
                for i, header in enumerate(data_headers):
                    if i < len(wind_values):
                        row_data[header] = wind_values[i]
                        # Tambahkan quality dan source jika ada
                        if i < len(quality_codes):
                            row_data[f"{header}_QUALITY"] = quality_codes[i]
                        if i < len(source_codes):
                            row_data[f"{header}_SOURCE"] = source_codes[i]
                    else:
                        row_data[header] = 'NaN'
                
                if len(row_data) >= 2:  # Minimal ada tanggal dan waktu
                    data_rows.append(row_data)
    
    # Report jumlah baris yang dihapus karena terlalu banyak missing values
    if removed_rows_count > 0:
        print(f"🗑️ Menghapus {removed_rows_count} baris karena memiliki lebih dari 3 missing values (-99.9)")
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Debug: tampilkan kolom yang berhasil diproses
    print(f"Kolom yang berhasil diproses: {df.columns.tolist()}")
    print(f"Jumlah baris data setelah filter missing values: {len(df)}")
    
    # Gabungkan kolom tanggal dan waktu ke timestamp
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
        df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # 🔧 IMPROVEMENT 3: Konversi individual missing values ke NaN (tetap dilakukan untuk sisa data)
    for col in df.columns:
        if col != 'Timestamp' and '_QUALITY' not in col and '_SOURCE' not in col:
            # Ganti hanya nilai missing yang spesifik (-99.9, -999.9, dll)
            df[col] = df[col].apply(lambda x: np.nan if str(x) in ['-99.9', '-999.9', '-9.999', '-9'] else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter berdasarkan quality dan source codes (tetap dipertahankan untuk validasi tambahan)
    quality_cols = [col for col in df.columns if '_QUALITY' in col]
    source_cols = [col for col in df.columns if '_SOURCE' in col]

    if quality_cols or source_cols:
        print("🔍 Menerapkan filter quality dan source codes pada file angin...")
        # Aplikasikan filter untuk setiap variabel angin
        for header in data_headers:
            quality_col = f"{header}_QUALITY" if f"{header}_QUALITY" in df.columns else None
            source_col = f"{header}_SOURCE" if f"{header}_SOURCE" in df.columns else None
            df = filter_by_quality_source(df, quality_col, source_col)
        # Setelah filtering, hapus baris yang sekarang memiliki semua data NaN
        wind_data_cols = [col for col in df.columns if col != 'Timestamp' and '_QUALITY' not in col and '_SOURCE' not in col]
        
        # Identifikasi baris dengan semua wind data NaN
        all_wind_nan_mask = df[wind_data_cols].isnull().all(axis=1)
        rows_to_drop_after_qc = all_wind_nan_mask.sum()
        
        if rows_to_drop_after_qc > 0:
            df = df[~all_wind_nan_mask].reset_index(drop=True)
            print(f"🗑️ Menghapus {rows_to_drop_after_qc} baris tambahan setelah quality/source filtering (semua data menjadi NaN)")
    
    # Hapus kolom quality dan source sebelum menyimpan
    df_output = df.drop(columns=quality_cols + source_cols, errors='ignore')
    
    # 🔧 IMPROVEMENT 5: Final check - pastikan tidak ada baris dengan semua wind data kosong
    wind_data_cols_final = [col for col in df_output.columns if col != 'Timestamp']
    final_empty_rows = df_output[wind_data_cols_final].isnull().all(axis=1).sum()
    
    if final_empty_rows > 0:
        df_output = df_output[~df_output[wind_data_cols_final].isnull().all(axis=1)].reset_index(drop=True)
        print(f"🗑️ Final cleanup: menghapus {final_empty_rows} baris dengan semua data kosong")
    
    # Simpan ke CSV
    df_output.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df_output)} baris data ke {output_file}")
    
    # Tampilkan statistik missing values per kolom
    missing_info = df_output.isnull().sum()
    if missing_info.sum() > 0:
        print("\n📊 Informasi Missing Values per kolom:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                percentage = (missing_count / len(df_output)) * 100
                print(f"   {col}: {missing_count} missing values ({percentage:.1f}%)")
    else:
        print("\n✅ Tidak ada missing values dalam dataset final")
    
    # Tampilkan ringkasan total removal
    total_original_data_rows = len(data_rows) + removed_rows_count
    if hasattr(locals(), 'rows_to_drop_after_qc'):
        total_removed = removed_rows_count + rows_to_drop_after_qc
    else:
        total_removed = removed_rows_count
        
    if hasattr(locals(), 'final_empty_rows'):
        total_removed += final_empty_rows
    
    print(f"   Data asli: ~{total_original_data_rows} baris")
    print(f"   Data tersimpan: {len(df_output)} baris") 
    print(f"   Total dihapus: ~{total_removed} baris")
    return output_file

def convert_general_ascii_to_csv(input_file, output_file):
    """Fungsi untuk konversi file format umum (rad, rain, rh, sst) - IMPROVED VERSION dengan SST Support"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Identifikasi header kolom dan data
    data_rows = []
    header_line = None
    
    for i, line in enumerate(lines):
        if 'YYYYMMDD' in line and 'HHMM' in line:
            header_line = line
            break
    
    if not header_line:
        print("❌ Tidak dapat menemukan header untuk file")
        return None
    
    # Ekstrak header
    headers = header_line.strip().split()
    
    # 🔧 PERBAIKAN 1: Deteksi format berdasarkan nama file atau header pattern
    filename = os.path.basename(input_file).lower()
    is_sst_format = False
    is_non_sst_format = False  # rad, rain, rh
    
    # Deteksi berdasarkan nama file
    if 'sst' in filename:
        is_sst_format = True
        print("🌡️ Format SST terdeteksi berdasarkan nama file - akan menggunakan Q dan S sebagai quality/source codes")
    elif any(var in filename for var in ['rad', 'rain', 'rh']):
        is_non_sst_format = True
        print(f"🌤️ Format non-SST terdeteksi ({filename}) - akan menghilangkan kolom Q dan S")
    else:
        # Fallback ke deteksi berdasarkan header
        if 'SST' in headers and ('Q' in headers and 'S' in headers):
            is_sst_format = True
            print("🌡️ Format SST terdeteksi - akan menggunakan Q dan S sebagai quality/source codes")
        elif ('Q' in headers and 'S' in headers):
            is_non_sst_format = True
            print("🌤️ Format non-SST terdeteksi - akan menghilangkan kolom Q dan S")
    
    # 🔧 PERBAIKAN 2: Identifikasi kolom berdasarkan format yang terdeteksi
    valid_headers = []
    data_headers = []
    quality_headers = []
    source_headers = []
    
    if is_sst_format:
        # Untuk format SST, tetap gunakan Q dan S
        for header in headers:
            if header not in ['']:
                valid_headers.append(header)
                if header in ['YYYYMMDD', 'HHMM']:
                    continue  # Skip date/time columns
                elif header in ['SST']:
                    data_headers.append(header)  # Data aktual
                elif header in ['Q', 'QUALITY']:
                    quality_headers.append(header)  # Quality codes
                elif header in ['S', 'SOURCE']:
                    source_headers.append(header)  # Source codes
    elif is_non_sst_format:
        # 🆕 UNTUK RAD, RAIN, RH: Abaikan kolom Q dan S
        for header in headers:
            if header not in ['Q', 'S', '']:  # ✨ KUNCI: Skip kolom Q dan S
                valid_headers.append(header)
                if header not in ['YYYYMMDD', 'HHMM']:
                    data_headers.append(header)
        
        print(f"✂️ Kolom Q dan S dihilangkan untuk variabel non-SST")
        print(f"📊 Data headers yang akan diproses: {data_headers}")
    else:
        # Untuk format lainnya (original logic)
        for header in headers:
            if header not in ['QUALITY', 'SOURCE'] and header != '':
                valid_headers.append(header)
                if header not in ['YYYYMMDD', 'HHMM']:
                    data_headers.append(header)
    
    print(f"📊 Data headers: {data_headers}")
    print(f"🔍 Quality headers: {quality_headers}")
    print(f"📡 Source headers: {source_headers}")
    
    # Proses baris data dengan improvement untuk missing values handling
    removed_rows_count = 0  # Counter untuk baris yang dihapus
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            # 🔧 PERBAIKAN 3: Handle ekstraksi data berdasarkan format
            if is_sst_format:
                # Untuk format SST, quality dan source sudah terpisah di kolom individual
                data_values = []
                quality_codes = []
                source_codes = []
                
                # Mapping berdasarkan posisi header
                for i, header in enumerate(headers):
                    if i + 2 < len(parts):  # Skip YYYYMMDD (0) dan HHMM (1)
                        value = parts[i]
                        
                        if header in data_headers:
                            data_values.append(value)
                        elif header in quality_headers:
                            try:
                                quality_codes.append(int(value))
                            except ValueError:
                                quality_codes.append(0)  # Default untuk nilai non-numeric
                        elif header in source_headers:
                            try:
                                source_codes.append(int(value))
                            except ValueError:
                                source_codes.append(0)  # Default untuk nilai non-numeric
                
            elif is_non_sst_format:
                # 🆕 UNTUK RAD, RAIN, RH: Ambil hanya data tanpa Q dan S
                data_values = []
                quality_codes = []
                source_codes = []
                
                # Ambil data berdasarkan posisi header yang valid (tanpa Q dan S)
                data_start_index = 2  # Skip YYYYMMDD dan HHMM
                
                for i, header in enumerate(headers):
                    if header in data_headers and (data_start_index + i) < len(parts):
                        data_values.append(parts[data_start_index + i])
                
                # ✨ TIDAK ada quality/source codes untuk rad, rain, rh
                print(f"📋 Data extracted untuk non-SST: {data_values} (Q dan S diabaikan)")
                
            else:
                # Untuk format lainnya, gunakan logic original
                if len(data_headers) > 0:
                    data_values, quality_codes, source_codes = extract_quality_source_from_line(parts, len(data_headers))
                else:
                    data_values = parts[2:] if len(parts) > 2 else []
                    quality_codes = []
                    source_codes = []
            
            # 🔧 PERBAIKAN 4: Pre-filtering berdasarkan quality/source codes HANYA untuk SST
            skip_record_due_to_quality_source = False
            
            if is_sst_format:  # ✨ HANYA terapkan filtering untuk SST
                # Check individual quality/source codes
                for i in range(len(data_values)):
                    current_quality = quality_codes[i] if i < len(quality_codes) else None
                    current_source = source_codes[i] if i < len(source_codes) else None
                    
                    # Skip jika quality code tidak acceptable
                    if current_quality is not None and current_quality in EXCLUDE_QUALITY_CODES:
                        skip_record_due_to_quality_source = True
                        print(f"🚫 SST Record ditolak karena quality code {current_quality}")
                        break
                    
                    # Skip jika source code tidak acceptable  
                    if current_source is not None and current_source in EXCLUDE_SOURCE_CODES:
                        skip_record_due_to_quality_source = True
                        print(f"🚫 SST Record ditolak karena source code {current_source}")
                        break
                    
                    # Skip jika kombinasi (0,0)
                    if current_quality == 0 and current_source == 0:
                        skip_record_due_to_quality_source = True
                        print(f"🚫 SST Record ditolak karena kombinasi quality=0 dan source=0")
                        break
            
            if skip_record_due_to_quality_source:
                removed_rows_count += 1
                continue  # Skip baris ini karena quality/source tidak acceptable
            
            # 🔧 IMPROVEMENT: Hitung jumlah missing values dalam baris ini
            missing_count = 0
            for value in data_values:
                if str(value) in ['-999.99', '-9.99', '-9.999', '-999.9', '-9.9']:
                    missing_count += 1
            
            # Skip baris jika lebih dari 2 missing values
            if missing_count > 2:
                removed_rows_count += 1
                continue  # Skip baris ini, jangan tambahkan ke data_rows
            
            # Pastikan panjang data sesuai dengan header
            if len(parts) >= 2:  # Minimal ada tanggal dan waktu
                row_data = {}
                row_data['YYYYMMDD'] = parts[0]
                row_data['HHMM'] = parts[1]
                
                # Tambahkan nilai data
                for i, header in enumerate(data_headers):
                    if i < len(data_values):
                        row_data[header] = data_values[i]
                        # ✨ Tambahkan quality dan source HANYA untuk SST
                        if is_sst_format:
                            if i < len(quality_codes):
                                row_data[f"{header}_QUALITY"] = quality_codes[i]
                            if i < len(source_codes):
                                row_data[f"{header}_SOURCE"] = source_codes[i]
                    else:
                        row_data[header] = 'NaN'
                
                if len(row_data) >= 2:  # Minimal ada tanggal dan waktu
                    data_rows.append(row_data)
    
    # Report jumlah baris yang dihapus
    if removed_rows_count > 0:
        print(f"🗑️ Menghapus {removed_rows_count} baris karena:")
        if is_sst_format:
            print(f"   - Quality/Source codes tidak acceptable (SST)")
        print(f"   - Lebih dari 2 missing values")
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Debug: tampilkan kolom yang berhasil diproses
    print(f"Kolom yang berhasil diproses: {df.columns.tolist()}")
    print(f"Jumlah baris data setelah filter: {len(df)}")
    
    # Gabungkan kolom tanggal dan waktu ke timestamp
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
        df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # 🔧 IMPROVEMENT: Konversi nilai ke numerik dan tangani missing values dengan pattern yang diperbaiki
    for col in df.columns:
        if col != 'Timestamp' and '_QUALITY' not in col and '_SOURCE' not in col:
            # Identifikasi dan ganti nilai missing dengan pattern yang diperbaiki
            df[col] = df[col].apply(lambda x: np.nan if str(x) in ['-999.99', '-9.99', '-9.999', '-999.9', '-9.9'] else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🔧 IMPROVEMENT: Additional filtering menggunakan DataFrame-level quality/source filter HANYA untuk SST
    quality_cols = [col for col in df.columns if '_QUALITY' in col]
    source_cols = [col for col in df.columns if '_SOURCE' in col]

    if (quality_cols or source_cols) and is_sst_format:  # ✨ HANYA untuk SST
        print("🔍 Menerapkan additional DataFrame-level quality dan source filtering untuk SST...")
        original_len = len(df)
        
        for header in data_headers:
            quality_col = f"{header}_QUALITY" if f"{header}_QUALITY" in df.columns else None
            source_col = f"{header}_SOURCE" if f"{header}_SOURCE" in df.columns else None
            df = filter_by_quality_source(df, quality_col, source_col)
        
        additional_removed = original_len - len(df)
        if additional_removed > 0:
            print(f"🗑️ DataFrame-level filtering menghapus {additional_removed} baris tambahan")
    
    # Final cleanup - hapus baris yang sekarang memiliki semua data NaN
    data_cols_final = [col for col in df.columns if col != 'Timestamp' and '_QUALITY' not in col and '_SOURCE' not in col]
    
    if data_cols_final:
        # Identifikasi baris dengan semua data NaN
        all_data_nan_mask = df[data_cols_final].isnull().all(axis=1)
        rows_to_drop_final = all_data_nan_mask.sum()
        
        if rows_to_drop_final > 0:
            df = df[~all_data_nan_mask].reset_index(drop=True)
            print(f"🗑️ Final cleanup: menghapus {rows_to_drop_final} baris dengan semua data menjadi NaN")
    
    # ✨ HAPUS kolom quality dan source untuk non-SST format
    if is_non_sst_format:
        # Untuk rad, rain, rh - hapus semua kolom quality/source (jika ada yang tersisa)
        df = df.drop(columns=quality_cols + source_cols, errors='ignore')
        print("✂️ Kolom Q dan S berhasil dihilangkan dari output final")
    elif is_sst_format:
        # Untuk SST, tetap simpan kolom quality/source jika diinginkan
        # Atau uncomment baris berikut jika ingin menghapus juga untuk SST:
        # df = df.drop(columns=quality_cols + source_cols, errors='ignore')
        pass
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    
    # Tampilkan statistik missing values per kolom
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print("\n📊 Informasi Missing Values per kolom:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                percentage = (missing_count / len(df)) * 100
                print(f"   {col}: {missing_count} missing values ({percentage:.1f}%)")
    else:
        print("\n✅ Tidak ada missing values dalam dataset final")
    
    # Tampilkan ringkasan total removal
    total_original_estimated = len(data_rows) + removed_rows_count
    total_removed = removed_rows_count
    if 'additional_removed' in locals():
        total_removed += additional_removed
    if 'rows_to_drop_final' in locals():
        total_removed += rows_to_drop_final
    
    print(f"\n📊 Ringkasan Data Processing:")
    print(f"   Data asli (estimasi): ~{total_original_estimated} baris")
    print(f"   Data tersimpan: {len(df)} baris") 
    print(f"   Total dihapus: ~{total_removed} baris")
    
    # ✨ Status kolom Q dan S
    if is_non_sst_format:
        print(f"   ✂️ Kolom Q dan S berhasil dihilangkan untuk variabel non-SST")
    elif is_sst_format:
        print(f"   🔍 Kolom Q dan S dipertahankan untuk variabel SST")
    
    return output_file

def process_date_columns(df):
    """
    Memproses kolom timestamp menjadi komponen tanggal terpisah
    dengan format bulan numerik (1-12)
    
    Parameters:
    df (pandas.DataFrame): DataFrame untuk diproses
    
    Returns:
    pandas.DataFrame: DataFrame dengan kolom tanggal yang diperbarui
    """
    if 'Timestamp' in df.columns:
        # Konversi timestamp ke datetime jika belum
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Buat kolom Date dengan format YYYY-MM-DD
        df['Date'] = df['Timestamp'].dt.date
        
        # Tambah kolom tahun, bulan (angka), dan hari
        df['Year'] = df['Timestamp'].dt.year
        
        # Ubah bulan menjadi numerik (1-12) bukan nama bulan
        df['Month'] = df['Timestamp'].dt.month
        
        # Tambah kolom Day
        df['Day'] = df['Timestamp'].dt.day
        
        # Hapus kolom Timestamp original
        df.drop('Timestamp', axis=1, inplace=True)
    
        # Atur ulang urutan kolom (taruh quality/source di akhir)
        date_cols = ['Date', 'Year', 'Month', 'Day']
        data_cols = [col for col in df.columns if col not in date_cols and '_QUALITY' not in col and '_SOURCE' not in col]
        quality_source_cols = [col for col in df.columns if '_QUALITY' in col or '_SOURCE' in col]
        
        df = df[date_cols + data_cols + quality_source_cols]
    return df

def save_to_excel(df, output_file, excel_directory=None):
    """
    Menyimpan DataFrame ke format Excel dengan penyesuaian lebar kolom otomatis.
    
    Parameters:
    df (pandas.DataFrame): DataFrame yang akan disimpan
    output_file (str): Path untuk menyimpan file CSV (digunakan sebagai referensi nama)
    excel_directory (str, optional): Direktori untuk menyimpan file Excel
    
    Returns:
    str: Path ke file Excel yang dihasilkan
    """
    # Dapatkan nama file saja tanpa path
    filename = os.path.basename(output_file)
    basename = os.path.splitext(filename)[0]
    
    # Tentukan lokasi output Excel
    if excel_directory:
        # Pastikan direktori Excel ada
        os.makedirs(excel_directory, exist_ok=True)
        excel_file = os.path.join(excel_directory, basename + '.xlsx')
    else:
        # Jika tidak ada direktori Excel, simpan di lokasi yang sama dengan CSV
        excel_file = output_file.replace('.csv', '.xlsx')
    
    # Buat Excel writer dengan xlsxwriter engine
    try:
        writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
        
        # Tulis DataFrame ke Excel
        df.to_excel(writer, index=False, sheet_name='Data')
        
        # Dapatkan workbook dan worksheet
        workbook = writer.book
        worksheet = writer.sheets['Data']
        
        # Format untuk tanggal
        date_format = workbook.add_format({'num_format': 'yyyy-mm-dd'})
        
        # Sesuaikan lebar kolom
        for idx, col in enumerate(df.columns):
            # Hitung lebar maksimum
            max_length = max(
                df[col].astype(str).apply(len).max(),  # Panjang data
                len(str(col))  # Panjang header
            )
            
            # Tambahkan sedikit padding
            adjusted_width = max_length + 2
            
            # Set lebar kolom
            worksheet.set_column(idx, idx, adjusted_width)
            
            # Terapkan format date untuk kolom Date
            if col == 'Date':
                worksheet.set_column(idx, idx, adjusted_width, date_format)
        
        # Simpan file
        writer.close()
        print(f"✅ Berhasil menyimpan ke Excel: {excel_file}")
        return excel_file
        
    except Exception as e:
        print(f"❌ Error saat menyimpan Excel: {str(e)}")
        if 'writer' in locals():
            writer.close()
        return None

def convert_ascii_to_excel(input_file, output_file=None, excel_directory=None):
    """
    Mengkonversi file ASCII ke format Excel (.xlsx)
    
    Parameters:
    input_file (str): Path ke file ASCII
    output_file (str, optional): Path untuk menyimpan file CSV
    excel_directory (str, optional): Direktori untuk menyimpan file Excel
    
    Returns:
    str: Path ke file Excel yang dihasilkan
    """
    # Gunakan fungsi convert_ascii_to_csv yang sudah ada untuk mendapatkan DataFrame
    csv_file = convert_ascii_to_csv(input_file, output_file)
    
    if csv_file:
        # Baca CSV yang baru dibuat
        df = pd.read_csv(csv_file)
        
        # Proses kolom tanggal
        df = process_date_columns(df)
        
        # Simpan kembali ke CSV dengan format baru
        df.to_csv(csv_file, index=False)
        print(f"✅ Berhasil memperbarui CSV dengan format tanggal baru: {csv_file}")
        
        # Simpan ke Excel di direktori terpisah jika diminta
        excel_file = save_to_excel(df, csv_file, excel_directory)
        return excel_file
    
    return None

def process_multiple_files_with_excel(input_directory, csv_directory=None, excel_directory=None, file_pattern='*.ascii'):
    """
    Memproses banyak file ASCII dalam satu direktori dan menghasilkan file CSV dan Excel
    di direktori terpisah.
    
    Parameters:
    input_directory (str): Path ke direktori yang berisi file ASCII
    csv_directory (str, optional): Path direktori untuk menyimpan file CSV hasil
    excel_directory (str, optional): Path direktori untuk menyimpan file Excel hasil
    file_pattern (str, optional): Pola file yang akan diproses (default: *.ascii)
    
    Returns:
    tuple: (list of CSV files, list of Excel files)
    """
    if not os.path.exists(input_directory):
        print(f"❌ Direktori input tidak ditemukan: {input_directory}")
        return [], []
    
    # Set default directories if not provided
    if csv_directory is None:
        csv_directory = os.path.join(input_directory, '../CSV')
    if excel_directory is None:
        excel_directory = os.path.join(input_directory, '../EXCEL')
    
    # Pastikan direktori CSV dan Excel ada
    os.makedirs(csv_directory, exist_ok=True)
    os.makedirs(excel_directory, exist_ok=True)
    
    print(f"📁 Menggunakan direktori CSV: {csv_directory}")
    print(f"📊 Menggunakan direktori Excel: {excel_directory}")
    
    total_files = len(glob.glob(os.path.join(input_directory, file_pattern)))
    print(f"🔍 Menemukan {total_files} file dengan pola '{file_pattern}' di direktori input")
    
    if total_files == 0:
        print("⚠️ Tidak ada file yang ditemukan untuk diproses.")
        return [], []
    
    processed_csv_files = []
    processed_excel_files = []
    
    # Cari semua file yang sesuai pola
    for i, input_file in enumerate(glob.glob(os.path.join(input_directory, file_pattern)), 1):
        filename = os.path.basename(input_file)
        base_name = os.path.splitext(filename)[0]
        csv_output = os.path.join(csv_directory, base_name + '.csv')
        
        print(f"\n🔄 Memproses file {i}/{total_files}: {filename}...")
        try:
            # Konversi ke CSV dan Excel dengan format tanggal baru
            excel_file = convert_ascii_to_excel(input_file, csv_output, excel_directory)
            if excel_file:
                processed_excel_files.append(excel_file)
                processed_csv_files.append(csv_output)
                print(f"✅ Berhasil mengkonversi {filename}")
            else:
                print(f"⚠️ Gagal mengkonversi {filename}")
                
        except Exception as e:
            print(f"❌ Error saat memproses {filename}: {str(e)}")
    
    # Ringkasan hasil proses
    print(f"\n✅ Selesai memproses {len(processed_csv_files)}/{total_files} file")
    print(f"📄 File CSV yang dihasilkan: {len(processed_csv_files)} (disimpan di {csv_directory})")
    print(f"📊 File Excel yang dihasilkan: {len(processed_excel_files)} (disimpan di {excel_directory})")
    
    return processed_csv_files, processed_excel_files

if __name__ == "__main__":
    locations = ["0N90E", "4N90E", "8N90E"]
    print("\n📋 KONVERSI DATA BUOY RAMA ASCII KE CSV DAN EXCEL 📋")
    print("=" * 60)

    # Statistik keseluruhan
    total_files_processed = 0
    total_files_found = 0
    all_csv_files = []
    all_excel_files = []

    for location in locations:
        print(f"\n🚀 Memproses lokasi: {location}")
        print("-" * 50)

        base_dir = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
        input_directory = f'{base_dir}/{location}/ASCII'
        csv_directory = f'{base_dir}/{location}/CSV'
        excel_directory = f'{base_dir}/{location}/EXCEL'

        print(f"📂 Direktori input: {input_directory}")
        print(f"📄 Direktori output CSV: {csv_directory}")
        print(f"📊 Direktori output Excel: {excel_directory}")

        # Hitung total file yang ditemukan untuk lokasi ini
        files_in_location = len(glob.glob(os.path.join(input_directory, '*.ascii')))
        total_files_found += files_in_location
        print(f"🔍 Ditemukan {files_in_location} file ASCII di {location}")

        # Proses semua file ASCII dalam direktori
        csv_files, excel_files = process_multiple_files_with_excel(
            input_directory=input_directory,
            csv_directory=csv_directory,
            excel_directory=excel_directory,
            file_pattern='*.ascii'
        )
        
        # Update statistik keseluruhan
        total_files_processed += len(csv_files)
        all_csv_files.extend(csv_files)
        all_excel_files.extend(excel_files)

        # Tampilkan hasil untuk lokasi ini
        if len(csv_files) > 0:
            print(f"\n📋 Hasil Konversi untuk {location}:")
            print(f"✅ Berhasil mengkonversi {len(csv_files)} dari {files_in_location} file ASCII")
            
            print(f"\n📄 File CSV yang dihasilkan di {location}:")
            for i, file in enumerate(csv_files, 1):
                print(f"   {i}. {os.path.basename(file)}")
            
            print(f"\n📊 File Excel yang dihasilkan di {location}:")
            for i, file in enumerate(excel_files, 1):
                print(f"   {i}. {os.path.basename(file)}")
        else:
            print(f"⚠️ Tidak ada file yang berhasil dikonversi di {location}")

    # Ringkasan akhir untuk semua lokasi
    print("\n" + "=" * 60)
    print("📊 RINGKASAN KESELURUHAN PROSES KONVERSI")
    print("=" * 60)
    print(f"🌐 Total lokasi diproses: {len(locations)}")
    print(f"📁 Total file ASCII ditemukan: {total_files_found}")
    print(f"✅ Total file berhasil dikonversi: {total_files_processed}")
    print(f"📄 Total file CSV dihasilkan: {len(all_csv_files)}")
    print(f"📊 Total file Excel dihasilkan: {len(all_excel_files)}")
    
    # Tampilkan daftar lengkap semua file yang dihasilkan
    if len(all_csv_files) > 0:
        print(f"\n📋 DAFTAR LENGKAP SEMUA FILE CSV YANG DIHASILKAN ({len(all_csv_files)} file):")
        print("-" * 60)
        
        # Kelompokkan berdasarkan lokasi untuk tampilan yang lebih rapi
        for location in locations:
            location_csv_files = [f for f in all_csv_files if f'/{location}/CSV/' in f]
            
            if location_csv_files:
                print(f"\n🌊 {location} ({len(location_csv_files)} file):")
                for i, file in enumerate(location_csv_files, 1):
                    filename = os.path.basename(file)
                    print(f"   {i}. {filename}")
    
    # Status akhir
    success_rate = (total_files_processed / total_files_found * 100) if total_files_found > 0 else 0
    
    print(f"\n🎯 Tingkat keberhasilan: {success_rate:.1f}%")
    
    if total_files_processed == total_files_found:
        print("🎉 Semua file berhasil dikonversi!")
    elif total_files_processed > 0:
        print(f"⚠️ {total_files_found - total_files_processed} file gagal dikonversi")
    else:
        print("❌ Tidak ada file yang berhasil dikonversi")
    
    print("\n✨ Proses konversi selesai! ✨")