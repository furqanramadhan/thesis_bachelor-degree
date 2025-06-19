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

# KONFIGURASI LOKASI DAN VARIABEL
LOCATIONS = ['0N90E', '4N90E', '8N90E']
VARIABLES = {
    'sst': 'Sea Surface Temperature (SST)',
    't': 'Water Temperature (T(z))',
    'rain': 'Rainfall (Rain)', 
    'rh': 'Relative Humidity (RH)',
    'wind': 'Wind Speed (Wspd)',
    'rad': 'Short Wave Radiation (SW Rad)'
}

def format_timestamp_columns(df):
    """
    Mengubah kolom Timestamp menjadi Date, Year, Month, Day dan menempatkan di awal
    
    Parameters:
    df (DataFrame): DataFrame dengan kolom Timestamp
    
    Returns:
    DataFrame: DataFrame dengan kolom Date, Year, Month, Day di awal
    """
    if 'Timestamp' not in df.columns:
        return df
    
    # Buat kolom Date, Year, Month, Day dari Timestamp
    df['Date'] = df['Timestamp'].dt.date
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    
    # Hapus kolom Timestamp
    df = df.drop('Timestamp', axis=1)
    
    # Urutkan kolom: Date, Year, Month, Day di awal, lalu variabel lainnya
    date_columns = ['Date', 'Year', 'Month', 'Day']
    other_columns = [col for col in df.columns if col not in date_columns]
    
    # Reorder kolom
    df = df[date_columns + other_columns]
    
    return df

def extract_quality_source_codes(data_line):
    """
    Ekstrak quality dan source codes dari baris data
    
    Parameters:
    data_line (str): Baris data ASCII
    
    Returns:
    tuple: (data_values, quality_codes, source_codes)
    """
    parts = data_line.strip().split()
    
    # Temukan di mana quality codes dimulai
    quality_start_idx = -1
    source_start_idx = -1
    
    # Cari pola untuk quality codes (biasanya digit 0-5 atau C)
    for i, val in enumerate(parts[2:], 2):  # Mulai setelah YYYYMMDD HHMM
        # Quality codes biasanya berupa string panjang dengan digit 0-5
        if re.match(r'^[0-5C]+$', val) and len(val) > 3:
            quality_start_idx = i
            break
    
    if quality_start_idx != -1:
        # Source codes biasanya setelah quality codes
        if quality_start_idx + 1 < len(parts):
            potential_source = parts[quality_start_idx + 1]
            if re.match(r'^[0-8]+$', potential_source) and len(potential_source) > 3:
                source_start_idx = quality_start_idx + 1
    
    # Ekstrak data, quality, dan source
    if quality_start_idx != -1:
        data_values = parts[2:quality_start_idx]
        quality_codes = parts[quality_start_idx] if quality_start_idx < len(parts) else ""
        source_codes = parts[source_start_idx] if source_start_idx != -1 and source_start_idx < len(parts) else ""
    else:
        # Jika tidak ada quality codes yang terdeteksi, ambil semua sebagai data
        data_values = parts[2:]
        quality_codes = ""
        source_codes = ""
    
    return data_values, quality_codes, source_codes

def validate_data_quality(quality_codes, source_codes, num_data_points):
    """
    Validasi kualitas data berdasarkan quality dan source codes
    
    Parameters:
    quality_codes (str): String quality codes
    source_codes (str): String source codes  
    num_data_points (int): Jumlah data points yang diharapkan
    
    Returns:
    list: Boolean mask untuk data yang valid
    """
    valid_mask = [True] * num_data_points
    
    # Konversi quality codes ke list integers
    if quality_codes:
        try:
            quality_list = [int(c) if c.isdigit() else -9 for c in quality_codes[:num_data_points]]
        except:
            quality_list = [2] * num_data_points  # Default quality jika error
    else:
        quality_list = [2] * num_data_points  # Default quality
    
    # Konversi source codes ke list integers
    if source_codes:
        try:
            source_list = [int(c) if c.isdigit() else 2 for c in source_codes[:num_data_points]]
        except:
            source_list = [2] * num_data_points  # Default source jika error
    else:
        source_list = [2] * num_data_points  # Default source
    
    # Pastikan panjang sama dengan jumlah data points
    while len(quality_list) < num_data_points:
        quality_list.append(2)
    while len(source_list) < num_data_points:
        source_list.append(2)
    
    # Validasi setiap data point
    for i in range(num_data_points):
        quality_valid = quality_list[i] in ACCEPTABLE_QUALITY_CODES
        source_valid = source_list[i] in ACCEPTABLE_SOURCE_CODES
        
        valid_mask[i] = quality_valid and source_valid
    
    return valid_mask, quality_list, source_list

def apply_quality_filter(data_values, quality_codes, source_codes):
    """
    Terapkan filter kualitas pada data values
    
    Parameters:
    data_values (list): List nilai data
    quality_codes (str): String quality codes
    source_codes (str): String source codes
    
    Returns:
    tuple: (filtered_data_values, quality_info)
    """
    if not data_values:
        return data_values, {}
    
    num_data_points = len(data_values)
    valid_mask, quality_list, source_list = validate_data_quality(
        quality_codes, source_codes, num_data_points
    )
    
    # Filter data berdasarkan valid mask
    filtered_values = []
    for i, value in enumerate(data_values):
        if i < len(valid_mask) and valid_mask[i]:
            filtered_values.append(value)
        else:
            filtered_values.append('NaN')  # Ganti data tidak valid dengan NaN
    
    # Informasi kualitas untuk logging
    quality_info = {
        'total_points': num_data_points,
        'valid_points': sum(valid_mask),
        'filtered_points': num_data_points - sum(valid_mask),
        'quality_codes': quality_list[:num_data_points],
        'source_codes': source_list[:num_data_points]
    }
    
    return filtered_values, quality_info

def convert_temperature_ascii_to_csv_with_quality(input_file, output_file):
    """Fungsi untuk konversi file format suhu multi-kedalaman dengan quality filtering"""
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
    
    # Ekstrak nilai kedalaman
    depth_parts = depth_line.split(':')[1].strip().split()
    depth_values = []
    for part in depth_parts:
        try:
            depth = float(part)
            depth_values.append(f"TEMP_{depth}m")
        except ValueError:
            continue
    
    # Jika kedalaman pertama adalah 1, itu SST
    if depth_values and "TEMP_1.0m" in depth_values[0]:
        depth_values[0] = "SST"
    
    data_rows = []
    total_quality_info = {'total_filtered': 0, 'total_processed': 0}
    
    # Proses baris data
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            date = parts[0]
            time = parts[1]
            
            # Ekstrak data, quality, dan source codes
            data_values, quality_codes, source_codes = extract_quality_source_codes(line)
            
            # Terapkan filter kualitas
            filtered_values, quality_info = apply_quality_filter(
                data_values, quality_codes, source_codes
            )
            
            # Update statistik kualitas
            total_quality_info['total_filtered'] += quality_info['filtered_points']
            total_quality_info['total_processed'] += quality_info['total_points']
            
            # Pastikan jumlah nilai sesuai dengan jumlah kedalaman
            if len(filtered_values) > len(depth_values):
                filtered_values = filtered_values[:len(depth_values)]
            elif len(filtered_values) < len(depth_values):
                filtered_values.extend(['NaN'] * (len(depth_values) - len(filtered_values)))
            
            # Buat row data
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            for i, depth_name in enumerate(depth_values):
                if i < len(filtered_values):
                    row_data[depth_name] = filtered_values[i]
                else:
                    row_data[depth_name] = 'NaN'
            
            data_rows.append(row_data)
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Proses timestamp
    df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
    df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Konversi ke numerik
    for col in df.columns:
        if col != 'Timestamp':
            df[col] = df[col].replace('-9.999', 'NaN')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Format kolom timestamp menjadi Date, Year, Month, Day
    df = format_timestamp_columns(df)
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    
    # Tampilkan statistik quality filtering
    filter_percentage = (total_quality_info['total_filtered'] / max(total_quality_info['total_processed'], 1)) * 100
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    print(f"📊 Quality Filter: {total_quality_info['total_filtered']}/{total_quality_info['total_processed']} data points difilter ({filter_percentage:.1f}%)")
    
    return output_file

def convert_general_ascii_to_csv_with_quality(input_file, output_file):
    """Fungsi untuk konversi file format umum dengan quality filtering"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Identifikasi header
    header_line = None
    for i, line in enumerate(lines):
        if 'YYYYMMDD' in line and 'HHMM' in line:
            header_line = line
            break
    
    if not header_line:
        print("❌ Tidak dapat menemukan header untuk file")
        return None
    
    headers = header_line.strip().split()
    valid_headers = [h for h in headers if h not in ['QUALITY', 'SOURCE'] and h != '']
    
    data_rows = []
    total_quality_info = {'total_filtered': 0, 'total_processed': 0}
    
    # Proses baris data
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            date = parts[0] if len(parts) > 0 else ''
            time = parts[1] if len(parts) > 1 else ''
            
            # Ekstrak data, quality, dan source codes
            data_values, quality_codes, source_codes = extract_quality_source_codes(line)
            
            # Terapkan filter kualitas
            filtered_values, quality_info = apply_quality_filter(
                data_values, quality_codes, source_codes
            )
            
            # Update statistik
            total_quality_info['total_filtered'] += quality_info['filtered_points']
            total_quality_info['total_processed'] += quality_info['total_points']
            
            # Buat row data
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            
            # Map data ke header yang valid
            data_headers = [h for h in valid_headers if h not in ['YYYYMMDD', 'HHMM']]
            for i, header in enumerate(data_headers):
                if i < len(filtered_values):
                    row_data[header] = filtered_values[i]
                else:
                    row_data[header] = 'NaN'
            
            if len(row_data) >= 2:  # Minimal ada tanggal dan waktu
                data_rows.append(row_data)
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    if df.empty:
        print("❌ Tidak ada data yang valid setelah quality filtering")
        return None
    
    # Proses timestamp
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
        df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Konversi ke numerik dan tangani missing values
    for col in df.columns:
        if col != 'Timestamp':
            df[col] = df[col].apply(lambda x: np.nan if re.match(r'^-9\.9+$|^-999\.9+$', str(x)) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Format kolom timestamp menjadi Date, Year, Month, Day
    df = format_timestamp_columns(df)
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    
    # Statistik quality filtering
    filter_percentage = (total_quality_info['total_filtered'] / max(total_quality_info['total_processed'], 1)) * 100
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    print(f"📊 Quality Filter: {total_quality_info['total_filtered']}/{total_quality_info['total_processed']} data points difilter ({filter_percentage:.1f}%)")
    
    # Tampilkan missing values
    missing_values_df = df.isnull().sum().to_frame(name="Missing Values")
    print("📋 Missing Values setelah Quality Filtering:")
    print(missing_values_df)
    
    return output_file

def convert_wind_ascii_to_csv_with_quality(input_file, output_file):
    """Fungsi khusus untuk konversi file format angin dengan quality filtering"""
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Cari header data
    header_line = None
    for i, line in enumerate(lines):
        if 'Depth (M):' in line:
            if i + 1 < len(lines):
                header_line = lines[i + 1]
            break
    
    if not header_line:
        print("❌ Tidak dapat menemukan header untuk file angin")
        return None
    
    headers = header_line.strip().split()
    valid_headers = [h for h in headers if h in ['YYYYMMDD', 'HHMM', 'UWND', 'VWND', 'WSPD', 'WDIR']]
    
    data_rows = []
    total_quality_info = {'total_filtered': 0, 'total_processed': 0}
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            date = parts[0] if len(parts) > 0 else ''
            time = parts[1] if len(parts) > 1 else ''
            
            # Ekstrak dan filter data
            data_values, quality_codes, source_codes = extract_quality_source_codes(line)
            filtered_values, quality_info = apply_quality_filter(
                data_values, quality_codes, source_codes
            )
            
            total_quality_info['total_filtered'] += quality_info['filtered_points']
            total_quality_info['total_processed'] += quality_info['total_points']
            
            # Map ke header
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            data_headers = [h for h in valid_headers if h not in ['YYYYMMDD', 'HHMM']]
            
            for i, header in enumerate(data_headers):
                if i < len(filtered_values):
                    row_data[header] = filtered_values[i]
                else:
                    row_data[header] = 'NaN'
            
            if len(row_data) >= 2:
                data_rows.append(row_data)
    
    # Proses DataFrame
    df = pd.DataFrame(data_rows)
    
    if df.empty:
        print("❌ Tidak ada data yang valid setelah quality filtering")
        return None
    
    # Timestamp processing
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
        df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Convert to numeric
    for col in df.columns:
        if col != 'Timestamp':
            df[col] = df[col].apply(lambda x: np.nan if re.match(r'^-\d{1,2}\.?\d*$', str(x)) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Format kolom timestamp menjadi Date, Year, Month, Day
    df = format_timestamp_columns(df)
    
    df.to_csv(output_file, index=False)
    
    filter_percentage = (total_quality_info['total_filtered'] / max(total_quality_info['total_processed'], 1)) * 100
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    print(f"📊 Quality Filter: {total_quality_info['total_filtered']}/{total_quality_info['total_processed']} data points difilter ({filter_percentage:.1f}%)")
    
    return output_file

def convert_ascii_to_csv_with_quality_control(input_file, output_file=None):
    """
    Fungsi utama untuk konversi dengan quality control
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.csv'
    
    # Deteksi format file seperti sebelumnya
    with open(input_file, 'r') as f:
        preview_lines = [f.readline() for _ in range(20)]
    
    is_temperature_format = False
    is_wind_format = False
    
    # Deteksi logika sama seperti sebelumnya
    has_index_line = any('Index:' in line for line in preview_lines)
    
    depth_pattern = False
    for line in preview_lines:
        if 'Depth(M):' in line and len(line.split()) > 6:
            try:
                depth_parts = line.split(':')[1].strip().split()
                depth_count = sum(1 for part in depth_parts if re.match(r'^\d+\.?\d*$', part))
                if depth_count >= 3:
                    depth_pattern = True
                    break
            except (ValueError, IndexError):
                pass
    
    wind_pattern = False
    for i, line in enumerate(preview_lines):
        if 'Depth (M):' in line and 'WDIR' in ''.join(preview_lines[i:i+3]):
            wind_pattern = True
            break
    
    if has_index_line and depth_pattern:
        is_temperature_format = True
    elif wind_pattern:
        is_wind_format = True
    
    # Panggil fungsi yang sesuai dengan quality control
    if is_temperature_format:
        print(f"🌡️ Memproses format suhu dengan quality control: {input_file}")
        return convert_temperature_ascii_to_csv_with_quality(input_file, output_file)
    elif is_wind_format:
        print(f"🌬️ Memproses format angin dengan quality control: {input_file}")
        return convert_wind_ascii_to_csv_with_quality(input_file, output_file)
    else:
        print(f"📊 Memproses format umum dengan quality control: {input_file}")
        return convert_general_ascii_to_csv_with_quality(input_file, output_file)

def process_single_location_with_quality_control(input_directory, csv_directory, variables=None):
    """
    Memproses satu lokasi dengan quality control
    
    Parameters:
    input_directory (str): Path ke direktori ASCII input
    csv_directory (str): Path ke direktori CSV output  
    variables (list, optional): List variabel yang akan diproses
    
    Returns:
    dict: Summary hasil processing
    """
    if variables is None:
        variables = list(VARIABLES.keys())
    
    processing_summary = {
        'total_files': 0,
        'successful_conversions': 0,
        'failed_conversions': 0,
        'variables_processed': {var: 0 for var in variables}
    }
    
    print("🌊 MEMULAI KONVERSI DATA BUOY RAMA DENGAN QUALITY CONTROL 🌊")
    print("=" * 70)
    print(f"📂 Direktori input: {input_directory}")
    print(f"📂 Direktori output: {csv_directory}")
    print(f"📊 Variabel: {', '.join([VARIABLES[v] for v in variables])}")
    print("=" * 70)
    
    # Periksa apakah direktori input ada
    if not os.path.exists(input_directory):
        print(f"❌ Direktori input tidak ditemukan: {input_directory}")
        return processing_summary
    
    # Buat direktori output jika belum ada
    os.makedirs(csv_directory, exist_ok=True)
    
    # Proses setiap variabel
    for variable in variables:
        var_pattern = f"*{variable}*.ascii"
        var_files = glob.glob(os.path.join(input_directory, var_pattern))
        
        if not var_files:
            print(f"📋 {variable.upper()}: Tidak ada file ditemukan")
            continue
        
        print(f"📋 {variable.upper()}: Ditemukan {len(var_files)} file")
        
        for ascii_file in var_files:
            filename = os.path.basename(ascii_file)
            csv_filename = os.path.splitext(filename)[0] + '_filtered.csv'
            csv_output = os.path.join(csv_directory, csv_filename)
            
            try:
                result = convert_ascii_to_csv_with_quality_control(ascii_file, csv_output)
                if result:
                    processing_summary['successful_conversions'] += 1
                    processing_summary['variables_processed'][variable] += 1
                    print(f"   ✅ {filename}")
                else:
                    processing_summary['failed_conversions'] += 1
                    print(f"   ❌ {filename}")
                    
            except Exception as e:
                processing_summary['failed_conversions'] += 1
                print(f"   ❌ {filename}: {str(e)}")
            
            processing_summary['total_files'] += 1
    
    # Tampilkan ringkasan akhir
    print("\n" + "=" * 70)
    print("📋 RINGKASAN KONVERSI DENGAN QUALITY CONTROL")
    print("=" * 70)
    print(f"📁 Total file diproses: {processing_summary['total_files']}")
    print(f"✅ Berhasil: {processing_summary['successful_conversions']}")
    print(f"❌ Gagal: {processing_summary['failed_conversions']}")
    success_rate = (processing_summary['successful_conversions'] / max(processing_summary['total_files'], 1)) * 100
    print(f"📊 Tingkat keberhasilan: {success_rate:.1f}%")
    
    print(f"\n📊 Per Variabel:")
    for variable, count in processing_summary['variables_processed'].items():
        print(f"   {VARIABLES[variable]}: {count} file")
    
    print(f"\n🎉 KONVERSI SELESAI! File CSV dengan quality filtering tersimpan di: {csv_directory}")
    
    return processing_summary

if __name__ == "__main__":
    # Definisikan direktori input dan output CSV
    input_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/0N90E/ASCII'
    csv_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/0N90E/CSV_FILTERED'
    
    # Opsi 1: Proses single location dengan quality control
    print("🚀 Memulai konversi dengan Quality Control untuk lokasi 0N90E")
    print("=" * 80)
    
    # Jalankan konversi dengan quality filtering
    summary = process_single_location_with_quality_control(
        input_directory=input_directory,
        csv_directory=csv_directory,
        variables=['sst', 't', 'rain', 'rh', 'wind', 'rad']  # Semua variabel
    )
    
    # Tampilkan ringkasan hasil
    print("\n" + "🎯 HASIL AKHIR KONVERSI" + "\n" + "=" * 50)
    print(f"📊 Total file diproses: {summary['total_files']}")
    print(f"✅ Konversi berhasil: {summary['successful_conversions']}")
    print(f"❌ Konversi gagal: {summary['failed_conversions']}")
    
    if summary['total_files'] > 0:
        success_percentage = (summary['successful_conversions'] / summary['total_files']) * 100
        print(f"📈 Persentase keberhasilan: {success_percentage:.2f}%")
    
    print(f"\n📁 File hasil tersimpan di: {csv_directory}")
    print("\n🔍 INFORMASI FORMAT OUTPUT:")
    print("   • Kolom Date: Tanggal dalam format YYYY-MM-DD")
    print("   • Kolom Year: Tahun (numerik)")
    print("   • Kolom Month: Bulan (1-12)")
    print("   • Kolom Day: Hari (1-31)")
    print("   • Urutan kolom: Date, Year, Month, Day, [variabel lainnya]")