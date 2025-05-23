import os
import re
import numpy as np
import pandas as pd
import glob

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
    """Fungsi untuk konversi file format suhu multi-kedalaman"""
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

def convert_wind_ascii_to_csv(input_file, output_file):
    """Fungsi khusus untuk konversi file format angin"""
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
    for header in headers:
        if header in ['YYYYMMDD', 'HHMM', 'UWND', 'VWND', 'WSPD', 'WDIR']:
            valid_headers.append(header)
    
    # Baca data
    data_rows = []
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            # Ambil tanggal, waktu, dan komponen angin
            row_data = {}
            for i, header in enumerate(headers):
                if i < len(parts) and header in valid_headers:
                    row_data[header] = parts[i]
            
            if len(row_data) >= 2:  # Minimal ada tanggal dan waktu
                data_rows.append(row_data)
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Debug: tampilkan kolom yang berhasil diproses
    print(f"Kolom yang berhasil diproses: {df.columns.tolist()}")
    print(f"Jumlah baris data: {len(df)}")
    
    # Gabungkan kolom tanggal dan waktu ke timestamp
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
        df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Konversi nilai angin ke numerik dan tangani missing values
    for col in df.columns:
        if col != 'Timestamp':
            df[col] = df[col].apply(lambda x: np.nan if re.match(r'^-\d{1,2}\.?\d*$', str(x)) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    
    # Tampilkan total missing values
    missing_values_df = df.isnull().sum().to_frame(name="Jumlah Missing Values")
    missing_values_df.loc["Total Data yang hilang"] = missing_values_df.sum()
    print("Total Baris Hilang per Kolom:\n")
    print(missing_values_df)
    
    return output_file

def convert_general_ascii_to_csv(input_file, output_file):
    """Fungsi untuk konversi file format umum (rad, rain, rh, sst)"""
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
    
    # Identifikasi kolom-kolom data (bukan QUALITY/SOURCE)
    valid_headers = []
    for header in headers:
        if header not in ['QUALITY', 'SOURCE'] and header != '':
            valid_headers.append(header)
    
    # Proses baris data
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            # Pastikan panjang data sesuai dengan header
            if len(parts) >= len(valid_headers):
                # Ambil data sesuai header yang valid
                row_data = {header: parts[i] for i, header in enumerate(headers) if i < len(parts) and header in valid_headers}
                
                if len(row_data) >= 2:  # Minimal ada tanggal dan waktu
                    data_rows.append(row_data)
    
    # Buat DataFrame
    df = pd.DataFrame(data_rows)
    
    # Debug: tampilkan kolom yang berhasil diproses
    print(f"Kolom yang berhasil diproses: {df.columns.tolist()}")
    print(f"Jumlah baris data: {len(df)}")
    
    # Gabungkan kolom tanggal dan waktu ke timestamp
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], format='%Y%m%d %H%M', errors='coerce')
        df.drop(['YYYYMMDD', 'HHMM'], axis=1, inplace=True)
    
    # Konversi nilai ke numerik dan tangani missing values
    for col in df.columns:
        if col != 'Timestamp':
            # Identifikasi dan ganti nilai missing sesuai dengan pattern yang umum
            df[col] = df[col].apply(lambda x: np.nan if re.match(r'^-9\.9+$|^-999\.9+$', str(x)) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Simpan ke CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Berhasil menyimpan {len(df)} baris data ke {output_file}")
    
    # Tampilkan total missing values
    missing_values_df = df.isnull().sum().to_frame(name="Jumlah Missing Values")
    missing_values_df.loc["Total Data yang hilang"] = missing_values_df.sum()
    print("Total Baris Hilang per Kolom:\n")
    print(missing_values_df)
    
    return output_file

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
    
    # Pastikan direktori CSV ada
    os.makedirs(csv_directory, exist_ok=True)
    print(f"📁 Menggunakan direktori CSV: {csv_directory}")
    
    # Tampilkan info direktori Excel jika ditentukan
    if excel_directory:
        os.makedirs(excel_directory, exist_ok=True)
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
    
    if excel_directory:
        print(f"📊 File Excel yang dihasilkan: {len(processed_excel_files)} (disimpan di {excel_directory})")
    else:
        print(f"📊 File Excel yang dihasilkan: {len(processed_excel_files)} (disimpan bersama file CSV)")
    
    return processed_csv_files, processed_excel_files

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
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    
    try:
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
        writer.close()
        return None
    
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
        
        # Atur ulang urutan kolom
        date_cols = ['Date', 'Year', 'Month', 'Day']
        other_cols = [col for col in df.columns if col not in date_cols]
        df = df[date_cols + other_cols]
    
    return df

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

def process_multiple_files_with_excel(input_directory, output_directory=None, file_pattern='*.ascii'):
    """
    Memproses banyak file ASCII dalam satu direktori dan menghasilkan file CSV dan Excel.
    
    Parameters:
    input_directory (str): Path ke direktori yang berisi file ASCII
    output_directory (str, optional): Path direktori untuk menyimpan file hasil
    file_pattern (str, optional): Pola file yang akan diproses (default: *.ascii)
    
    Returns:
    tuple: (list of CSV files, list of Excel files)
    """
    if not os.path.exists(input_directory):
        print(f"❌ Direktori input tidak ditemukan: {input_directory}")
        return [], []
    
    # Pastikan output directory disediakan dan dibuat jika belum ada
    if output_directory is None:
        output_directory = os.path.join(input_directory, 'convert')
        print(f"⚠️ Direktori output tidak ditentukan, menggunakan default: {output_directory}")
    
    # Pastikan direktori output ada
    os.makedirs(output_directory, exist_ok=True)
    print(f"📁 Menggunakan direktori output: {output_directory}")
    
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
        base_output = os.path.join(output_directory, os.path.splitext(filename)[0])
        
        print(f"\n🔄 Memproses file {i}/{total_files}: {filename}...")
        try:
            # Konversi ke CSV dan Excel dengan format tanggal baru
            excel_file = convert_ascii_to_excel(input_file, base_output + '.csv')
            if excel_file:
                processed_excel_files.append(excel_file)
                processed_csv_files.append(base_output + '.csv')
                print(f"✅ Berhasil mengkonversi {filename}")
            else:
                print(f"⚠️ Gagal mengkonversi {filename}")
                
        except Exception as e:
            print(f"❌ Error saat memproses {filename}: {str(e)}")
    
    # Ringkasan hasil proses
    print(f"\n✅ Selesai memproses {len(processed_csv_files)}/{total_files} file")
    print(f"📄 File CSV yang dihasilkan: {len(processed_csv_files)}")
    print(f"📊 File Excel yang dihasilkan: {len(processed_excel_files)}")
    
    return processed_csv_files, processed_excel_files

if __name__ == "__main__":
    # Definisikan direktori input, output CSV, dan output Excel
    input_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/8N90E/ASCII'
    csv_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/8N90E/CSV'
    excel_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/8N90E/EXCEL'
    
    print("\n📋 KONVERSI DATA BUOY RAMA ASCII KE CSV DAN EXCEL 📋")
    print("=" * 50)
    print(f"📂 Direktori input: {input_directory}")
    print(f"📄 Direktori output CSV: {csv_directory}")
    print(f"📊 Direktori output Excel: {excel_directory}")
    print("=" * 50)
    
    # Proses semua file ASCII dalam direktori input
    csv_files, excel_files = process_multiple_files_with_excel(
        input_directory=input_directory,
        csv_directory=csv_directory,
        excel_directory=excel_directory,
        file_pattern='*.ascii'
    )
    
    # Menampilkan ringkasan lengkap jika ada banyak file yang diproses
    if len(csv_files) > 0:
        print("\n📋 Ringkasan Hasil Konversi:")
        print(f"✅ Berhasil mengkonversi {len(csv_files)} dari {len(glob.glob(os.path.join(input_directory, '*.ascii')))} file ASCII")
        
        # Hanya tampilkan 5 file pertama jika ada banyak file
        if len(csv_files) > 5:
            print(f"\n📄 Beberapa file CSV yang dihasilkan (5 dari {len(csv_files)}):")
            for i, file in enumerate(csv_files[:5], 1):
                print(f"   {i}. {os.path.basename(file)}")
            print(f"   ... dan {len(csv_files) - 5} file lainnya")
            
            print(f"\n📊 Beberapa file Excel yang dihasilkan (5 dari {len(excel_files)}):")
            for i, file in enumerate(excel_files[:5], 1):
                print(f"   {i}. {os.path.basename(file)}")
            print(f"   ... dan {len(excel_files) - 5} file lainnya")
        else:
            print("\n📄 File CSV yang dihasilkan:")
            for i, file in enumerate(csv_files, 1):
                print(f"   {i}. {os.path.basename(file)}")
            
            print("\n📊 File Excel yang dihasilkan:")
            for i, file in enumerate(excel_files, 1):
                print(f"   {i}. {os.path.basename(file)}")
    
    print("\n✨ Proses konversi selesai! ✨")