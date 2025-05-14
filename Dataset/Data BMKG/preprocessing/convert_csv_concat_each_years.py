import os
import pandas as pd
import re
from datetime import datetime
import glob
import numpy as np

def process_bmkg_excel(file_path):
    """
    Membaca file Excel BMKG, menghapus 7 baris teratas dan baris bawah setelah data,
    dan memisahkan tanggal menjadi kolom Date, Year, Month, Day.
    
    Args:
        file_path: Path ke file Excel BMKG
        
    Returns:
        DataFrame hasil proses
    """
    print(f"Memproses file: {file_path}")
    
    # Baca Excel tanpa header
    df = pd.read_excel(file_path, header=None)
    
    # Cari indeks baris yang berisi header kolom (biasanya baris ke-7, indeks 6)
    header_row_idx = None
    for i in range(10):  # Cek 10 baris pertama
        row = df.iloc[i].astype(str)
        if "TANGGAL" in row.values:
            header_row_idx = i
            break
    
    if header_row_idx is None:
        print(f"Warning: Header tidak ditemukan di file {file_path}")
        return None
    
    # Gunakan baris header sebagai nama kolom dan hapus baris-baris sebelumnya
    headers = df.iloc[header_row_idx].tolist()
    df = df.iloc[header_row_idx+1:].reset_index(drop=True)
    df.columns = headers
    
    # Replace text-based missing values dengan NaN - menggunakan pendekatan modern
    # Buat mask untuk nilai yang ingin diganti dengan NaN
    missing_mask = df.isin(['-', 'nan', 'NaN', 'NULL', ''])
    # Terapkan mask untuk mengubah nilai menjadi NaN
    df = df.mask(missing_mask, np.nan)
    
    # Cari indeks baris terakhir sebelum keterangan
    last_row_idx = len(df)
    for i in range(len(df)):
        # Cek apakah baris berisi data valid atau hanya berisi nilai kosong/NaN
        row_data = df.iloc[i].dropna()
        if len(row_data) == 0 or (isinstance(df.iloc[i, 0], str) and ("KETERANGAN" in df.iloc[i, 0] or df.iloc[i, 0].strip() == "")):
            last_row_idx = i
            break
    
    # Ambil hanya data yang valid
    df = df.iloc[:last_row_idx]
    
    # Pastikan kolom TANGGAL ada
    if "TANGGAL" not in df.columns:
        print(f"Warning: Kolom TANGGAL tidak ditemukan di file {file_path}")
        return None
    
    # Konversi kolom TANGGAL ke datetime dan ekstrak Year, Month, Day
    # Format tanggal biasanya DD-MM-YYYY
    df['Date'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y', errors='coerce')
    
    # Jika konversi gagal, coba format lain
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    
    # Ekstrak komponen tanggal sebagai integer
    df['Year'] = df['Date'].dt.year.astype('Int64')  # Gunakan Int64 untuk mendukung nilai NaN
    df['Month'] = df['Date'].dt.month.astype('Int64')
    df['Day'] = df['Date'].dt.day.astype('Int64')
    
    # Reorganisasi kolom
    new_cols = ['Date', 'Year', 'Month', 'Day'] + [col for col in df.columns if col not in ['Date', 'Year', 'Month', 'Day', 'TANGGAL']]
    df = df[new_cols]
    
    # Ganti nilai 8888 dan 9999 dengan NaN - Memperbaiki warning
    # Konversi terlebih dahulu ke tipe yang sesuai untuk menghindari downcasting warning
    for col in df.columns:
        if col != 'Date' and pd.api.types.is_numeric_dtype(df[col]):
            # Ganti nilai 8888 dan 9999 dengan NaN
            mask_8888 = df[col] == 8888
            mask_9999 = df[col] == 9999
            if mask_8888.any() or mask_9999.any():
                # Buat salinan Series dengan tipe data yang mendukung NaN
                series = df[col].copy().astype('float64')
                series[mask_8888 | mask_9999] = np.nan
                df[col] = series
    
    # Konversi kolom numerik ke tipe data yang sesuai (tanpa desimal jika nilai selalu integer)
    for col in df.columns:
        if col not in ['Date', 'TANGGAL']:
            # Cek apakah kolom berisi nilai numerik
            if pd.api.types.is_numeric_dtype(df[col]):
                # Cek apakah semua nilai non-NaN adalah integer
                non_na_values = df[col].dropna()
                if len(non_na_values) > 0:
                    # Jika semua nilai adalah bilangan bulat, konversi ke integer
                    if all(non_na_values == non_na_values.astype(int)):
                        df[col] = df[col].astype('Int64')  # Int64 untuk mendukung nilai NaN
    
    # Hapus baris yang seluruhnya NaN
    df = df.dropna(how='all')
    
    return df

def process_year_directory(year_dir):
    """
    Memproses semua file Excel dalam direktori tahun tertentu
    dan menggabungkannya menjadi satu DataFrame.
    
    Args:
        year_dir: Path ke direktori tahun
        
    Returns:
        DataFrame gabungan dari semua file dalam direktori
    """
    print(f"Memproses direktori: {year_dir}")
    
    # Cari semua file Excel di direktori
    excel_files = glob.glob(os.path.join(year_dir, "*.xlsx"))
    
    if not excel_files:
        print(f"Tidak ada file Excel di direktori {year_dir}")
        return None
    
    # Proses semua file dan gabungkan hasilnya
    dfs = []
    for file_path in excel_files:
        df = process_bmkg_excel(file_path)
        if df is not None and not df.empty:
            dfs.append(df)
    
    if not dfs:
        print(f"Tidak ada data valid di direktori {year_dir}")
        return None
    
    # Gabungkan semua DataFrame
    combined_df = pd.concat(dfs, ignore_index=True, copy=False)
    
    # Urutkan berdasarkan tanggal
    combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)
    
    return combined_df

def main():
    # Direktori induk yang berisi subfolder tahun (Excel data)
    parent_dir = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/EXCEL'
    
    # Output direktori untuk file CSV (folder terpisah di direktori yang sama)
    output_dir = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV'
    os.makedirs(output_dir, exist_ok=True)
    
    # Cari semua subfolder tahun
    year_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d)) and d.isdigit()]
    
    for year in sorted(year_dirs):
        year_path = os.path.join(parent_dir, year)
        combined_df = process_year_directory(year_path)
        
        if combined_df is not None and not combined_df.empty:
            # Simpan sebagai CSV
            output_file = os.path.join(output_dir, f"BMKG_Data_{year}.csv")
            combined_df.to_csv(output_file, index=False)
            print(f"File CSV untuk tahun {year} telah dibuat: {output_file}")
            print(f"Jumlah baris data: {len(combined_df)}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()