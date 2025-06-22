import pandas as pd
import os
from pathlib import Path

def read_combined_data(file_path):
    """
    Membaca file CSV gabungan dengan error handling
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Berhasil membaca file: {file_path}")
            print(f"Total records: {len(df)}")
            print(f"Kolom yang tersedia: {list(df.columns)}")
            return df
        else:
            print(f"Error: File {file_path} tidak ditemukan")
            return None
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

def validate_data(df):
    """
    Validasi data dan tampilkan informasi dasar
    """
    print("\n=== VALIDASI DATA ===")
    
    # Cek kolom Location
    if 'Location' not in df.columns:
        print("Error: Kolom 'Location' tidak ditemukan!")
        return False
    
    # Tampilkan unique locations
    unique_locations = df['Location'].unique()
    print(f"Lokasi yang ditemukan: {unique_locations}")
    
    # Tampilkan jumlah data per lokasi
    print("\nJumlah data per lokasi:")
    location_counts = df['Location'].value_counts()
    for location, count in location_counts.items():
        print(f"- {location}: {count} records")
    
    # Cek range tanggal
    if 'Date' in df.columns:
        print(f"\nRange tanggal: {df['Date'].min()} sampai {df['Date'].max()}")
    
    return True

def split_by_location(df, output_directory):
    """
    Memisahkan data berdasarkan lokasi dan menyimpan ke file terpisah
    """
    print(f"\n=== MEMISAHKAN DATA PER LOKASI ===")
    
    # Buat direktori output jika belum ada
    os.makedirs(output_directory, exist_ok=True)
    
    # Ambil unique locations dan urutkan
    locations = sorted(df['Location'].unique())
    
    split_summary = {}
    
    for location in locations:
        print(f"\nMemproses lokasi: {location}")
        
        # Filter data untuk lokasi tertentu
        location_data = df[df['Location'] == location].copy()
        
        # Sort berdasarkan Date untuk memastikan urutan chronological
        if 'Date' in location_data.columns:
            # Convert Date to datetime untuk sorting yang proper
            location_data['Date'] = pd.to_datetime(location_data['Date'])
            location_data = location_data.sort_values('Date')
            # Convert kembali ke string format
            location_data['Date'] = location_data['Date'].dt.strftime('%Y-%m-%d')
        
        # Hapus kolom Location karena sudah tidak diperlukan
        if 'Location' in location_data.columns:
            location_data = location_data.drop('Location', axis=1)
            print(f"  ✓ Kolom 'Location' dihapus dari data {location}")
        
        # Reset index
        location_data = location_data.reset_index(drop=True)
        
        # Nama file output
        output_filename = f"Buoys_Data_{location}.csv"
        output_path = os.path.join(output_directory, output_filename)
        
        # Simpan ke CSV
        try:
            location_data.to_csv(output_path, index=False)
            print(f"✓ Berhasil menyimpan: {output_path}")
            print(f"  Records: {len(location_data)}")
            print(f"  Kolom: {len(location_data.columns)} (tanpa Location)")
            
            # Simpan summary
            split_summary[location] = {
                'filename': output_filename,
                'records': len(location_data),
                'columns': len(location_data.columns),
                'date_range': f"{location_data['Date'].min()} to {location_data['Date'].max()}" if 'Date' in location_data.columns else "N/A"
            }
            
        except Exception as e:
            print(f"✗ Error menyimpan {output_path}: {e}")
    
    return split_summary

def create_summary_report(split_summary, output_directory):
    """
    Membuat laporan ringkasan pemisahan data
    """
    report_path = os.path.join(output_directory, "split_summary_report.txt")
    
    try:
        with open(report_path, 'w') as f:
            f.write("LAPORAN PEMISAHAN DATA BUOYS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Tanggal pemrosesan: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_records = 0
            for location, info in split_summary.items():
                f.write(f"Lokasi: {location}\n")
                f.write(f"  File: {info['filename']}\n")
                f.write(f"  Records: {info['records']}\n")
                f.write(f"  Kolom: {info['columns']} (tanpa Location)\n")
                f.write(f"  Range Tanggal: {info['date_range']}\n\n")
                total_records += info['records']
            
            f.write(f"TOTAL RECORDS: {total_records}\n")
            f.write(f"TOTAL LOKASI: {len(split_summary)}\n")
        
        print(f"\n✓ Laporan ringkasan disimpan: {report_path}")
        
    except Exception as e:
        print(f"✗ Error membuat laporan: {e}")

def preview_split_files(output_directory, split_summary):
    """
    Menampilkan preview dari setiap file yang telah dibuat
    """
    print(f"\n=== PREVIEW FILE HASIL SPLIT ===")
    
    for location, info in split_summary.items():
        file_path = os.path.join(output_directory, info['filename'])
        
        try:
            df_preview = pd.read_csv(file_path)
            print(f"\n--- {info['filename']} ---")
            print(f"Shape: {df_preview.shape}")
            print("3 baris pertama:")
            print(df_preview.head(3).to_string(index=False))
            
        except Exception as e:
            print(f"Error membaca preview {file_path}: {e}")

def main():
    """
    Fungsi utama untuk memisahkan data buoys
    """
    print("PROGRAM PEMISAHAN DATA BUOYS")
    print("=" * 40)
    
    # Pengaturan file input dan direktori output
    base_dir = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
    input_file = os.path.join(base_dir, 'CSV/Buoys_Data_All.csv')
    output_directory = os.path.join(base_dir, 'CSV/Separated')
    
    # 1. Baca file input
    print(f"Membaca file input: {input_file}")
    df = read_combined_data(input_file)
    
    if df is None:
        print("Program dihentikan karena file input tidak dapat dibaca.")
        return
    
    # 2. Validasi data
    if not validate_data(df):
        print("Program dihentikan karena validasi data gagal.")
        return
    
    # 3. Split data berdasarkan lokasi
    split_summary = split_by_location(df, output_directory)
    
    if not split_summary:
        print("Program dihentikan karena pemisahan data gagal.")
        return
    
    # 4. Buat laporan ringkasan
    create_summary_report(split_summary, output_directory)
    
    # 5. Tampilkan preview hasil
    preview_split_files(output_directory, split_summary)
    
    # 6. Ringkasan final
    print(f"\n{'='*50}")
    print("PEMISAHAN DATA SELESAI!")
    print(f"{'='*50}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_directory}")
    print(f"Files created: {len(split_summary)}")
    
    for location, info in split_summary.items():
        print(f"  - {info['filename']} ({info['records']} records)")

if __name__ == "__main__":
    main()