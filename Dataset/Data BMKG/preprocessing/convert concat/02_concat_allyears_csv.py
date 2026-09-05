import os
import pandas as pd
import glob

def merge_csv_files(csv_directory):
    """
    Menggabungkan semua file CSV BMKG di direktori yang ditentukan menjadi satu dataset.
    
    Args:
        csv_directory: Path ke direktori yang berisi file CSV BMKG
        
    Returns:
        DataFrame gabungan dari semua file CSV
    """
    print(f"Membaca file CSV dari direktori: {csv_directory}")
    
    # Cari semua file CSV yang sesuai dengan pola nama
    csv_files = glob.glob(os.path.join(csv_directory, "BMKG_Data_*.csv"))
    
    if not csv_files:
        print(f"Tidak ada file CSV yang ditemukan di direktori {csv_directory}")
        return None
    
    # Urutkan file berdasarkan tahun
    csv_files.sort()
    
    # List untuk menyimpan semua DataFrame
    all_dfs = []
    
    # Loop melalui semua file CSV
    for file_path in csv_files:
        # Ekstrak tahun dari nama file
        filename = os.path.basename(file_path)
        year = filename.replace("BMKG_Data_", "").replace(".csv", "")
        
        print(f"Membaca file: {filename}")
        
        try:
            # Baca file CSV
            df = pd.read_csv(file_path)
            
            # Pastikan format Date konsisten
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Tambahkan ke list DataFrame
            all_dfs.append(df)
            print(f"Berhasil membaca {len(df)} baris data dari tahun {year}")
            
        except Exception as e:
            print(f"Error saat membaca file {filename}: {str(e)}")
    
    if not all_dfs:
        print("Tidak ada data yang berhasil dibaca")
        return None
    
    # Gabungkan semua DataFrame
    print("Menggabungkan semua data...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    initial_count = len(combined_df)
    duplicates = combined_df[combined_df.duplicated(subset=['Date'], keep=False)]

    if not duplicates.empty:
        print(f"Duplikasi terdeteksi saat merge!")
        print(f"Jumlah: {len(duplicates)} baris")
        dup_dates = duplicates['Date'].unique()
        print(f" Tanggal: {sorted([d.strftime('%Y-%m-%d') for d in dup_dates])}")
        
    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='first')
    final_count = len(combined_df)

    if initial_count != final_count:
        print(f"{initial_count - final_count} baris duplikat dihapus dari hasil merge")
    
    # Urutkan berdasarkan tanggal
    combined_df = combined_df.sort_values(by='Date').reset_index(drop=True)
    
    return combined_df

def main():
    # Direktori yang berisi file CSV
    csv_directory = '/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Besar/Stasiun Klimatologi Aceh/CSV'
    
    # Output file untuk dataset gabungan
    output_file = os.path.join(csv_directory, "BMKG_Data_All.csv")
    
    # Gabungkan semua file CSV
    combined_df = merge_csv_files(csv_directory)
    
    if combined_df is not None:
        # Simpan dataset gabungan sebagai CSV
        combined_df.to_csv(output_file, index=False)
        print(f"\nDataset gabungan telah dibuat: {output_file}")
        print(f"Total jumlah baris data: {len(combined_df)}")
        print(f"Rentang tahun: {combined_df['Year'].min()} - {combined_df['Year'].max()}")
        
        # Tampilkan informasi tambahan
        print("\nInformasi Dataset:")
        print(f"Jumlah kolom: {len(combined_df.columns)}")
        print(f"Ukuran dataset: {combined_df.memory_usage().sum() / (1024*1024):.2f} MB")
        
        # Cek missing values
        missing_values = combined_df.isna().sum()
        print("\nJumlah nilai yang hilang (NA) per kolom:")
        for col, count in missing_values.items():
            if count > 0:
                percent = (count / len(combined_df)) * 100
                print(f"  {col}: {count} ({percent:.2f}%)")
    else:
        print("Tidak dapat membuat dataset gabungan")

if __name__ == "__main__":
    main()



