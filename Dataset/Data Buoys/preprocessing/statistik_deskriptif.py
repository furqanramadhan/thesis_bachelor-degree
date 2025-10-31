import pandas as pd
import numpy as np
from pathlib import Path

# Konfigurasi
base_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Buoys"
locations = ['0N90E', '4N90E', '8N90E']

def calculate_descriptive_stats(df, location_name):
    """
    Menghitung statistik deskriptif untuk satu lokasi
    """
    print(f"\n{'='*100}")
    print(f"STATISTIK DESKRIPTIF: {location_name}")
    print(f"{'='*100}")
    
    # Variables to analyze
    variables = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad']
    
    # Filter hanya variabel yang ada
    available_vars = [v for v in variables if v in df.columns]
    
    # Buat dataframe statistik
    stats_data = []
    
    for var in available_vars:
        data_series = df[var]
        
        stats = {
            'Variabel': var,
            'Jumlah Data': len(data_series),
            'Jumlah Data Kosong': data_series.isna().sum(),
            'Minimum': data_series.min(),
            'Q1': data_series.quantile(0.25),
            'Median': data_series.median(),
            'Mean': data_series.mean(),
            'Q3': data_series.quantile(0.75),
            'Maksimum': data_series.max(),
            'Standar Deviasi': data_series.std()
        }
        
        stats_data.append(stats)
    
    # Buat DataFrame
    df_stats = pd.DataFrame(stats_data)
    
    # Print dengan format rapi
    print(f"\n{df_stats.to_string(index=False)}")
    
    # Print format tabel yang lebih detail
    print(f"\n\nFORMAT TABEL DETAIL:")
    print(f"{'-'*100}")
    print(f"{'Variabel':<15} {'Jumlah Data':<15} {'Data Kosong':<15} {'Minimum':<12} {'Q1':<12} {'Median':<12}")
    print(f"{'-'*100}")
    
    for _, row in df_stats.iterrows():
        print(f"{row['Variabel']:<15} {row['Jumlah Data']:<15} {row['Jumlah Data Kosong']:<15} {row['Minimum']:<12.2f} {row['Q1']:<12.2f} {row['Median']:<12.2f}")
    
    print(f"\n{'-'*100}")
    print(f"{'Variabel':<15} {'Mean':<12} {'Q3':<12} {'Maksimum':<12} {'Std Dev':<12}")
    print(f"{'-'*100}")
    
    for _, row in df_stats.iterrows():
        print(f"{row['Variabel']:<15} {row['Mean']:<12.2f} {row['Q3']:<12.2f} {row['Maksimum']:<12.2f} {row['Standar Deviasi']:<12.2f}")
    
    return df_stats

def create_summary_table(all_stats):
    """
    Membuat tabel ringkasan untuk semua lokasi
    """
    print(f"\n{'='*100}")
    print(f"RINGKASAN STATISTIK DESKRIPTIF - SEMUA LOKASI")
    print(f"{'='*100}")
    
    variables = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad']
    
    for var in variables:
        print(f"\n{'='*100}")
        print(f"VARIABEL: {var}")
        print(f"{'='*100}")
        print(f"{'Location':<12} {'N Data':<12} {'Missing':<12} {'Min':<10} {'Q1':<10} {'Median':<10} {'Mean':<10} {'Q3':<10} {'Max':<10} {'Std':<10}")
        print(f"{'-'*100}")
        
        for location, df_stats in all_stats.items():
            var_stats = df_stats[df_stats['Variabel'] == var]
            
            if not var_stats.empty:
                row = var_stats.iloc[0]
                print(f"{location:<12} {int(row['Jumlah Data']):<12} {int(row['Jumlah Data Kosong']):<12} "
                      f"{row['Minimum']:<10.2f} {row['Q1']:<10.2f} {row['Median']:<10.2f} "
                      f"{row['Mean']:<10.2f} {row['Q3']:<10.2f} {row['Maksimum']:<10.2f} "
                      f"{row['Standar Deviasi']:<10.2f}")
            else:
                print(f"{location:<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

def export_to_csv(all_stats, base_dir):
    """
    Export statistik ke CSV
    """
    output_dir = Path(base_dir) / 'Statistics_Summary'
    output_dir.mkdir(exist_ok=True)
    
    # Export per lokasi
    for location, df_stats in all_stats.items():
        output_file = output_dir / f'descriptive_stats_{location}.csv'
        df_stats.to_csv(output_file, index=False)
        print(f"✓ Exported: {output_file}")
    
    # Export gabungan
    combined_data = []
    for location, df_stats in all_stats.items():
        df_temp = df_stats.copy()
        df_temp['Location'] = location
        combined_data.append(df_temp)
    
    df_combined = pd.concat(combined_data, ignore_index=True)
    
    # Reorder columns
    cols = ['Location', 'Variabel', 'Jumlah Data', 'Jumlah Data Kosong', 
            'Minimum', 'Q1', 'Median', 'Mean', 'Q3', 'Maksimum', 'Standar Deviasi']
    df_combined = df_combined[cols]
    
    output_file_combined = output_dir / 'descriptive_stats_all_locations.csv'
    df_combined.to_csv(output_file_combined, index=False)
    print(f"✓ Exported: {output_file_combined}")

# Main process
print("\n" + "="*100)
print("MEMULAI ANALISIS STATISTIK DESKRIPTIF BUOYS")
print("="*100)

all_stats = {}

for location in locations:
    combined_file = Path(base_dir) / location / 'CSV' / 'COMBINED' / f'{location}.csv'
    
    if not combined_file.exists():
        print(f"\n⚠️  File tidak ditemukan: {combined_file}")
        continue
    
    # Load data
    df = pd.read_csv(combined_file)
    
    # Hitung statistik
    df_stats = calculate_descriptive_stats(df, location)
    all_stats[location] = df_stats

# Buat tabel ringkasan
if all_stats:
    create_summary_table(all_stats)
    
    # Export ke CSV
    print(f"\n{'='*100}")
    print("EXPORTING TO CSV...")
    print(f"{'='*100}\n")
    export_to_csv(all_stats, base_dir)

print(f"\n{'='*100}")
print("✅ ANALISIS STATISTIK DESKRIPTIF SELESAI")
print(f"{'='*100}")