import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import re
from typing import List, Dict, Optional
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# Set style untuk plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def detect_header_end(file_path):
    """
    Deteksi akhir header NASA POWER menggunakan regex pattern
    Returns: number of lines to skip
    """
    header_end_pattern = r'-END HEADER-'
    
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                if re.search(header_end_pattern, line):
                    return line_num  # Skip hingga baris setelah -END HEADER-
        
        # Jika tidak ditemukan pattern, coba dengan fallback
        print("⚠️  Warning: '-END HEADER-' pattern tidak ditemukan, menggunakan fallback skip=10")
        return 10
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 10  # Default fallback

class NASAPowerPrecipitationPreprocessor:
    def __init__(self, data):
        """
        Inisialisasi preprocessor untuk data curah hujan NASA POWER
        
        Parameters:
        data: DataFrame dengan kolom YEAR, DOY, PRECTOTCORR, dan kolom lainnya
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.processed_data = None
        self.missing_stats = {}
        self.outliers = {}
        
    def load_and_prepare_data(self):
        """
        Load data dan persiapan awal - konversi DOY ke Date
        """
        print("=== LOADING DAN PERSIAPAN DATA NASA POWER ===")
        
        # Konversi YEAR dan DOY ke Date
        self.data['Date'] = pd.to_datetime(
            self.data['YEAR'].astype(str) + '-' + self.data['DOY'].astype(str).str.zfill(3), 
            format='%Y-%j'
        )
        

        # Sort berdasarkan tanggal
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Buat kolom tambahan
        self.data['Year'] = self.data['Date'].dt.year
        self.data['month'] = self.data['Date'].dt.month
        self.data['day'] = self.data['Date'].dt.day
        
        # Handle missing values NASA (-999)
        self.data['PRECTOTCORR'] = self.data['PRECTOTCORR'].replace(-999, np.nan)
        
        print(f"✅ Data loaded: {len(self.data)} records from {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"📋 Kolom curah hujan: PRECTOTCORR → PRECTOTCORR")

        return self.data
    
    def analyze_precipitation_statistics(self):
        """
        Analisis statistik dasar curah hujan NASA POWER (mirip BMKG format)
        """
        print("\n=== ANALISIS STATISTIK CURAH HUJAN NASA POWER ===")
        
        # Data curah hujan yang valid
        valid_prec = self.data['PRECTOTCORR'].dropna()
        
        # Hitung statistik dasar
        stats = {
            'Jumlah Data': len(self.data),
            'Jumlah Data Kosong': self.data['PRECTOTCORR'].isna().sum(),
            'Jumlah Data Valid': len(valid_prec),
            'Minimum': valid_prec.min() if len(valid_prec) > 0 else np.nan,
            'Q1 (25%)': valid_prec.quantile(0.25) if len(valid_prec) > 0 else np.nan,
            'Median (Q2)': valid_prec.median() if len(valid_prec) > 0 else np.nan,
            'Mean': valid_prec.mean() if len(valid_prec) > 0 else np.nan,
            'Q3 (75%)': valid_prec.quantile(0.75) if len(valid_prec) > 0 else np.nan,
            'Maksimum': valid_prec.max() if len(valid_prec) > 0 else np.nan,
            'Standar Deviasi': valid_prec.std() if len(valid_prec) > 0 else np.nan
        }
        
        # Simpan statistik untuk reference
        self.missing_stats = {
            'total_records': len(self.data),
            'missing_count': self.data['PRECTOTCORR'].isna().sum(),
            'valid_count': len(valid_prec),
            'missing_percentage': (self.data['PRECTOTCORR'].isna().sum() / len(self.data)) * 100
        }
        
        # Print statistik dalam format yang mirip BMKG
        print("\n📊 STATISTIK DESKRIPTIF CURAH HUJAN:")
        print("=" * 50)
        
        for key, value in stats.items():
            if pd.isna(value):
                print(f"{key:<20}: N/A")
            elif key in ['Jumlah Data', 'Jumlah Data Kosong', 'Jumlah Data Valid']:
                print(f"{key:<20}: {int(value):,}")
            else:
                print(f"{key:<20}: {value:.2f} mm")
        
        # Tambahan: Persentase missing dan distribusi
        if len(valid_prec) > 0:
            print(f"\n📋 INFORMASI TAMBAHAN:")
            print(f"Persentase Data Kosong  : {self.missing_stats['missing_percentage']:.2f}%")
            print(f"Persentase Data Valid   : {(len(valid_prec)/len(self.data))*100:.2f}%")
            print(f"Skewness               : {valid_prec.skew():.3f}")
            print(f"Kurtosis               : {valid_prec.kurtosis():.3f}")
            
        return stats
    
    def analyze_missing_patterns(self):
        """
        Analisis pola missing values per tahun dan bulan
        """
        print("\n=== ANALISIS POLA MISSING VALUES ===")
        
        # Missing values per tahun
        yearly_missing = self.data.groupby('Year').agg({
            'PRECTOTCORR': [
                'count',
                lambda x: x.isna().sum(),
                lambda x: (x == 0).sum()
            ]
        }).round(2)

        
        yearly_missing.columns = ['Total_Records', 'Missing_Count', 'Zero_Values']
        yearly_missing['Missing_Percentage'] = (yearly_missing['Missing_Count'] / yearly_missing['Total_Records']) * 100
        yearly_missing['Valid_Data'] = yearly_missing['Total_Records'] - yearly_missing['Missing_Count']
        
        print("\n📅 MISSING VALUES PER TAHUN:")
        print(yearly_missing.to_string())
        
        # Missing values per bulan
        monthly_missing = self.data.groupby('month').agg({
            'PRECTOTCORR': [
                'count',
                lambda x: x.isna().sum()
            ]
        }).round(2)
        
        monthly_missing.columns = ['Total_Records', 'Missing_Count']
        monthly_missing['Missing_Percentage'] = (monthly_missing['Missing_Count'] / monthly_missing['Total_Records']) * 100
        
        print(f"\n📅 MISSING VALUES PER BULAN:")
        print(monthly_missing.to_string())
        
        return yearly_missing, monthly_missing
    
    def categorize_precipitation(self):
        """
        Kategorisasi curah hujan berdasarkan intensitas sesuai standar BMKG
        """
        print("\n=== KATEGORISASI CURAH HUJAN (STANDAR BMKG) ===")

        valid_data = self.data['PRECTOTCORR'].dropna()

        if len(valid_data) == 0:
            print("❌ Tidak ada data valid untuk kategorisasi")
            return {}
        
        # Kategori berdasarkan BMKG
        categories = {
            'Berawan (0–0.4 mm)': ((valid_data >= 0) & (valid_data <= 0.4)).sum(),
            'Hujan ringan (0.5–19.9 mm)': ((valid_data > 0.4) & (valid_data <= 19.9)).sum(),
            'Hujan sedang (20–49.9 mm)': ((valid_data >= 20) & (valid_data <= 49.9)).sum(),
            'Hujan lebat (50–99.9 mm)': ((valid_data >= 50) & (valid_data <= 99.9)).sum(),
            'Hujan sangat lebat (100–150 mm)': ((valid_data >= 100) & (valid_data <= 150)).sum(),
            'Hujan ekstrem (>150 mm)': (valid_data > 150).sum()
        }
        
        print("📊 DISTRIBUSI KATEGORI CURAH HUJAN:")
        print("=" * 55)
        
        total_valid = len(valid_data)
        for category, count in categories.items():
            percentage = (count / total_valid) * 100
            print(f"{category:<30}: {count:>6,} ({percentage:>5.1f}%)")
        
        print(f"{'Total':<30}: {total_valid:>6,} (100.0%)")
        
        return categories
    
    def create_basic_visualization(self, output_dir="nasa_prec_plots", save_plots=True):
        """
        Membuat visualisasi dasar untuk data NASA POWER precipitation
        """
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Direktori {output_dir} dibuat")
        
        print("\n=== MEMBUAT VISUALISASI DASAR ===")
        
        # Set style
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
    
        print("🔄 Plot #1: Time Series Curah Hujan NASA POWER")

        plt.figure(figsize=(16, 6))

        # Data preparation
        valid_data = self.data[self.data['Date'].notna() & self.data['PRECTOTCORR'].notna()].sort_values('Date')

        if not valid_data.empty:
            # Full time series (only line, no dots)
            plt.plot(valid_data['Date'], valid_data['PRECTOTCORR'],
                    color='blue', alpha=0.7, linewidth=1.5, label='PRECTOTCORR')
            
            # Remove scatter for extreme events (no dots)
            # (No plt.scatter here)
            
            # Set y-axis limit ke nilai maksimum selama 20 tahun + padding 5%
            plt.ylim(0, valid_data['PRECTOTCORR'].max() * 1.05)
            
            plt.ylabel('Curah Hujan (mm/day)', fontsize=12)
            plt.xlabel('Tahun', fontsize=12)
            plt.title('NASA POWER: Precipitation Corrected (PRECTOTCORR) Time Series', fontsize=13, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.legend()

        plt.tight_layout()
        if save_plots:
                plt.savefig(f'{output_dir}/01_nasa_precipitation_timeseries.png', dpi=300, 
                        bbox_inches='tight', facecolor='white')
        print("✅ 01_nasa_precipitation_timeseries.png saved")
        
        plt.show()
        
        # PLOT 2: Boxplot distribusi bulanan
        print("🔄 Plot #2: Distribusi Bulanan Curah Hujan")
        
        plt.figure(figsize=(14, 8))
        
        # Prepare monthly data
        months = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        month_labels = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei',
                        'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov']
        
        monthly_data = []
        for month in months:
            month_prec = self.data[self.data['month'] == month]['PRECTOTCORR'].dropna()
            monthly_data.append(month_prec if not month_prec.empty else pd.Series(dtype=float))
        
        if any(len(m) > 0 for m in monthly_data):
            box_plot = plt.boxplot(monthly_data, labels=month_labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        plt.ylabel('Curah Hujan (mm/day)', fontsize=12)
        plt.xlabel('Bulan', fontsize=12)
        plt.title('Distribusi Curah Hujan Bulanan NASA POWER (2005-2025)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        
        if save_plots:
            plt.savefig(f'{output_dir}/02_nasa_monthly_distribution.png', dpi=300, 
                        bbox_inches='tight', facecolor='white')
            print("✅ 02_nasa_monthly_distribution.png saved")
        
        plt.show()
        
        print("🔄 Plot #3: Dekomposisi Deret Waktu Curah Hujan (Trend, Seasonal, Residual)")

        # Persiapkan data untuk dekomposisi
        decomp_data = self.data[['Date', 'PRECTOTCORR']].copy()
        decomp_data = decomp_data.dropna()
        decomp_data = decomp_data.set_index('Date')
        decomp_data = decomp_data.sort_index()

        # Pastikan data memiliki frekuensi yang konsisten
        decomp_data = decomp_data.asfreq('D')

        # Interpolasi untuk gap kecil (maksimal 7 hari)
        decomp_data['PRECTOTCORR'] = decomp_data['PRECTOTCORR'].interpolate(method='linear', limit=7)

        # Hapus data yang masih NaN setelah interpolasi
        decomp_data = decomp_data.dropna()

        if len(decomp_data) >= 730:  # Minimal 2 tahun data
            try:
                # Seasonal decomposition dengan periode 365 hari (tahunan)
                decomposition = seasonal_decompose(
                    decomp_data['PRECTOTCORR'],
                    model='additive',  # additive cocok untuk curah hujan harian
                    period=365,
                    extrapolate_trend='freq'
                )

                # Plot dekomposisi (hanya 3 komponen)
                fig, axes = plt.subplots(3, 1, figsize=(16, 10))

                # Trend
                axes[0].plot(decomposition.trend.index, decomposition.trend.values,
                            color='blue', linewidth=2)
                axes[0].set_ylabel('Trend (mm)', fontsize=11)
                axes[0].set_title('Dekomposisi Deret Waktu Curah Hujan NASA POWER',
                                fontsize=13, fontweight='bold')
                axes[0].grid(True, alpha=0.3)

                # Seasonal
                axes[1].plot(decomposition.seasonal.index, decomposition.seasonal.values,
                            color='blue', linewidth=1)
                axes[1].set_ylabel('Seasonal (mm)', fontsize=11)
                axes[1].grid(True, alpha=0.3)

                # Residual
                axes[2].plot(decomposition.resid.index, decomposition.resid.values,
                            color='blue', linewidth=0.5, alpha=0.7)
                axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axes[2].set_ylabel('Residual (mm)', fontsize=11)
                axes[2].set_xlabel('Tahun', fontsize=11)
                axes[2].grid(True, alpha=0.3)

                plt.tight_layout()

                if save_plots:
                    plt.savefig(f'{output_dir}/03_nasa_precipitation_decomposition.png',
                                dpi=300, bbox_inches='tight', facecolor='white')
                    print("✅ 03_nasa_precipitation_decomposition.png saved")

                plt.show()

                # Print statistik dekomposisi
                print("\n📊 STATISTIK DEKOMPOSISI:")
                print(f"   • Trend range: {decomposition.trend.min():.2f} mm - {decomposition.trend.max():.2f} mm")
                print(f"   • Seasonal amplitude: ±{decomposition.seasonal.abs().max():.2f} mm")
                print(f"   • Residual std: {decomposition.resid.std():.2f} mm")

            except Exception as e:
                print(f"⚠️  Warning: Tidak dapat membuat dekomposisi - {str(e)}")
        else:
            print(f"⚠️  Warning: Data tidak cukup untuk dekomposisi (minimum 2 tahun diperlukan)")
            print(f"   Data tersedia: {len(decomp_data)} hari")
    
    def summary_report(self):
        """
        Laporan ringkasan data NASA POWER precipitation
        """
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN DATA NASA POWER PRECIPITATION")
        print("="*60)
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Periode: {self.data['Date'].min()} s/d {self.data['Date'].max()}")
        print(f"   • Rentang tahun: {self.data['Year'].max() - self.data['Year'].min() + 1} tahun")
        print(f"   • Parameter: PRECTOTCORR (Precipitation Corrected)")
        print(f"   • Sumber: NASA POWER MERRA-2")
        print(f"   • Resolusi: Harian")
        
        # Data quality overview
        valid_data = self.data['PRECTOTCORR'].dropna()
        missing_count = self.data['PRECTOTCORR'].isna().sum()
        
        print(f"\n🔍 KUALITAS DATA:")
        print(f"   • Data valid: {len(valid_data):,} ({len(valid_data)/len(self.data)*100:.2f}%)")
        print(f"   • Missing values: {missing_count:,} ({missing_count/len(self.data)*100:.2f}%)")
        
        if len(valid_data) > 0:
            print(f"   • Range nilai: {valid_data.min():.2f} - {valid_data.max():.2f} mm/day")
            print(f"   • Nilai negatif: {(valid_data < 0).sum():,}")
            
            # Extreme events summary
            extreme_count = (valid_data > 50).sum()
            very_extreme_count = (valid_data > 100).sum()
            
            print(f"\n⚡ EVENTS EKSTREM:")
            print(f"   • Hujan lebat (>50mm): {extreme_count:,} hari ({extreme_count/len(valid_data)*100:.2f}%)")
            print(f"   • Hujan sangat lebat (>100mm): {very_extreme_count:,} hari ({very_extreme_count/len(valid_data)*100:.2f}%)")
            
            # Seasonal summary
            seasonal_mean = self.data.groupby('month')['PRECTOTCORR'].mean().dropna()
            if not seasonal_mean.empty:
                wet_month = seasonal_mean.idxmax()
                dry_month = seasonal_mean.idxmin()
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                              'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
                
                print(f"\n🌦️ POLA MUSIMAN:")
                print(f"   • Bulan terbasah: {month_names[wet_month]} ({seasonal_mean[wet_month]:.2f} mm/day)")
                print(f"   • Bulan terkering: {month_names[dry_month]} ({seasonal_mean[dry_month]:.2f} mm/day)")
        
        print(f"\n📋 STATUS PREPROCESSING:")
        print(f"   • ✅ Data loading dan konversi format")
        print(f"   • ✅ Analisis statistik deskriptif")
        print(f"   • ✅ Identifikasi missing values")
        print(f"   • ✅ Kategorisasi intensitas curah hujan")
        print(f"   • ⏳ Advanced preprocessing (belum diimplementasi)")
        
        print(f"\n🎯 READY FOR ADVANCED PREPROCESSING!")
        print("   • Log transformation")
        print("   • Missing value imputation")
        print("   • Outlier detection")
        print("   • Seasonal decomposition")
        
        print("="*60)

def main():
    """
    Fungsi utama untuk menjalankan basic preprocessing NASA POWER precipitation
    """
    print("🛰️  BASIC PREPROCESSING NASA POWER PRECIPITATION")
    print("="*60)
    
    # Setup direktori output
    output_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Power NASA/preprocessing/curah hujan"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Data path - sesuaikan dengan lokasi file NASA POWER Anda
        data_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Power NASA/Aceh Jaya/Kec Setia Bakti/POWER_Point_Daily_20050101_20250930_004d83N_095d49E_LST.csv"
        
        # Load data
        print(f"📂 Loading NASA POWER data dari: {data_path}")
        
        # Deteksi akhir header menggunakan regex
        skip_rows = detect_header_end(data_path)
        print(f"🔍 Header detection: Skipping {skip_rows} baris (hingga -END HEADER-)")
        
        # Read CSV dengan skip header yang terdeteksi
        df = pd.read_csv(data_path, skiprows=skip_rows)
        
        print(f"✅ Data berhasil dimuat: {len(df)} records")
        print(f"📋 Kolom tersedia: {list(df.columns)}")
        
        # Validasi kolom yang diperlukan
        required_columns = ['YEAR', 'DOY', 'PRECTOTCORR']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Kolom berikut tidak ditemukan: {missing_columns}")
            print(f"💡 Kolom yang tersedia: {list(df.columns)}")
            return None
        
        # Inisialisasi preprocessor
        preprocessor = NASAPowerPrecipitationPreprocessor(df)
        
        # FASE 1: Load dan persiapan data
        print("\n🔄 FASE 1: Loading dan Persiapan Data")
        preprocessor.load_and_prepare_data()
        
        # FASE 2: Analisis statistik dasar (format BMKG)
        print("\n🔄 FASE 2: Analisis Statistik Dasar")
        basic_stats = preprocessor.analyze_precipitation_statistics()
        
        # FASE 3: Analisis pola missing values
        print("\n🔄 FASE 3: Analisis Pola Missing Values")
        yearly_missing, monthly_missing = preprocessor.analyze_missing_patterns()
        
        # FASE 4: Kategorisasi curah hujan
        print("\n🔄 FASE 4: Kategorisasi Curah Hujan")
        categories = preprocessor.categorize_precipitation()
        
        # FASE 5: Visualisasi dasar
        print("\n🔄 FASE 5: Visualisasi Dasar")
        plots_dir = os.path.join(output_dir, "nasa_prec_plots")
        preprocessor.create_basic_visualization(output_dir=plots_dir, save_plots=True)
        
        # FASE 6: Laporan ringkasan
        print("\n🔄 FASE 6: Laporan Ringkasan")
        preprocessor.summary_report()
        
        # Simpan hasil basic preprocessing
        output_path = os.path.join(output_dir, "nasa_power_precipitation_basic.csv")
        
        # Simpan data dengan kolom yang sudah distandarisasi
        output_columns = [
            'Date', 'Year', 'month', 'day',
            'YEAR', 'DOY', 'PRECTOTCORR'
        ]
        
        available_cols = [col for col in output_columns if col in preprocessor.data.columns]
        preprocessor.data[available_cols].to_csv(output_path, index=False)
        
        print(f"\n🎉 BASIC PREPROCESSING SELESAI!")
        print(f"📁 File output: {output_path}")
        print(f"📊 Ready untuk advanced preprocessing selanjutnya!")
        
        return preprocessor
        
    except FileNotFoundError:
        print(f"❌ Error: File data NASA POWER tidak ditemukan!")
        print("📋 Pastikan file tersedia dengan format:")
        print("   • Header diawali dengan -BEGIN HEADER- dan diakhiri -END HEADER-")
        print("   • Kolom: YEAR, DOY, PRECTOTCORR")
        return None
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()