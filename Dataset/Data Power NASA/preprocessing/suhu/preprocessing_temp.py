import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import os
import re
from typing import List, Dict, Optional
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

class NASAPowerTemperaturePreprocessor:
    def __init__(self, data):
        """
        Inisialisasi preprocessor untuk data temperatur NASA POWER
        
        Parameters:
        data: DataFrame dengan kolom YEAR, DOY, T2M, dan kolom lainnya
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
        self.data['T2M'] = self.data['T2M'].replace(-999, np.nan)
        
        print(f"✅ Data loaded: {len(self.data)} records from {self.data['Date'].min()} to {self.data['Date'].max()}")
        print(f"📋 Kolom temperatur: T2M (Temperature at 2 Meters)")

        return self.data
    
    def analyze_temperature_statistics(self):
        """
        Analisis statistik dasar temperatur NASA POWER
        """
        print("\n=== ANALISIS STATISTIK TEMPERATUR NASA POWER ===")
        
        # Data temperatur yang valid
        valid_temp = self.data['T2M'].dropna()
        
        # Hitung statistik dasar
        stats = {
            'Jumlah Data': len(self.data),
            'Jumlah Data Kosong': self.data['T2M'].isna().sum(),
            'Jumlah Data Valid': len(valid_temp),
            'Minimum': valid_temp.min() if len(valid_temp) > 0 else np.nan,
            'Q1 (25%)': valid_temp.quantile(0.25) if len(valid_temp) > 0 else np.nan,
            'Median (Q2)': valid_temp.median() if len(valid_temp) > 0 else np.nan,
            'Mean': valid_temp.mean() if len(valid_temp) > 0 else np.nan,
            'Q3 (75%)': valid_temp.quantile(0.75) if len(valid_temp) > 0 else np.nan,
            'Maksimum': valid_temp.max() if len(valid_temp) > 0 else np.nan,
            'Standar Deviasi': valid_temp.std() if len(valid_temp) > 0 else np.nan
        }
        
        # Simpan statistik untuk reference
        self.missing_stats = {
            'total_records': len(self.data),
            'missing_count': self.data['T2M'].isna().sum(),
            'valid_count': len(valid_temp),
            'missing_percentage': (self.data['T2M'].isna().sum() / len(self.data)) * 100
        }
        
        # Print statistik
        print("\n📊 STATISTIK DESKRIPTIF TEMPERATUR:")
        print("=" * 50)
        
        for key, value in stats.items():
            if pd.isna(value):
                print(f"{key:<20}: N/A")
            elif key in ['Jumlah Data', 'Jumlah Data Kosong', 'Jumlah Data Valid']:
                print(f"{key:<20}: {int(value):,}")
            else:
                print(f"{key:<20}: {value:.2f} °C")
        
        # Tambahan: Persentase missing dan distribusi
        if len(valid_temp) > 0:
            print(f"\n📋 INFORMASI TAMBAHAN:")
            print(f"Persentase Data Kosong  : {self.missing_stats['missing_percentage']:.2f}%")
            print(f"Persentase Data Valid   : {(len(valid_temp)/len(self.data))*100:.2f}%")
            print(f"Range Temperatur        : {valid_temp.max() - valid_temp.min():.2f} °C")
            print(f"Skewness               : {valid_temp.skew():.3f}")
            print(f"Kurtosis               : {valid_temp.kurtosis():.3f}")
            
        return stats
    
    def analyze_missing_patterns(self):
        """
        Analisis pola missing values per tahun dan bulan
        """
        print("\n=== ANALISIS POLA MISSING VALUES ===")
        
        # Missing values per tahun
        yearly_missing = self.data.groupby('Year').agg({
            'T2M': [
                'count',
                lambda x: x.isna().sum()
            ]
        }).round(2)
        
        yearly_missing.columns = ['Total_Records', 'Missing_Count']
        yearly_missing['Missing_Percentage'] = (yearly_missing['Missing_Count'] / yearly_missing['Total_Records']) * 100
        yearly_missing['Valid_Data'] = yearly_missing['Total_Records'] - yearly_missing['Missing_Count']
        
        print("\n📅 MISSING VALUES PER TAHUN:")
        print(yearly_missing.to_string())
        
        # Missing values per bulan
        monthly_missing = self.data.groupby('month').agg({
            'T2M': [
                'count',
                lambda x: x.isna().sum()
            ]
        }).round(2)
        
        monthly_missing.columns = ['Total_Records', 'Missing_Count']
        monthly_missing['Missing_Percentage'] = (monthly_missing['Missing_Count'] / monthly_missing['Total_Records']) * 100
        
        print(f"\n📅 MISSING VALUES PER BULAN:")
        print(monthly_missing.to_string())
        
        return yearly_missing, monthly_missing
    
    def categorize_temperature(self):
        """
        Kategorisasi temperatur berdasarkan standar BMKG
        """
        print("\n=== KATEGORISASI TEMPERATUR (STANDAR BMKG) ===")

        valid_data = self.data['T2M'].dropna()

        if len(valid_data) == 0:
            print("❌ Tidak ada data valid untuk kategorisasi")
            return {}
        
        # Kategori berdasarkan BMKG untuk daerah tropis
        categories = {
            'Sangat Dingin (<18°C)': (valid_data < 18).sum(),
            'Dingin (18-22°C)': ((valid_data >= 18) & (valid_data < 22)).sum(),
            'Sejuk (22-26°C)': ((valid_data >= 22) & (valid_data < 26)).sum(),
            'Normal (26-28°C)': ((valid_data >= 26) & (valid_data < 28)).sum(),
            'Hangat (28-30°C)': ((valid_data >= 28) & (valid_data < 30)).sum(),
            'Panas (30-32°C)': ((valid_data >= 30) & (valid_data < 32)).sum(),
            'Sangat Panas (≥32°C)': (valid_data >= 32).sum()
        }
        
        print("📊 DISTRIBUSI KATEGORI TEMPERATUR:")
        print("=" * 55)
        
        total_valid = len(valid_data)
        for category, count in categories.items():
            percentage = (count / total_valid) * 100
            print(f"{category:<30}: {count:>6,} ({percentage:>5.1f}%)")
        
        print(f"{'Total':<30}: {total_valid:>6,} (100.0%)")
        
        return categories
    
    def create_basic_visualization(self, output_dir="nasa_temp_plots", save_plots=True):
        """
        Membuat visualisasi dasar untuk data NASA POWER temperature
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
        
        # PLOT 1: Time series temperatur
        print("🔄 Plot #1: Time Series Temperatur NASA POWER")
        
        plt.figure(figsize=(16, 6))
        
        # Data preparation
        valid_data = self.data[self.data['Date'].notna() & self.data['T2M'].notna()].sort_values('Date')
        
        if not valid_data.empty:
            # Full time series (only line, no dots, no legend)
            plt.plot(valid_data['Date'], valid_data['T2M'],
                    color='blue', alpha=0.7, linewidth=1.5)
            
            # (No scatter for extreme events, no legend)
            
            plt.ylabel('Temperatur (°C)', fontsize=12)
            plt.xlabel('Tahun', fontsize=12)
            plt.grid(True, alpha=0.3)
            # plt.legend()  # Legend removed
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/01_nasa_temperature_timeseries.png', dpi=300, 
                        bbox_inches='tight', facecolor='white')
            print("✅ 01_nasa_temperature_timeseries.png saved")
        
        plt.show()
        
      
              # PLOT 2: Boxplot distribusi bulanan
        print("🔄 Plot #2: Distribusi Bulanan Temperatur")
        
        # Create month names in order (Dec - Jan - Feb - ... - Nov)
        month_order = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
        month_names = {12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 
                       6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov'}

        # Map month numbers to names
        self.data['Month_Name'] = self.data['month'].map(month_names)

        # Prepare data for boxplot
        data_by_month = [self.data[self.data['month'] == month]['T2M'].dropna().values 
                         for month in [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 7))

        # Create boxplot
        bp = ax.boxplot(data_by_month, 
                        labels=month_order,
                        patch_artist=True,
                        notch=False,
                        showmeans=False,
                        showfliers=True,
                        widths=0.6)

        # Use single sky blue color for all boxes
        sky_blue = '#87CEEB'
        for patch in bp['boxes']:
            patch.set_facecolor(sky_blue)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Customize median lines
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # Customize whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
            
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)

        # Customize fliers (outliers)
        for flier in bp['fliers']:
            flier.set_marker('o')
            flier.set_markerfacecolor('white')
            flier.set_markeredgecolor('black')
            flier.set_markersize(5)
            flier.set_markeredgewidth(0.8)

        # Customize plot
        ax.set_xlabel('Month', fontsize=11, fontweight='normal')
        ax.set_ylabel('Temperature (°C)', fontsize=11, fontweight='normal')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y', color='gray')
        ax.set_axisbelow(True)

        # Set background color
        fig.patch.set_facecolor('white')

        # Customize ticks
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/02_nasa_monthly_temperature_distribution.png', dpi=300, 
                        bbox_inches='tight', facecolor='white')
            print("✅ 02_nasa_monthly_temperature_distribution.png saved")
        
        plt.show()
        
        # Print monthly statistics
        print("\n=== Monthly Temperature Statistics (°C) ===\n")
        for i, month in enumerate(month_order):
            month_num = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][i]
            data = self.data[self.data['month'] == month_num]['T2M'].dropna()
            if len(data) > 0:
                print(f"{month:>3}: Mean={data.mean():6.2f}, Median={data.median():6.2f}, "
                      f"Min={data.min():6.2f}, Max={data.max():6.2f}, Std={data.std():6.2f}")
        
        # PLOT 2B: Boxplot distribusi tahunan
        print("\n🔄 Plot #2B: Distribusi Tahunan Temperatur")
        
        # Get unique years and prepare data for boxplot
        years = sorted(self.data['Year'].unique())
        data_by_year = [self.data[self.data['Year'] == year]['T2M'].dropna().values for year in years]

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(16, 8))

        # Create boxplot
        bp = ax.boxplot(data_by_year, 
                        labels=years,
                        patch_artist=True,
                        notch=False,
                        showmeans=False,
                        showfliers=True,
                        widths=0.6)

        # Use single sky blue color for all boxes
        for patch in bp['boxes']:
            patch.set_facecolor(sky_blue)
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

        # Customize median lines
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # Customize whiskers and caps
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1)
            
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1)

        # Customize fliers (outliers)
        for flier in bp['fliers']:
            flier.set_marker('o')
            flier.set_markerfacecolor('white')
            flier.set_markeredgecolor('black')
            flier.set_markersize(5)
            flier.set_markeredgewidth(0.8)

        # Customize plot
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y', color='gray')
        ax.set_axisbelow(True)

        # Set background color
        fig.patch.set_facecolor('white')

        # Customize ticks
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/02b_nasa_annual_temperature_distribution.png', dpi=300, 
                        bbox_inches='tight', facecolor='white')
            print("✅ 02b_nasa_annual_temperature_distribution.png saved")
        
        plt.show()
        
        # Print annual statistics
        print("\n=== Annual Daily Temperature Statistics (°C) ===\n")
        for year in years:
            data = self.data[self.data['Year'] == year]['T2M'].dropna()
            if len(data) > 0:
                status = "(Partial)" if year == 2025 else "(Complete)"
                print(f"{year}: Mean={data.mean():6.2f}, Median={data.median():6.2f}, "
                      f"Min={data.min():6.2f}, Max={data.max():6.2f}, Std={data.std():6.2f} {status}")
      
        # PLOT 3: Dekomposisi Deret Waktu (BARU!)
        print("🔄 Plot #3: Dekomposisi Deret Waktu (Trend, Seasonal, Residual)")
        
        # Persiapkan data untuk dekomposisi
        decomp_data = self.data[['Date', 'T2M']].copy()
        decomp_data = decomp_data.dropna()
        decomp_data = decomp_data.set_index('Date')
        decomp_data = decomp_data.sort_index()
        
        # Pastikan data memiliki frekuensi yang konsisten
        decomp_data = decomp_data.asfreq('D')
        
        # Interpolasi untuk gap kecil (maksimal 7 hari)
        decomp_data['T2M'] = decomp_data['T2M'].interpolate(method='linear', limit=7)
        
        # Hapus data yang masih NaN setelah interpolasi
        decomp_data = decomp_data.dropna()
        
        if len(decomp_data) >= 730:  # Minimal 2 tahun data
            try:
                # Seasonal decomposition dengan periode 365 hari (tahunan)
                decomposition = seasonal_decompose(
                    decomp_data['T2M'], 
                    model='additive',  # additive untuk temperatur
                    period=365,
                    extrapolate_trend='freq'
                )
                
                # Plot dekomposisi (hanya 3 komponen)
                fig, axes = plt.subplots(3, 1, figsize=(16, 10))
                
                # Trend
                axes[0].plot(decomposition.trend.index, decomposition.trend.values, 
                           color='blue', linewidth=2)
                axes[0].set_ylabel('Trend (°C)', fontsize=11)
                axes[0].grid(True, alpha=0.3)
                
                # Seasonal
                axes[1].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                           color='blue', linewidth=1)
                axes[1].set_ylabel('Seasonal (°C)', fontsize=11)
                axes[1].grid(True, alpha=0.3)
                
                # Residual
                axes[2].scatter(decomposition.resid.index, decomposition.resid.values,
                               color='blue', s=20, alpha=0.6)  # s=20 for larger points
                axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axes[2].set_ylabel('Residual (°C)', fontsize=11)
                axes[2].set_xlabel('Year', fontsize=11)
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f'{output_dir}/03_nasa_temperature_decomposition.png', 
                              dpi=300, bbox_inches='tight', facecolor='white')
                    print("✅ 03_nasa_temperature_decomposition.png saved")
                
                plt.show()
                
                # Print statistik dekomposisi
                print("\n📊 STATISTIK DEKOMPOSISI:")
                print(f"   • Trend range: {decomposition.trend.min():.2f}°C - {decomposition.trend.max():.2f}°C")
                print(f"   • Seasonal amplitude: ±{decomposition.seasonal.abs().max():.2f}°C")
                print(f"   • Residual std: {decomposition.resid.std():.2f}°C")
                
            except Exception as e:
                print(f"⚠️  Warning: Tidak dapat membuat dekomposisi - {str(e)}")
        else:
            print(f"⚠️  Warning: Data tidak cukup untuk dekomposisi (minimum 2 tahun diperlukan)")
            print(f"   Data tersedia: {len(decomp_data)} hari")
    
    def summary_report(self):
        """
        Laporan ringkasan data NASA POWER temperature
        """
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN DATA NASA POWER TEMPERATURE")
        print("="*60)
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Periode: {self.data['Date'].min()} s/d {self.data['Date'].max()}")
        print(f"   • Rentang tahun: {self.data['Year'].max() - self.data['Year'].min() + 1} tahun")
        print(f"   • Parameter: T2M (Temperature at 2 Meters)")
        print(f"   • Sumber: NASA POWER MERRA-2")
        print(f"   • Resolusi: Harian")
        
        # Data quality overview
        valid_data = self.data['T2M'].dropna()
        missing_count = self.data['T2M'].isna().sum()
        
        print(f"\n🔍 KUALITAS DATA:")
        print(f"   • Data valid: {len(valid_data):,} ({len(valid_data)/len(self.data)*100:.2f}%)")
        print(f"   • Missing values: {missing_count:,} ({missing_count/len(self.data)*100:.2f}%)")
        
        if len(valid_data) > 0:
            print(f"   • Range nilai: {valid_data.min():.2f} - {valid_data.max():.2f} °C")
            print(f"   • Range temperatur: {valid_data.max() - valid_data.min():.2f} °C")
            
            # Temperature statistics
            hot_days = (valid_data > 30).sum()
            cold_days = (valid_data < 22).sum()
            normal_days = ((valid_data >= 26) & (valid_data < 28)).sum()
            
            print(f"\n🌡️ STATISTIK TEMPERATUR:")
            print(f"   • Hari panas (>30°C): {hot_days:,} hari ({hot_days/len(valid_data)*100:.2f}%)")
            print(f"   • Hari normal (26-28°C): {normal_days:,} hari ({normal_days/len(valid_data)*100:.2f}%)")
            print(f"   • Hari dingin (<22°C): {cold_days:,} hari ({cold_days/len(valid_data)*100:.2f}%)")
            
            # Seasonal summary
            seasonal_mean = self.data.groupby('month')['T2M'].mean().dropna()
            if not seasonal_mean.empty:
                hottest_month = seasonal_mean.idxmax()
                coldest_month = seasonal_mean.idxmin()
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                              'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
                
                print(f"\n🌦️ POLA MUSIMAN:")
                print(f"   • Bulan terpanas: {month_names[hottest_month]} ({seasonal_mean[hottest_month]:.2f}°C)")
                print(f"   • Bulan terdingin: {month_names[coldest_month]} ({seasonal_mean[coldest_month]:.2f}°C)")
                print(f"   • Variasi musiman: {seasonal_mean[hottest_month] - seasonal_mean[coldest_month]:.2f}°C")
        
        print(f"\n📋 STATUS PREPROCESSING:")
        print(f"   • ✅ Data loading dan konversi format")
        print(f"   • ✅ Analisis statistik deskriptif")
        print(f"   • ✅ Identifikasi missing values")
        print(f"   • ✅ Kategorisasi temperatur")
        print(f"   • ✅ Dekomposisi deret waktu (Trend, Seasonal, Residual)")
        print(f"   • ⏳ Advanced preprocessing (belum diimplementasi)")
        
        print(f"\n🎯 READY FOR ADVANCED PREPROCESSING!")
        print("   • Outlier detection")
        print("   • Missing value imputation")
        print("   • Data normalization")
        print("   • Feature engineering")
        
        print("="*60)

def main():
    """
    Fungsi utama untuk menjalankan basic preprocessing NASA POWER temperature
    """
    print("🛰️  BASIC PREPROCESSING NASA POWER TEMPERATURE")
    print("="*60)
    
    # Setup direktori output
    output_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Power NASA/preprocessing/suhu"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Data path 
        data_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Power NASA/Bireun/Kec Kota Juang/POWER_Point_Daily_20050101_20250930_005d19N_096d67E_LST.csv"
        
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
        required_columns = ['YEAR', 'DOY', 'T2M']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Kolom berikut tidak ditemukan: {missing_columns}")
            print(f"💡 Kolom yang tersedia: {list(df.columns)}")
            return None
        
        # Inisialisasi preprocessor
        preprocessor = NASAPowerTemperaturePreprocessor(df)
        
        # FASE 1: Load dan persiapan data
        print("\n🔄 FASE 1: Loading dan Persiapan Data")
        preprocessor.load_and_prepare_data()
        
        # FASE 2: Analisis statistik dasar
        print("\n🔄 FASE 2: Analisis Statistik Dasar")
        basic_stats = preprocessor.analyze_temperature_statistics()
        
        # FASE 3: Analisis pola missing values
        print("\n🔄 FASE 3: Analisis Pola Missing Values")
        yearly_missing, monthly_missing = preprocessor.analyze_missing_patterns()
        
        # FASE 4: Kategorisasi temperatur
        print("\n🔄 FASE 4: Kategorisasi Temperatur")
        categories = preprocessor.categorize_temperature()
        
        # FASE 5: Visualisasi dasar (termasuk dekomposisi)
        print("\n🔄 FASE 5: Visualisasi Dasar + Dekomposisi Deret Waktu")
        plots_dir = os.path.join(output_dir, "nasa_temp_plots")
        preprocessor.create_basic_visualization(output_dir=plots_dir, save_plots=True)
        
        # FASE 6: Laporan ringkasan
        print("\n🔄 FASE 6: Laporan Ringkasan")
        preprocessor.summary_report()
        
        # Simpan hasil basic preprocessing
        output_path = os.path.join(output_dir, "nasa_power_temperature_basic.csv")
        
        # Simpan data dengan kolom yang sudah distandarisasi
        output_columns = [
            'Date', 'Year', 'month', 'day',
            'YEAR', 'DOY', 'T2M'
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
        print("   • Kolom: YEAR, DOY, T2M")
        return None
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()