import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')
import os

class BuoysPreprocessor:
    def __init__(self, csv_path, output_dir="plots"):
        self.csv_path = csv_path
        self.df = None
        self.processed_df = None
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")
        
        # Konfigurasi lokasi dan batas waktu
        self.loc_ranges = {
            '4N90E': '2018-12-31',
            '0N90E': '2020-06-07',
            '8N90E': '2020-02-21'
        }
        
        # Parameter clipping
        self.sst_bounds = (20, 35)
        self.rad_bounds = (0, None)
        
    def load_data(self):
        """Tahap 1: Persiapan Data"""
        print("📋 Tahap 1: Persiapan Data")
        
        # Muat data
        self.df = pd.read_csv(self.csv_path, parse_dates=['Date'])
        print(f"Data dimuat: {len(self.df)} baris")
        
        # Seleksi kolom & filter waktu
        self.df = self.df[['Date', 'Location', 'SST', 'RAD']]
        self.df = self.df[(self.df['Date'] >= '2005-01-01') & (self.df['Date'] <= '2020-12-31')]
        print(f"Setelah filter: {len(self.df)} baris")
        
        # Info missing values awal
        print("Missing values per kolom:")
        print(self.df.isnull().sum())
        
    def handle_location_specific_ranges(self):
        """Tahap 2: Penanganan Khusus Lokasi"""
        print("\n⚠️ Tahap 2: Penanganan Khusus Lokasi")
        
        dfs = []
        for loc, end_date in self.loc_ranges.items():
            loc_df = self.df[self.df['Location'] == loc].copy()
            initial_count = len(loc_df)
            loc_df = loc_df[loc_df['Date'] <= end_date]
            final_count = len(loc_df)
            print(f"{loc}: {initial_count} → {final_count} baris (sampai {end_date})")
            dfs.append(loc_df)
            
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Total setelah potong: {len(self.df)} baris")
        
    def initial_cleaning(self):
        """Tahap 3: Pembersihan Awal"""
        print("\n🧹 Tahap 3: Pembersihan Awal")
        
        # Statistik sebelum clipping
        print("Statistik sebelum clipping:")
        print(f"SST: min={self.df['SST'].min():.2f}, max={self.df['SST'].max():.2f}")
        print(f"RAD: min={self.df['RAD'].min():.2f}, max={self.df['RAD'].max():.2f}")
        
        # Clip nilai ekstrem
        self.df['SST'] = self.df['SST'].clip(self.sst_bounds[0], self.sst_bounds[1])
        if self.rad_bounds[1] is not None:
            self.df['RAD'] = self.df['RAD'].clip(self.rad_bounds[0], self.rad_bounds[1])
        else:
            self.df['RAD'] = self.df['RAD'].clip(lower=self.rad_bounds[0])
        
        # Statistik setelah clipping
        print("Statistik setelah clipping:")
        print(f"SST: min={self.df['SST'].min():.2f}, max={self.df['SST'].max():.2f}")
        print(f"RAD: min={self.df['RAD'].min():.2f}, max={self.df['RAD'].max():.2f}")
        
        # Urutkan data
        self.df = self.df.sort_values(['Location', 'Date']).reset_index(drop=True)
        print("Data diurutkan berdasarkan Location dan Date")
        
    def daily_resampling(self):
        """Tahap 4: Resampling Harian"""
        print("\n🔄 Tahap 4: Resampling Harian")
        
        full_dfs = []
        for loc in self.df['Location'].unique():
            loc_df = self.df[self.df['Location'] == loc]
            
            # Buat grid tanggal lengkap
            date_range = pd.date_range(
                start=loc_df['Date'].min(),
                end=self.loc_ranges[loc],
                freq='D'
            )
            
            print(f"{loc}: {len(date_range)} hari dari {date_range[0].date()} hingga {date_range[-1].date()}")
            
            # Buat DataFrame lengkap
            full_df = pd.DataFrame({'Date': date_range, 'Location': loc})
            full_df = full_df.merge(loc_df, how='left', on=['Date', 'Location'])
            full_dfs.append(full_df)
        
        self.df = pd.concat(full_dfs, ignore_index=True)
        
        print(f"Total baris setelah resampling: {len(self.df)}")
        print("Missing values setelah resampling:")
        print(self.df.isnull().sum())
        
    def hybrid_impute(self, series, col_name):
        """Imputasi Hybrid menggunakan STL"""
        
        # Langkah 1: Interpolasi linear untuk gap kecil (1-3 hari)
        series_filled = series.interpolate(method='linear', limit=3)
        
        if series_filled.isna().sum() == 0:
            return series_filled
            
        # Langkah 2: STL untuk gap menengah (4-30 hari)
        try:
            # Perlu data minimal 2 tahun untuk STL
            if len(series_filled.dropna()) > 730:  # 2 tahun
                stl = STL(series_filled.dropna(), period=365, robust=True)
                result = stl.fit()
                
                # Rekonstruksi komponen
                reconstructed = result.trend + result.seasonal
                
                # Extend rekonstruksi ke seluruh series
                full_index = series.index
                recon_series = pd.Series(index=full_index, dtype=float)
                recon_series.loc[reconstructed.index] = reconstructed
                
                # Interpolasi rekonstruksi untuk gap
                recon_series = recon_series.interpolate(method='linear')
                
                # Isi gap dengan komponen rekonstruksi
                na_mask = series_filled.isna()
                series_filled.loc[na_mask] = recon_series.loc[na_mask]
                
            else:
                print(f"  Data {col_name} tidak cukup untuk STL, gunakan seasonal mean")
                # Fallback: imputasi mean musiman
                day_of_year = series.index.map(lambda x: x.dayofyear)
                series_with_doy = pd.DataFrame({'value': series, 'doy': day_of_year})
                seasonal_means = series_with_doy.groupby('doy')['value'].transform('mean')
                series_filled = series_filled.fillna(seasonal_means)
                
        except Exception as e:
            print(f"  STL gagal untuk {col_name}: {e}")
            # Fallback: forward fill + backward fill
            series_filled = series_filled.fillna(method='ffill').fillna(method='bfill')
        
        # Langkah 3: Final fillna untuk sisa gap
        if series_filled.isna().sum() > 0:
            # Gunakan median sebagai last resort
            median_val = series_filled.median()
            series_filled = series_filled.fillna(median_val)
            
        return series_filled
    
    def advanced_imputation(self):
        """Tahap 5: Imputasi Canggih (Strategi Hybrid)"""
        print("\n🧩 Tahap 5: Imputasi Canggih (Strategi Hybrid)")
        
        # Terapkan ke setiap lokasi dan variabel
        for loc in self.df['Location'].unique():
            print(f"\nProcessing lokasi: {loc}")
            loc_mask = self.df['Location'] == loc
            loc_data = self.df[loc_mask].copy()
            
            for col in ['SST', 'RAD']:
                missing_before = loc_data[col].isna().sum()
                print(f"  {col}: {missing_before} missing values")
                
                if missing_before > 0:
                    # Set tanggal sebagai index untuk time series processing
                    ts_data = loc_data.set_index('Date')[col]
                    filled_data = self.hybrid_impute(ts_data, f"{loc}_{col}")
                    
                    # Update data
                    self.df.loc[loc_mask, col] = filled_data.values
                    
                    missing_after = filled_data.isna().sum()
                    print(f"  {col}: {missing_before} → {missing_after} missing values")
        
        print("\nMissing values setelah imputasi:")
        print(self.df.isnull().sum())
        
    def feature_engineering(self):
        """Tahap 6: Rekayasa Fitur & Gabungan"""
        print("\n📊 Tahap 6: Rekayasa Fitur & Gabungan")
        
        # Tambahkan fitur temporal
        self.df['DayOfYear'] = self.df['Date'].dt.dayofyear
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Day'] = self.df['Date'].dt.day
        
        print("Fitur temporal ditambahkan: DayOfYear, Year, Month, Day")
        
        # Gabungkan semua lokasi (sudah dalam satu DataFrame)
        self.processed_df = self.df.sort_values(['Location', 'Date']).reset_index(drop=True)
        
        print(f"Dataset final: {len(self.processed_df)} baris")
        print("Kolom:", list(self.processed_df.columns))
        
    def save_processed_data(self, output_path):
        """Tahap 7: Penyimpanan"""
        print(f"\n💾 Tahap 7: Penyimpanan ke {output_path}")
        
        self.processed_df.to_csv(output_path, index=False)
        print(f"Data berhasil disimpan: {len(self.processed_df)} baris")
        
        # Tampilkan info final
        print("\nInfo dataset final:")
        print(self.processed_df.info())
        
    def validation_summary(self):
        """Validasi Pasca-Imputasi"""
        print("\n" + "="*50)
        print("📊 VALIDASI PASCA-IMPUTASI")
        print("="*50)
        
        # 1. Statistik Deskriptif
        print("\n1. Statistik Deskriptif per Lokasi:")
        stats = self.processed_df.groupby('Location')[['SST', 'RAD']].describe()
        print(stats.round(2))
        
        # 2. Missing values check
        print("\n2. Missing Values Check:")
        missing = self.processed_df.isnull().sum()
        print(missing)
        
        # 3. Range check
        print("\n3. Range Validation:")
        for col in ['SST', 'RAD']:
            print(f"{col}: min={self.processed_df[col].min():.2f}, max={self.processed_df[col].max():.2f}")
        
        # 4. Temporal coverage
        print("\n4. Temporal Coverage per Lokasi:")
        for loc in self.processed_df['Location'].unique():
            loc_data = self.processed_df[self.processed_df['Location'] == loc]
            start_date = loc_data['Date'].min().date()
            end_date = loc_data['Date'].max().date()
            days = (loc_data['Date'].max() - loc_data['Date'].min()).days + 1
            print(f"{loc}: {start_date} hingga {end_date} ({days} hari)")
    
    def plot_validation_sample(self, n_samples=2):
        """Plot sample data untuk validasi visual"""
        print(f"\n5. Sample Visual Validation ({n_samples} sample per lokasi):")
        
        fig, axes = plt.subplots(len(self.processed_df['Location'].unique()), 2, 
                                figsize=(15, 4*len(self.processed_df['Location'].unique())))
        
        for i, loc in enumerate(self.processed_df['Location'].unique()):
            loc_df = self.processed_df[self.processed_df['Location'] == loc]
            
            # Sample random periods
            sample_dates = np.random.choice(
                loc_df['Date'].dt.to_pydatetime(), 
                min(n_samples, len(loc_df)//100), 
                replace=False
            )
            
            for j, col in enumerate(['SST', 'RAD']):
                ax = axes[i, j] if len(self.processed_df['Location'].unique()) > 1 else axes[j]
                
                # Plot full time series
                ax.plot(loc_df['Date'], loc_df[col], alpha=0.7, linewidth=0.5)
                
                # Highlight sample periods
                for date in sample_dates:
                    start = pd.to_datetime(date) - pd.Timedelta(days=15)
                    end = pd.to_datetime(date) + pd.Timedelta(days=15)
                    segment = loc_df[(loc_df['Date'] >= start) & (loc_df['Date'] <= end)]
                    ax.plot(segment['Date'], segment[col], 'r-', linewidth=2, alpha=0.8)
                
                ax.set_title(f"{loc} - {col}")
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/validation_sample.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_annual_trends(self):
        """Visualisasi Annual Trends untuk SST dan RAD"""
        print("\n📈 Annual Trends Analysis")
        
        # Hitung annual means
        annual_data = self.processed_df.groupby(['Location', 'Year'])[['SST', 'RAD']].mean().reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # SST Annual Trends
        for loc in annual_data['Location'].unique():
            loc_data = annual_data[annual_data['Location'] == loc]
            axes[0, 0].plot(loc_data['Year'], loc_data['SST'], marker='o', linewidth=2, label=loc)
        
        axes[0, 0].set_title('Annual SST Trends by Location', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('SST (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RAD Annual Trends
        for loc in annual_data['Location'].unique():
            loc_data = annual_data[annual_data['Location'] == loc]
            axes[0, 1].plot(loc_data['Year'], loc_data['RAD'], marker='s', linewidth=2, label=loc)
        
        axes[0, 1].set_title('Annual RAD Trends by Location', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('RAD (W/m²)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Combined annual anomalies (normalized)
        for loc in annual_data['Location'].unique():
            loc_data = annual_data[annual_data['Location'] == loc]
            sst_norm = (loc_data['SST'] - loc_data['SST'].mean()) / loc_data['SST'].std()
            rad_norm = (loc_data['RAD'] - loc_data['RAD'].mean()) / loc_data['RAD'].std()
            
            axes[1, 0].plot(loc_data['Year'], sst_norm, marker='o', linewidth=2, label=f'{loc} SST')
            axes[1, 1].plot(loc_data['Year'], rad_norm, marker='s', linewidth=2, label=f'{loc} RAD')
        
        axes[1, 0].set_title('Normalized SST Anomalies', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Normalized Anomaly')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Normalized RAD Anomalies', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Normalized Anomaly')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/annual_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print annual statistics
        print("\nAnnual Statistics Summary:")
        for loc in annual_data['Location'].unique():
            loc_data = annual_data[annual_data['Location'] == loc]
            sst_trend = np.polyfit(loc_data['Year'], loc_data['SST'], 1)[0]
            rad_trend = np.polyfit(loc_data['Year'], loc_data['RAD'], 1)[0]
            
            print(f"\n{loc}:")
            print(f"  SST trend: {sst_trend:.4f} °C/year")
            print(f"  RAD trend: {rad_trend:.4f} W/m²/year")
            print(f"  SST range: {loc_data['SST'].min():.2f} - {loc_data['SST'].max():.2f} °C")
            print(f"  RAD range: {loc_data['RAD'].min():.1f} - {loc_data['RAD'].max():.1f} W/m²")
    
    def plot_monthly_trends(self):
        """Visualisasi Monthly Trends untuk SST dan RAD"""
        print("\n📅 Monthly Trends Analysis")
        
        # Hitung monthly climatology
        monthly_data = self.processed_df.groupby(['Location', 'Month'])[['SST', 'RAD']].agg(['mean', 'std']).reset_index()
        monthly_data.columns = ['Location', 'Month', 'SST_mean', 'SST_std', 'RAD_mean', 'RAD_std']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # SST Monthly Climatology
        for loc in monthly_data['Location'].unique():
            loc_data = monthly_data[monthly_data['Location'] == loc]
            axes[0, 0].plot(loc_data['Month'], loc_data['SST_mean'], marker='o', linewidth=2, label=loc)
            axes[0, 0].fill_between(loc_data['Month'], 
                                   loc_data['SST_mean'] - loc_data['SST_std'],
                                   loc_data['SST_mean'] + loc_data['SST_std'],
                                   alpha=0.2)
        
        axes[0, 0].set_title('Monthly SST Climatology ± 1σ', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('SST (°C)')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RAD Monthly Climatology
        for loc in monthly_data['Location'].unique():
            loc_data = monthly_data[monthly_data['Location'] == loc]
            axes[0, 1].plot(loc_data['Month'], loc_data['RAD_mean'], marker='s', linewidth=2, label=loc)
            axes[0, 1].fill_between(loc_data['Month'], 
                                   loc_data['RAD_mean'] - loc_data['RAD_std'],
                                   loc_data['RAD_mean'] + loc_data['RAD_std'],
                                   alpha=0.2)
        
        axes[0, 1].set_title('Monthly RAD Climatology ± 1σ', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('RAD (W/m²)')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SST Seasonal Variability (coefficient of variation)
        for loc in monthly_data['Location'].unique():
            loc_data = monthly_data[monthly_data['Location'] == loc]
            cv_sst = loc_data['SST_std'] / loc_data['SST_mean'] * 100
            axes[1, 0].plot(loc_data['Month'], cv_sst, marker='o', linewidth=2, label=loc)
        
        axes[1, 0].set_title('SST Seasonal Variability (CV%)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Coefficient of Variation (%)')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # RAD Seasonal Variability
        for loc in monthly_data['Location'].unique():
            loc_data = monthly_data[monthly_data['Location'] == loc]
            cv_rad = loc_data['RAD_std'] / loc_data['RAD_mean'] * 100
            axes[1, 1].plot(loc_data['Month'], cv_rad, marker='s', linewidth=2, label=loc)
        
        axes[1, 1].set_title('RAD Seasonal Variability (CV%)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Coefficient of Variation (%)')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/monthly_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print monthly statistics
        print("\nMonthly Climatology Summary:")
        for loc in monthly_data['Location'].unique():
            loc_data = monthly_data[monthly_data['Location'] == loc]
            
            sst_max_month = loc_data.loc[loc_data['SST_mean'].idxmax(), 'Month']
            sst_min_month = loc_data.loc[loc_data['SST_mean'].idxmin(), 'Month']
            rad_max_month = loc_data.loc[loc_data['RAD_mean'].idxmax(), 'Month']
            rad_min_month = loc_data.loc[loc_data['RAD_mean'].idxmin(), 'Month']
            
            print(f"\n{loc}:")
            print(f"  SST: Max in month {sst_max_month} ({loc_data['SST_mean'].max():.2f}°C), Min in month {sst_min_month} ({loc_data['SST_mean'].min():.2f}°C)")
            print(f"  RAD: Max in month {rad_max_month} ({loc_data['RAD_mean'].max():.1f} W/m²), Min in month {rad_min_month} ({loc_data['RAD_mean'].min():.1f} W/m²)")
            print(f"  SST seasonal amplitude: {loc_data['SST_mean'].max() - loc_data['SST_mean'].min():.2f}°C")
            print(f"  RAD seasonal amplitude: {loc_data['RAD_mean'].max() - loc_data['RAD_mean'].min():.1f} W/m²")
    
    def plot_correlation_matrix(self):
        """Visualisasi Correlation Matrix"""
        print("\n🔗 Correlation Analysis")
        
        # Prepare data untuk korelasi
        corr_data = self.processed_df[['SST', 'RAD', 'DayOfYear', 'Year']].copy()
        
        # Hitung korelasi per lokasi
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        locations = self.processed_df['Location'].unique()
        
        for i, loc in enumerate(locations):
            if i >= 3:  # Maximum 3 locations
                break
            
            loc_data = self.processed_df[self.processed_df['Location'] == loc]
            loc_corr_data = loc_data[['SST', 'RAD', 'DayOfYear', 'Year']].corr()
            
            # Plot correlation heatmap
            row, col = i // 2, i % 2
            sns.heatmap(loc_corr_data, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=axes[row, col], cbar_kws={'shrink': 0.8})
            axes[row, col].set_title(f'Correlation Matrix - {loc}', fontsize=12, fontweight='bold')
        
        # Overall correlation (all locations combined)
        if len(locations) <= 3:
            overall_corr = corr_data.corr()
            sns.heatmap(overall_corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
            axes[1, 1].set_title('Overall Correlation Matrix', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Cross-location correlation
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # SST cross-location correlation
        sst_pivot = self.processed_df.pivot_table(index='Date', columns='Location', values='SST')
        sst_cross_corr = sst_pivot.corr()
        
        sns.heatmap(sst_cross_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('SST Cross-Location Correlation', fontsize=12, fontweight='bold')
        
        # RAD cross-location correlation
        rad_pivot = self.processed_df.pivot_table(index='Date', columns='Location', values='RAD')
        rad_cross_corr = rad_pivot.corr()
        
        sns.heatmap(rad_cross_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('RAD Cross-Location Correlation', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cross_location_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print correlation insights
        print("\nCorrelation Insights:")
        overall_corr = corr_data.corr()
        sst_rad_corr = overall_corr.loc['SST', 'RAD']
        print(f"Overall SST-RAD correlation: {sst_rad_corr:.3f}")
        
        print("\nCross-location correlations:")
        print("SST correlations between locations:")
        print(sst_cross_corr.round(3))
        print("\nRAD correlations between locations:")
        print(rad_cross_corr.round(3))
    
    def plot_outlier_analysis(self):
        """Visualisasi Boxplot untuk Outlier Detection"""
        print("\n🎯 Outlier Analysis")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Boxplot per Location
        locations = self.processed_df['Location'].unique()
        
        # SST boxplot by location
        sst_data_by_loc = [self.processed_df[self.processed_df['Location'] == loc]['SST'] for loc in locations]
        bp1 = axes[0, 0].boxplot(sst_data_by_loc, labels=locations, patch_artist=True)
        axes[0, 0].set_title('SST Distribution by Location', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('SST (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp1['boxes'], colors[:len(locations)]):
            patch.set_facecolor(color)
        
        # RAD boxplot by location
        rad_data_by_loc = [self.processed_df[self.processed_df['Location'] == loc]['RAD'] for loc in locations]
        bp2 = axes[0, 1].boxplot(rad_data_by_loc, labels=locations, patch_artist=True)
        axes[0, 1].set_title('RAD Distribution by Location', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('RAD (W/m²)')
        axes[0, 1].grid(True, alpha=0.3)
        
        for patch, color in zip(bp2['boxes'], colors[:len(locations)]):
            patch.set_facecolor(color)
        
        # 2. Seasonal boxplot (quarterly)
        self.processed_df['Quarter'] = self.processed_df['Month'].map({1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 
                                                                      7:3, 8:3, 9:3, 10:4, 11:4, 12:4})
        
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        sst_data_by_quarter = [self.processed_df[self.processed_df['Quarter'] == i]['SST'] for i in range(1, 5)]
        bp3 = axes[1, 0].boxplot(sst_data_by_quarter, labels=quarters, patch_artist=True)
        axes[1, 0].set_title('SST Seasonal Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('SST (°C)')
        axes[1, 0].grid(True, alpha=0.3)
        
        season_colors = ['lightblue', 'lightcoral', 'orange', 'lightgreen']
        for patch, color in zip(bp3['boxes'], season_colors):
            patch.set_facecolor(color)
        
        rad_data_by_quarter = [self.processed_df[self.processed_df['Quarter'] == i]['RAD'] for i in range(1, 5)]
        bp4 = axes[1, 1].boxplot(rad_data_by_quarter, labels=quarters, patch_artist=True)
        axes[1, 1].set_title('RAD Seasonal Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('RAD (W/m²)')
        axes[1, 1].grid(True, alpha=0.3)
        
        for patch, color in zip(bp4['boxes'], season_colors):
            patch.set_facecolor(color)
        
        # 3. Year-wise boxplot (every 5 years)
        year_groups = ['2005-2009', '2010-2014', '2015-2020']
        year_conditions = [
            (self.processed_df['Year'] >= 2005) & (self.processed_df['Year'] <= 2009),
            (self.processed_df['Year'] >= 2010) & (self.processed_df['Year'] <= 2014),
            (self.processed_df['Year'] >= 2015) & (self.processed_df['Year'] <= 2020)
        ]
        
        sst_data_by_period = [self.processed_df[condition]['SST'] for condition in year_conditions]
        bp5 = axes[2, 0].boxplot(sst_data_by_period, labels=year_groups, patch_artist=True)
        axes[2, 0].set_title('SST Distribution by Period', fontsize=14, fontweight='bold')
        axes[2, 0].set_ylabel('SST (°C)')
        axes[2, 0].grid(True, alpha=0.3)
        
        period_colors = ['lightsteelblue', 'lightcoral', 'lightgoldenrodyellow']
        for patch, color in zip(bp5['boxes'], period_colors):
            patch.set_facecolor(color)
        
        rad_data_by_period = [self.processed_df[condition]['RAD'] for condition in year_conditions]
        bp6 = axes[2, 1].boxplot(rad_data_by_period, labels=year_groups, patch_artist=True)
        axes[2, 1].set_title('RAD Distribution by Period', fontsize=14, fontweight='bold')
        axes[2, 1].set_ylabel('RAD (W/m²)')
        axes[2, 1].grid(True, alpha=0.3)
        
        for patch, color in zip(bp6['boxes'], period_colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Outlier detection statistics
        print("\nOutlier Detection Results:")
        
        for loc in locations:
            loc_data = self.processed_df[self.processed_df['Location'] == loc]
            
            # IQR method for outlier detection
            for var in ['SST', 'RAD']:
                Q1 = loc_data[var].quantile(0.25)
                Q3 = loc_data[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_fence = Q1 - 1.5 * IQR
                upper_fence = Q3 + 1.5 * IQR
                
                outliers = loc_data[(loc_data[var] < lower_fence) | (loc_data[var] > upper_fence)]
                outlier_percentage = (len(outliers) / len(loc_data)) * 100
                
                print(f"\n{loc} - {var}:")
                print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
                print(f"  Outlier range: < {lower_fence:.2f} or > {upper_fence:.2f}")
                print(f"  Outliers detected: {len(outliers)} ({outlier_percentage:.2f}%)")
                
                if len(outliers) > 0:
                    print(f"  Outlier range: {outliers[var].min():.2f} to {outliers[var].max():.2f}")
    
    def run_comprehensive_analysis(self):
        """Menjalankan semua analisis visual"""
        print("\n" + "="*60)
        print("🔍 COMPREHENSIVE VISUAL ANALYSIS")
        print("="*60)
    
        try:
            self.plot_annual_trends()
        except Exception as e:
            print(f"Error in annual trends: {e}")
        try:
            self.plot_monthly_trends()
        except Exception as e:
            print(f"Error in monthly trends: {e}")
        try:
            self.plot_correlation_matrix()
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
        try:
            self.plot_outlier_analysis()
        except Exception as e:
            print(f"Error in outlier analysis: {e}")
    
    print(f"\n✅ Visual analysis completed!")
    
    def run_full_pipeline(self, output_path):
        """Menjalankan seluruh pipeline preprocessing"""
        try:
            # Jalankan semua tahap
            self.load_data()
            self.handle_location_specific_ranges()
            self.initial_cleaning()
            self.daily_resampling()
            self.advanced_imputation()
            self.feature_engineering()
            self.save_processed_data(output_path)
            
            # Validasi
            self.validation_summary()
            
            print("\n✅ Preprocessing berhasil diselesaikan!")
            return self.processed_df
            
        except Exception as e:
            print(f"\n❌ Error dalam preprocessing: {e}")
            raise e

def main():
    """Main function untuk menjalankan preprocessing"""
    print("🌊 BUOYS DATA PREPROCESSING")
    print("="*50)
    
    # Inisialisasi preprocessor
    preprocessor = BuoysPreprocessor('Buoys_Data_All.csv', output_dir='buoys_plots')

    
    # Jalankan full pipeline
    processed_data = preprocessor.run_full_pipeline('buoys_preprocessed.csv')
    
    # Optional: Plot validasi
    try:
        preprocessor.plot_validation_sample(n_samples=1)
    except Exception as e:
        print(f"Warning: Plotting gagal - {e}")
    
    # Jalankan comprehensive analysis
    try:
        print("\n" + "="*50)
        print("🔍 Starting Comprehensive Visual Analysis...")
        print("="*50)
        preprocessor.run_comprehensive_analysis()
    except Exception as e:
        print(f"Warning: Visual analysis gagal - {e}")
    
    print(f"\n🎉 Preprocessing selesai! Dataset tersimpan sebagai 'buoys_preprocessed.csv'")
    print(f"Shape dataset: {processed_data.shape}")
    
    return processed_data

if __name__ == "__main__":
    processed_data = main()