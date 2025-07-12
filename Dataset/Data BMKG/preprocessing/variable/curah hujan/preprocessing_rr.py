import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RainfallPreprocessor:
    def __init__(self, data):
        """
        Inisialisasi preprocessor untuk data curah hujan
        
        Parameters:
        data: DataFrame dengan kolom Date, RR, dan kolom lainnya
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.processed_data = None
        self.missing_stats = {}
        self.outliers = {}
        
    def load_and_prepare_data(self):
        """
        Load data dan persiapan awal
        """
        # Konversi Date ke datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Buat kolom tambahan untuk analisis
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        self.data['DayOfYear'] = self.data['Date'].dt.dayofyear
        
        print(f"Data loaded: {len(self.data)} records from {self.data['Date'].min()} to {self.data['Date'].max()}")
        return self.data
    
    def analyze_missing_values(self):
        """
        Analisis mendalam missing values dan nilai khusus
        """
        print("=== ANALISIS MISSING VALUES DAN NILAI KHUSUS ===")
        
        # Identifikasi nilai 8888 (tidak terukur)
        missing_8888 = (self.data['RR'] == 8888).sum()
        missing_nan = self.data['RR'].isna().sum()
        zero_values = (self.data['RR'] == 0).sum()
        total_records = len(self.data)
        
        print(f"Total records: {total_records}")
        print(f"Nilai 8888 (tidak terukur): {missing_8888} ({missing_8888/total_records*100:.1f}%)")
        print(f"Missing/NaN values: {missing_nan} ({missing_nan/total_records*100:.1f}%)")
        print(f"Zero values (tidak hujan): {zero_values} ({zero_values/total_records*100:.1f}%)")
        
        # Simpan statistik missing values
        self.missing_stats = {
            'total_records': total_records,
            'missing_8888': missing_8888,
            'missing_nan': missing_nan,
            'zero_values': zero_values,
            'missing_percentage': (missing_8888 + missing_nan) / total_records * 100
        }
        
        # Analisis missing values per tahun
        yearly_missing = self.data.groupby('Year').agg({
            'RR': [
                lambda x: (x == 8888).sum(),
                lambda x: x.isna().sum(),
                lambda x: (x == 0).sum(),
                'count'
            ]
        }).round(2)
        
        yearly_missing.columns = ['Missing_8888', 'Missing_NaN', 'Zero_Values', 'Total_Records']
        yearly_missing['Missing_Percentage'] = (yearly_missing['Missing_8888'] + yearly_missing['Missing_NaN']) / yearly_missing['Total_Records'] * 100
        
        print("\n=== MISSING VALUES PER TAHUN ===")
        print(yearly_missing)
        
        return yearly_missing
    def clean_rainfall_data(self):
        """
        Cleaning data curah hujan dengan handling khusus dan analisis gap
        """
        print("\n=== CLEANING DATA CURAH HUJAN ===")
        
        # Backup data asli
        self.data['RR_original'] = self.data['RR'].copy()
        
        # Step 1: Konversi 9999 ke NaN (tidak ada data)
        self.data['RR'] = self.data['RR'].replace(9999, np.nan)
        
        # Step 2: Analisis dan estimasi nilai 8888 berdasarkan parameter meteorologi
        mask_8888 = self.data['RR'] == 8888
        count_8888 = mask_8888.sum()
        
        print(f"Data dengan kode 8888 (tidak terukur): {count_8888} records")
        
        if count_8888 > 0:
            print("\n=== ESTIMASI DATA 8888 ===")
            
            # Buat dataset referensi dari data yang valid (bukan 8888 atau NaN)
            valid_data = self.data[(self.data['RR'] != 8888) & (self.data['RR'].notna())].copy()
            
            if len(valid_data) > 0:
                print(f"Menggunakan {len(valid_data)} data valid sebagai referensi")
                
                # Estimasi untuk setiap record dengan 8888
                estimated_values = []
                estimation_methods = []
                
                for idx in self.data[mask_8888].index:
                    row = self.data.loc[idx]
                    
                    # Ekstrak parameter meteorologi
                    rh_avg = row['RH_AVG'] if not pd.isna(row['RH_AVG']) else 75  # default jika NaN
                    ss = row['SS'] if not pd.isna(row['SS']) else 5  # default jika NaN
                    tn = row['TN'] if not pd.isna(row['TN']) else 25  # default jika NaN
                    
                    # Logika estimasi berdasarkan kombinasi parameter
                    if rh_avg >= 80 and ss <= 3 and tn <= 24:
                        # Kondisi sangat mendukung hujan
                        similar_conditions = valid_data[
                            (valid_data['RH_AVG'] >= 80) & 
                            (valid_data['SS'] <= 3) & 
                            (valid_data['TN'] <= 24)
                        ]
                        
                        if len(similar_conditions) >= 3:
                            # Gunakan median dari kondisi serupa
                            estimated_rr = similar_conditions['RR'].median()
                            method = "similar_wet_conditions"
                        else:
                            # Fallback: rata-rata hari hujan
                            rainy_days = valid_data[valid_data['RR'] > 0]
                            estimated_rr = rainy_days['RR'].median() if len(rainy_days) > 0 else 10.0
                            method = "median_rainy_days"
                    
                    elif rh_avg >= 70 and ss <= 5:
                        # Kondisi sedang mendukung hujan ringan
                        similar_conditions = valid_data[
                            (valid_data['RH_AVG'] >= 70) & 
                            (valid_data['RH_AVG'] < 80) & 
                            (valid_data['SS'] <= 5)
                        ]
                        
                        if len(similar_conditions) >= 3:
                            estimated_rr = similar_conditions['RR'].median()
                            method = "similar_moderate_conditions"
                        else:
                            # Estimasi hujan ringan
                            estimated_rr = 2.0
                            method = "light_rain_estimate"
                    
                    elif rh_avg <= 70 and ss >= 5:
                        # Kondisi cerah, kemungkinan tidak hujan
                        estimated_rr = 0.0
                        method = "clear_weather"
                    
                    else:
                        # Kondisi tidak jelas, gunakan interpolasi temporal
                        # Cari nilai sebelum dan sesudah
                        prev_idx = idx - 1
                        next_idx = idx + 1
                        
                        prev_val = None
                        next_val = None
                        
                        if prev_idx in self.data.index and self.data.loc[prev_idx, 'RR'] not in [8888, np.nan]:
                            prev_val = self.data.loc[prev_idx, 'RR']
                        
                        if next_idx in self.data.index and self.data.loc[next_idx, 'RR'] not in [8888, np.nan]:
                            next_val = self.data.loc[next_idx, 'RR']
                        
                        if prev_val is not None and next_val is not None:
                            estimated_rr = (prev_val + next_val) / 2
                            method = "temporal_interpolation"
                        elif prev_val is not None:
                            estimated_rr = prev_val
                            method = "previous_day"
                        elif next_val is not None:
                            estimated_rr = next_val
                            method = "next_day"
                        else:
                            # Fallback ke rata-rata musiman
                            month = row['Month']
                            monthly_data = valid_data[valid_data['Month'] == month]
                            estimated_rr = monthly_data['RR'].median() if len(monthly_data) > 0 else 5.0
                            method = "monthly_median"
                    
                    estimated_values.append(estimated_rr)
                    estimation_methods.append(method)
                    
                    print(f"  {row['Date'].strftime('%Y-%m-%d')}: RH={rh_avg:.1f}%, SS={ss:.1f}h, TN={tn:.1f}°C → {estimated_rr:.1f}mm ({method})")
                
                # Terapkan estimasi ke dataset
                self.data.loc[mask_8888, 'RR'] = estimated_values
                self.data.loc[mask_8888, 'RR_estimation_method'] = estimation_methods
                
                print(f"\nRingkasan estimasi:")
                method_counts = pd.Series(estimation_methods).value_counts()
                for method, count in method_counts.items():
                    print(f"  {method}: {count} records")
            
            else:
                print("Tidak ada data valid untuk referensi, konversi 8888 → NaN")
                self.data['RR'] = self.data['RR'].replace(8888, np.nan)
        
        # Step 3: Analisis gap patterns untuk strategi imputasi (untuk data yang masih NaN)
        self.data['is_missing'] = self.data['RR'].isna()
        
        # Identifikasi consecutive missing periods
        self.data['gap_group'] = (self.data['is_missing'] != self.data['is_missing'].shift()).cumsum()
        gap_analysis = self.data.groupby('gap_group').agg({
            'is_missing': ['first', 'count'],
            'Date': ['first', 'last']
        }).reset_index()
        
        # Flatten column names
        gap_analysis.columns = ['gap_group', 'is_missing', 'gap_length', 'start_date', 'end_date']
        missing_gaps = gap_analysis[gap_analysis['is_missing'] == True].copy()
        
        print(f"\nTotal missing values (setelah estimasi 8888): {self.data['RR'].isna().sum()}")
        print(f"Jumlah gap periods: {len(missing_gaps)}")
        
        if len(missing_gaps) > 0:
            print("\nAnalisis Gap Patterns:")
            print(f"  • Gap terpendek: {missing_gaps['gap_length'].min()} hari")
            print(f"  • Gap terpanjang: {missing_gaps['gap_length'].max()} hari")
            print(f"  • Rata-rata gap: {missing_gaps['gap_length'].mean():.1f} hari")
            
            # Kategorisasi gap
            short_gaps = missing_gaps[missing_gaps['gap_length'] <= 3]
            medium_gaps = missing_gaps[(missing_gaps['gap_length'] > 3) & (missing_gaps['gap_length'] <= 7)]
            long_gaps = missing_gaps[(missing_gaps['gap_length'] > 7) & (missing_gaps['gap_length'] <= 30)]
            extreme_gaps = missing_gaps[missing_gaps['gap_length'] > 30]
            
            print(f"  • Gap pendek (≤3 hari): {len(short_gaps)} gaps")
            print(f"  • Gap sedang (4-7 hari): {len(medium_gaps)} gaps")
            print(f"  • Gap panjang (8-30 hari): {len(long_gaps)} gaps")
            print(f"  • Gap ekstrem (>30 hari): {len(extreme_gaps)} gaps")
            
            # Simpan gap analysis untuk seasonal_imputation
            self.gap_analysis = missing_gaps
        
        # Step 4: Identifikasi outlier ekstrem (> 200mm/hari sangat jarang di Indonesia)
        outlier_threshold = 200
        outliers = self.data[self.data['RR'] > outlier_threshold]
        
        print(f"\nOutlier ekstrem (RR > {outlier_threshold}mm): {len(outliers)} records")
        if len(outliers) > 0:
            print("Outlier dates:")
            for idx, row in outliers.iterrows():
                print(f"  {row['Date'].strftime('%Y-%m-%d')}: {row['RR']}mm")
        
        # Simpan outliers untuk analisis
        self.outliers = outliers
        
        # Step 5: Statistik dasar setelah cleaning
        valid_data = self.data['RR'].dropna()
        if len(valid_data) > 0:
            print(f"\nStatistik RR setelah cleaning (n={len(valid_data)}):")
            print(f"  Mean: {valid_data.mean():.2f}mm")
            print(f"  Median: {valid_data.median():.2f}mm")
            print(f"  Std: {valid_data.std():.2f}mm")
            print(f"  Min: {valid_data.min():.2f}mm")
            print(f"  Max: {valid_data.max():.2f}mm")
            print(f"  Zero days: {(valid_data == 0).sum()} ({(valid_data == 0).sum()/len(valid_data)*100:.1f}%)")
            print(f"  Rainy days: {(valid_data > 0).sum()} ({(valid_data > 0).sum()/len(valid_data)*100:.1f}%)")
            
            # Statistik khusus untuk data yang diestimasi
            if 'RR_estimation_method' in self.data.columns:
                estimated_data = self.data[self.data['RR_estimation_method'].notna()]
                if len(estimated_data) > 0:
                    print(f"\nData yang diestimasi dari 8888: {len(estimated_data)} records")
                    print(f"  Mean estimated: {estimated_data['RR'].mean():.2f}mm")
                    print(f"  Median estimated: {estimated_data['RR'].median():.2f}mm")
    
    def seasonal_imputation(self):
        """
        Imputasi hybrid berdasarkan panjang gap dan pola musiman
        """
        print("\n=== IMPUTASI HYBRID BERDASARKAN GAP PATTERN ===")
        
        if not hasattr(self, 'gap_analysis'):
            print("Tidak ada gap analysis, skip imputasi")
            return
        
        # Inisialisasi kolom hasil imputasi
        self.data['RR_imputed'] = self.data['RR'].copy()
        self.data['imputation_method'] = 'original'
        
        # Hitung seasonal statistics untuk berbagai metode
        seasonal_stats = {}
        
        # Monthly statistics
        monthly_stats = self.data.groupby('Month')['RR'].agg(['mean', 'median', 'std', 'count']).reset_index()
        monthly_stats['dry_day_prob'] = self.data.groupby('Month').apply(
            lambda x: (x['RR'] == 0).sum() / x['RR'].notna().sum()
        ).values
        
        # Seasonal decomposition untuk gap panjang
        try:
            # Buat monthly aggregation untuk decomposition
            monthly_data = self.data.groupby(['Year', 'Month'])['RR'].mean().reset_index()
            monthly_data['date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
            monthly_ts = monthly_data.set_index('date')['RR'].dropna()
            
            if len(monthly_ts) >= 24:  # Minimal 2 tahun
                from statsmodels.tsa.seasonal import seasonal_decompose
                decomposition = seasonal_decompose(monthly_ts, model='additive', period=12)
                
                # Interpolate missing values dalam decomposition
                trend_interp = decomposition.trend.interpolate(method='linear')
                seasonal_interp = decomposition.seasonal.interpolate(method='linear')
                
                # Mapping back to daily data
                seasonal_pattern = {}
                for idx, row in monthly_data.iterrows():
                    date_key = row['date']
                    if date_key in seasonal_interp.index:
                        seasonal_pattern[row['Month']] = seasonal_interp[date_key]
                
                seasonal_stats['decomposition'] = {
                    'trend': trend_interp,
                    'seasonal': seasonal_pattern,
                    'available': True
                }
            else:
                seasonal_stats['decomposition'] = {'available': False}
                
        except Exception as e:
            print(f"Seasonal decomposition failed: {e}")
            seasonal_stats['decomposition'] = {'available': False}
        
        # Iterasi setiap gap untuk imputasi
        total_imputed = 0
        
        for _, gap in self.gap_analysis.iterrows():
            gap_length = gap['gap_length']
            gap_group = gap['gap_group']
            
            # Identifikasi indices untuk gap ini
            gap_indices = self.data[self.data['gap_group'] == gap_group].index
            
            print(f"\nProcessing gap: {gap['start_date'].strftime('%Y-%m-%d')} to {gap['end_date'].strftime('%Y-%m-%d')} ({gap_length} hari)")
            
            if gap_length <= 3:
                # METODE 1: Linear interpolation untuk gap pendek
                print("  → Menggunakan Linear Interpolation")
                
                # Log transformation untuk interpolation
                self.data['RR_log_temp'] = np.log1p(self.data['RR'])
                self.data['RR_log_temp'].iloc[gap_indices] = np.nan
                
                # Interpolasi linear
                interpolated = self.data['RR_log_temp'].interpolate(method='linear', limit_direction='both')
                
                # Transform back dan assign
                imputed_values = np.expm1(interpolated.iloc[gap_indices])
                self.data.loc[gap_indices, 'RR_imputed'] = imputed_values
                self.data.loc[gap_indices, 'imputation_method'] = 'linear_interpolation'
                
                # Cleanup temporary column
                self.data.drop('RR_log_temp', axis=1, inplace=True)
                
            elif gap_length <= 7:
                # METODE 2: Seasonal-aware interpolation untuk gap sedang
                print("  → Menggunakan Seasonal-aware Interpolation")
                
                for idx in gap_indices:
                    month = self.data.loc[idx, 'Month']
                    
                    # Ambil data bulan yang sama dari tahun lain
                    same_month_data = self.data[
                        (self.data['Month'] == month) & 
                        (self.data['RR'].notna())
                    ]['RR']
                    
                    if len(same_month_data) > 0:
                        # Gunakan median dengan noise kecil
                        base_value = same_month_data.median()
                        
                        # Tambahkan variability berdasarkan historical std
                        noise_factor = same_month_data.std() * 0.1
                        noise = np.random.normal(0, noise_factor)
                        
                        # Probabilitas hari kering
                        dry_prob = monthly_stats[monthly_stats['Month'] == month]['dry_day_prob'].iloc[0]
                        
                        if np.random.random() < dry_prob:
                            imputed_value = 0
                        else:
                            imputed_value = max(0, base_value + noise)
                        
                        self.data.loc[idx, 'RR_imputed'] = imputed_value
                        self.data.loc[idx, 'imputation_method'] = 'seasonal_interpolation'
                        
            elif gap_length <= 30:
                # METODE 3: Seasonal decomposition untuk gap panjang
                print("  → Menggunakan Seasonal Decomposition")
                
                if seasonal_stats['decomposition']['available']:
                    for idx in gap_indices:
                        month = self.data.loc[idx, 'Month']
                        
                        # Gunakan seasonal pattern dari decomposition
                        if month in seasonal_stats['decomposition']['seasonal']:
                            seasonal_component = seasonal_stats['decomposition']['seasonal'][month]
                            
                            # Tambahkan baseline dari monthly median
                            monthly_baseline = monthly_stats[monthly_stats['Month'] == month]['median'].iloc[0]
                            
                            # Combine dengan uncertainty
                            monthly_std = monthly_stats[monthly_stats['Month'] == month]['std'].iloc[0]
                            uncertainty = np.random.normal(0, monthly_std * 0.2)
                            
                            imputed_value = max(0, seasonal_component + monthly_baseline + uncertainty)
                            
                            # Probabilitas hari kering
                            dry_prob = monthly_stats[monthly_stats['Month'] == month]['dry_day_prob'].iloc[0]
                            if np.random.random() < dry_prob:
                                imputed_value = 0
                            
                            self.data.loc[idx, 'RR_imputed'] = imputed_value
                            self.data.loc[idx, 'imputation_method'] = 'seasonal_decomposition'
                        else:
                            # Fallback ke monthly median
                            monthly_median = monthly_stats[monthly_stats['Month'] == month]['median'].iloc[0]
                            self.data.loc[idx, 'RR_imputed'] = monthly_median
                            self.data.loc[idx, 'imputation_method'] = 'monthly_median'
                else:
                    # Fallback: Monthly median dengan variability
                    print("    → Fallback ke Monthly Median")
                    for idx in gap_indices:
                        month = self.data.loc[idx, 'Month']
                        monthly_median = monthly_stats[monthly_stats['Month'] == month]['median'].iloc[0]
                        self.data.loc[idx, 'RR_imputed'] = monthly_median
                        self.data.loc[idx, 'imputation_method'] = 'monthly_median'
                        
            else:
                # METODE 4: Historical seasonal average untuk gap ekstrem
                print("  → Menggunakan Historical Seasonal Average")
                
                for idx in gap_indices:
                    month = self.data.loc[idx, 'Month']
                    
                    # Gunakan historical data dengan confidence bounds
                    same_month_data = self.data[
                        (self.data['Month'] == month) & 
                        (self.data['RR'].notna())
                    ]['RR']
                    
                    if len(same_month_data) > 0:
                        # Gunakan percentile range untuk uncertainty
                        p25 = same_month_data.quantile(0.25)
                        p75 = same_month_data.quantile(0.75)
                        median = same_month_data.median()
                        
                        # Random selection dalam IQR
                        imputed_value = np.random.uniform(p25, p75)
                        
                        # Probabilitas hari kering
                        dry_prob = monthly_stats[monthly_stats['Month'] == month]['dry_day_prob'].iloc[0]
                        if np.random.random() < dry_prob:
                            imputed_value = 0
                        
                        self.data.loc[idx, 'RR_imputed'] = imputed_value
                        self.data.loc[idx, 'imputation_method'] = 'historical_seasonal'
                    else:
                        # Fallback ke 0 jika tidak ada data historical
                        self.data.loc[idx, 'RR_imputed'] = 0
                        self.data.loc[idx, 'imputation_method'] = 'zero_fallback'
            
            total_imputed += gap_length
            print(f"  ✓ {gap_length} values berhasil diimputasi")
        
        # Summary imputasi
        print(f"\n=== SUMMARY IMPUTASI ===")
        print(f"Total values diimputasi: {total_imputed}")
        
        method_counts = self.data['imputation_method'].value_counts()
        for method, count in method_counts.items():
            if method != 'original':
                print(f"  • {method}: {count} values")
        
        # Statistik hasil imputasi
        original_missing = self.data['RR'].isna().sum()
        after_imputation = self.data['RR_imputed'].isna().sum()
        improvement = original_missing - after_imputation
        
        print(f"\nImprovement: {improvement} values berhasil diimputasi")
        print(f"Missing values: {original_missing} → {after_imputation}")
        print(f"Success rate: {improvement/original_missing*100:.1f}%")
        
        # Validasi hasil imputasi
        imputed_data = self.data[self.data['imputation_method'] != 'original']['RR_imputed']
        if len(imputed_data) > 0:
            print(f"\nStatistik data hasil imputasi:")
            print(f"  • Mean: {imputed_data.mean():.2f}mm")
            print(f"  • Median: {imputed_data.median():.2f}mm")
            print(f"  • Std: {imputed_data.std():.2f}mm")
            print(f"  • Zero days: {(imputed_data == 0).sum()} ({(imputed_data == 0).sum()/len(imputed_data)*100:.1f}%)")
            print(f"  • Range: {imputed_data.min():.2f} - {imputed_data.max():.2f}mm")
        
        # Cleanup temporary columns
        self.data.drop(['is_missing', 'gap_group'], axis=1, inplace=True)
    
    def detect_outliers_advanced(self):
        """
        Deteksi outlier dengan multiple methods
        """
        print("\n=== DETEKSI OUTLIER ADVANCED ===")
        
        valid_data = self.data['RR_imputed'].dropna()
        
        if len(valid_data) == 0:
            print("Tidak ada data valid untuk deteksi outlier")
            return
        
        # Method 1: IQR Method
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        
        # Method 2: Z-Score Method
        z_scores = np.abs(stats.zscore(valid_data))
        z_outliers = valid_data[z_scores > 3]
        
        # Method 3: Modified Z-Score (robust)
        median = valid_data.median()
        mad = np.median(np.abs(valid_data - median))
        modified_z_scores = 0.6745 * (valid_data - median) / mad
        modified_z_outliers = valid_data[np.abs(modified_z_scores) > 3.5]
        
        print(f"IQR Method outliers: {len(iqr_outliers)} (threshold: {upper_bound:.2f}mm)")
        print(f"Z-Score outliers: {len(z_outliers)} (threshold: z > 3)")
        print(f"Modified Z-Score outliers: {len(modified_z_outliers)} (threshold: mz > 3.5)")
        
        # Simpan outliers
        self.outliers = {
            'iqr': iqr_outliers,
            'zscore': z_outliers,
            'modified_zscore': modified_z_outliers,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
        
        # Flag outliers di dataset
        outlier_mask = (self.data['RR_imputed'] < lower_bound) | (self.data['RR_imputed'] > upper_bound)
        self.data['is_outlier'] = outlier_mask
        
        print(f"Total outliers flagged: {outlier_mask.sum()}")
    
    def transform_data(self):
        """
        Transformasi data untuk stabilisasi varians
        """
        print("\n=== TRANSFORMASI DATA ===")
        
        valid_data = self.data['RR_imputed'].dropna()
        
        if len(valid_data) == 0:
            print("Tidak ada data valid untuk transformasi")
            return
        
        # Log transformation (log(x+1) untuk handle zero values)
        self.data['RR_log'] = np.log1p(self.data['RR_imputed'])
        
        # Square root transformation
        self.data['RR_sqrt'] = np.sqrt(self.data['RR_imputed'])
        
        # Box-Cox transformation (hanya untuk nilai positif)
        positive_data = valid_data[valid_data > 0]
        if len(positive_data) > 0:
            try:
                # Fit Box-Cox transformation
                boxcox_data, lambda_param = boxcox(positive_data)
                print(f"Box-Cox lambda parameter: {lambda_param:.4f}")
                
                # Apply Box-Cox to all positive values
                self.data['RR_boxcox'] = np.nan
                positive_mask = (self.data['RR_imputed'] > 0) & (self.data['RR_imputed'].notna())
                self.data.loc[positive_mask, 'RR_boxcox'] = boxcox(self.data.loc[positive_mask, 'RR_imputed'], lmbda=lambda_param)
                
            except Exception as e:
                print(f"Box-Cox transformation failed: {e}")
                self.data['RR_boxcox'] = np.nan
        
        # Evaluasi transformasi
        print("\nEvaluasi transformasi (skewness):")
        print(f"  Original: {valid_data.skew():.4f}")
        print(f"  Log(x+1): {self.data['RR_log'].skew():.4f}")
        print(f"  Sqrt: {self.data['RR_sqrt'].skew():.4f}")
        if 'RR_boxcox' in self.data.columns:
            print(f"  Box-Cox: {self.data['RR_boxcox'].skew():.4f}")
    
    def stationarity_tests(self):
        """
        Test stationarity untuk time series
        """
        print("\n=== UJI STATIONARITY ===")
        
        # Buat time series dengan frekuensi harian
        ts_data = self.data.set_index('Date')['RR_imputed'].dropna()
        
        if len(ts_data) < 50:
            print("Data terlalu sedikit untuk uji stationarity")
            return
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(ts_data)
            print("Augmented Dickey-Fuller Test:")
            print(f"  ADF Statistic: {adf_result[0]:.6f}")
            print(f"  p-value: {adf_result[1]:.6f}")
            print(f"  Critical Values:")
            for key, value in adf_result[4].items():
                print(f"    {key}: {value:.6f}")
            
            if adf_result[1] <= 0.05:
                print("  → Data STASIONER (p-value ≤ 0.05)")
            else:
                print("  → Data NON-STASIONER (p-value > 0.05)")
                
        except Exception as e:
            print(f"ADF test failed: {e}")
        
        # KPSS test
        try:
            kpss_result = kpss(ts_data)
            print("\nKPSS Test:")
            print(f"  KPSS Statistic: {kpss_result[0]:.6f}")
            print(f"  p-value: {kpss_result[1]:.6f}")
            print(f"  Critical Values:")
            for key, value in kpss_result[3].items():
                print(f"    {key}: {value:.6f}")
            
            if kpss_result[1] >= 0.05:
                print("  → Data STASIONER (p-value ≥ 0.05)")
            else:
                print("  → Data NON-STASIONER (p-value < 0.05)")
                
        except Exception as e:
            print(f"KPSS test failed: {e}")
    

    def create_individual_plots(self, output_dir="rainfall_plots", save_plots=True):
        """
        Membuat plots individual untuk analisis dan menyimpannya ke direktori terpisah
        
        Parameters:
        output_dir (str): Direktori untuk menyimpan plots
        save_plots (bool): Apakah akan menyimpan plots ke file
        """
        import os
        
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Direktori {output_dir} dibuat")
        
        print("\n=== MEMBUAT PLOTS INDIVIDUAL ===")
        
        # Setup style untuk semua plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Time Series Plot
        plt.figure(figsize=(16, 8))
        valid_data = self.data[self.data['RR_imputed'].notna()]

        if not valid_data.empty:
            # Urutkan data
            valid_data = valid_data.sort_values('Date')
            
            # Plot garis utama & moving average
            plt.plot(valid_data['Date'], valid_data['RR_imputed'], color='blue', alpha=0.7, linewidth=1.0, label='Curah Hujan')
            
            # Scatter hari hujan tinggi (>50mm)
            high_rain = valid_data[valid_data['RR_imputed'] > 50]
            if not high_rain.empty:
                plt.scatter(high_rain['Date'], high_rain['RR_imputed'], color='red', s=20, alpha=0.8, label='> 50 mm')

            # Garis vertikal awal tahun dan label tiap 2 tahun
            years = sorted(valid_data['Year'].unique())
            for i, y in enumerate(years):
                tgl_awal = pd.Timestamp(f"{y}-01-01")
                plt.axvline(tgl_awal, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
                if i % 2 == 0 or y == years[-1]:
                    plt.text(tgl_awal, plt.ylim()[1]*0.95, str(y), rotation=90, va='top', ha='right', fontsize=9, color='gray')

            # Sumbu X dan grid
            start_date, end_date = valid_data['Date'].min(), valid_data['Date'].max()
            date_ticks = pd.date_range(start=start_date, end=end_date, freq='YS')
            plt.xticks(date_ticks, [d.year for d in date_ticks], rotation=45)
            plt.gca().set_xticks(pd.date_range(start_date, end_date, freq='MS'), minor=True)
            plt.grid(True, alpha=0.3, which='major')
            plt.grid(True, alpha=0.1, which='minor')

            # Judul dan keterangan
            total_days = (end_date - start_date).days
            plt.title(f'Time Series Curah Hujan Harian (RR)\n{start_date:%d %b %Y} - {end_date:%d %b %Y}', fontsize=14, fontweight='bold')
            plt.xlabel('Tahun')
            plt.ylabel('Curah Hujan (mm)')
            plt.legend(loc='upper right', frameon=True)

            # Statistik pojok kiri atas
            plt.text(0.02, 0.98,
                    f'Data Valid: {len(valid_data):,} hari\nMaksimum: {valid_data["RR_imputed"].max():.1f}mm\nMinimum: {valid_data["RR_imputed"].min():.1f}mm',
                    transform=plt.gca().transAxes,
                    fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'Tidak ada data valid untuk ditampilkan',
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{output_dir}/01_time_series_plot.png', dpi=300, bbox_inches='tight')
            print("✅ 01_time_series_plot.png saved")
        plt.show()


        # 2. Histogram dan Distribusi
        plt.figure(figsize=(10, 6))
        valid_rr = self.data['RR_imputed'].dropna()

        if not valid_rr.empty:
            # Filter upper limit untuk menghindari outlier ekstrem
            upper_limit = min(valid_rr.quantile(0.95) * 1.1, 100)
            filtered_rr = valid_rr[valid_rr <= upper_limit]

            # Tentukan bin width merata
            bin_width = 2  # 2 mm per bin
            bins = np.arange(0, upper_limit + bin_width, bin_width)

            # Plot histogram
            plt.hist(filtered_rr, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)

            # Label dan tata letak
            plt.title('Distribusi Curah Hujan Harian (Filtered ≤ {:.0f} mm)'.format(upper_limit), fontsize=13, fontweight='bold')
            plt.xlabel('Curah Hujan (mm)', fontsize=11)
            plt.ylabel('Frekuensi', fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'Tidak ada data valid untuk ditampilkan',
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/02_histogram_distribusi.png', dpi=300, bbox_inches='tight')
            print("✅ 02_histogram_distribusi.png saved")
        plt.show()
        
        # 3. Boxplot Curah Hukan per Tahun
        plt.figure(figsize=(14, 6))
        yearly_data = []
        years = sorted(self.data['Year'].dropna().unique())

        for year in years:
            year_rr = self.data[self.data['Year'] == year]['RR_imputed'].dropna()
            if len(year_rr) > 0:
                yearly_data.append(year_rr)

        if yearly_data:
            plt.boxplot(yearly_data, labels=years, showfliers=True)
            plt.title('Boxplot Curah Hujan Harian per Tahun', fontsize=14, fontweight='bold')
            plt.xlabel('Tahun', fontsize=12)
            plt.ylabel('Curah Hujan Harian (mm)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/03_boxplot_tahunan.png', dpi=300, bbox_inches='tight')
            print("✅ 03_boxplot_tahunan.png saved")
        plt.show()
        
        
        # 4. Pola Musiman (Bar Chart)
        plt.figure(figsize=(10, 6))
        seasonal_mean = self.data.groupby('Month')['RR_imputed'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                    'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
        
        plt.bar(range(1, 13), seasonal_mean.values, alpha=0.7, color='lightcoral')
        plt.title('Rata-rata Curah Hujan per Bulan', fontsize=14, fontweight='bold')
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Rata-rata Curah Hujan (mm)', fontsize=12)
        plt.xticks(range(1, 13), month_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Tambahkan nilai di atas setiap bar
        for i, v in enumerate(seasonal_mean.values):
            plt.text(i+1, v + max(seasonal_mean.values)*0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/04_pola_musiman.png', dpi=300, bbox_inches='tight')
            print("✅ 04_pola_musiman.png saved")
        plt.show()
        

        # 5. Trend Rata-rata Curah Hujan Tahunan - Simplified
        plt.figure(figsize=(14, 6))

        # Hitung rata-rata tahunan
        yearly_mean = self.data.groupby('Year')['RR_imputed'].mean()
        years = yearly_mean.index.astype(int)
        values = yearly_mean.values

        # Plot garis utama
        plt.plot(years, values, marker='o', linewidth=2, color='green', label='Rata-rata Tahunan')

        # Tambahkan trendline
        z = np.polyfit(years, values, 1)
        plt.plot(years, np.poly1d(z)(years), linestyle='--', color='red', label=f'Trend: {z[0]:.2f} mm/tahun')

        # Highlight tahun tertinggi dan terendah
        plt.scatter(years[np.argmax(values)], values.max(), color='red', s=80, label=f'Tertinggi: {years[np.argmax(values)]}')
        plt.scatter(years[np.argmin(values)], values.min(), color='blue', s=80, label=f'Terendah: {years[np.argmin(values)]}')

        # Label, sumbu, grid
        plt.title('Rata-rata Curah Hujan Tahunan', fontsize=14, fontweight='bold')
        plt.xlabel('Tahun')
        plt.ylabel('Rata-rata (mm)')
        plt.xticks(years[::max(1, len(years)//10)], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Statistik ringkasan di pojok
        text = f'''Periode: {years.min()} - {years.max()}
        Rata-rata: {values.mean():.1f} mm/hari
        Rentang: {values.min():.1f} – {values.max():.1f} mm/hari
        Trend: {"↑" if z[0]>0 else "↓"} {abs(z[0]):.2f} mm/tahun'''
        plt.text(0.02, 0.95, text, transform=plt.gca().transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/05_trend_tahunan.png', dpi=300, bbox_inches='tight')
            print("✅ 05_trend_tahunan.png saved")
        plt.show()

        
        # 6. Pola Missing Values
        plt.figure(figsize=(12, 6))
        yearly_missing = self.data.groupby('Year').agg({
            'RR_original': lambda x: (x == 8888).sum() + x.isna().sum(),
            'Date': 'count'
        })
        missing_pct = yearly_missing['RR_original'] / yearly_missing['Date'] * 100
        
        plt.bar(missing_pct.index, missing_pct.values, alpha=0.7, color='red')
        plt.title('Persentase Missing Values per Tahun', fontsize=14, fontweight='bold')
        plt.xlabel('Tahun', fontsize=12)
        plt.ylabel('Missing Values (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/06_missing_values_pattern.png', dpi=300, bbox_inches='tight')
            print("✅ 06_missing_values_pattern.png saved")
        plt.show()
        
        # 7. Transformasi Comparison
        plt.figure(figsize=(15, 5))
        
        # Original distribution
        plt.subplot(1, 3, 1)
        valid_original = self.data['RR_imputed'].dropna()
        plt.hist(valid_original, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribusi Original', fontsize=12, fontweight='bold')
        plt.xlabel('RR (mm)')
        plt.ylabel('Frekuensi')
        plt.grid(True, alpha=0.3)
        
        # Log transformation
        plt.subplot(1, 3, 2)
        if 'RR_log' in self.data.columns:
            valid_log = self.data['RR_log'].dropna()
            plt.hist(valid_log, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.title('Distribusi Log Transform', fontsize=12, fontweight='bold')
            plt.xlabel('Log(RR + 1)')
            plt.ylabel('Frekuensi')
            plt.grid(True, alpha=0.3)
        
        # Square root transformation
        plt.subplot(1, 3, 3)
        if 'RR_sqrt' in self.data.columns:
            valid_sqrt = self.data['RR_sqrt'].dropna()
            plt.hist(valid_sqrt, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.title('Distribusi Sqrt Transform', fontsize=12, fontweight='bold')
            plt.xlabel('√RR')
            plt.ylabel('Frekuensi')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/07_transformasi_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ 07_transformasi_comparison.png saved")
        plt.show()
        
        # 8. Curah Hujan Bulanan Kumulatif
        plt.figure(figsize=(12, 6))
        monthly_cumsum = self.data.groupby(['Year', 'Month'])['RR_imputed'].sum().reset_index()
        monthly_cumsum['Date'] = pd.to_datetime(monthly_cumsum[['Year', 'Month']].assign(day=1))
        monthly_cumsum = monthly_cumsum.sort_values('Date')
        
        plt.plot(monthly_cumsum['Date'], monthly_cumsum['RR_imputed'], linewidth=2, color='darkblue')
        plt.title('Curah Hujan Bulanan Kumulatif', fontsize=14, fontweight='bold')
        plt.xlabel('Tanggal', fontsize=12)
        plt.ylabel('Curah Hujan Bulanan (mm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Tambahkan moving average
        if len(monthly_cumsum) > 12:
            monthly_cumsum['MA_12'] = monthly_cumsum['RR_imputed'].rolling(window=12).mean()
            plt.plot(monthly_cumsum['Date'], monthly_cumsum['MA_12'], 
                    linewidth=2, color='red', alpha=0.7, label='Moving Average (12 bulan)')
            plt.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/08_curah_hujan_kumulatif.png', dpi=300, bbox_inches='tight')
            print("✅ 08_curah_hujan_kumulatif.png saved")
        plt.show()
        
        # 9. Seasonal Decomposition
        plt.figure(figsize=(12, 10))
        try:
            # Buat monthly time series untuk decomposition
            monthly_ts = self.data.groupby(['Year', 'Month'])['RR_imputed'].mean().reset_index()
            monthly_ts['Date'] = pd.to_datetime(monthly_ts[['Year', 'Month']].assign(day=1))
            monthly_ts = monthly_ts.set_index('Date')['RR_imputed'].dropna()
            
            if len(monthly_ts) >= 24:  # Minimal 2 tahun
                decomposition = seasonal_decompose(monthly_ts, model='additive', period=12)
                
                # Plot components
                plt.subplot(4, 1, 1)
                plt.plot(decomposition.observed, linewidth=2, color='blue')
                plt.title('Observed', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(4, 1, 2)
                plt.plot(decomposition.trend, linewidth=2, color='green')
                plt.title('Trend', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(4, 1, 3)
                plt.plot(decomposition.seasonal, linewidth=2, color='orange')
                plt.title('Seasonal', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(4, 1, 4)
                plt.plot(decomposition.resid, linewidth=2, color='red')
                plt.title('Residual', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.suptitle('Seasonal Decomposition', fontsize=16, fontweight='bold')
                
            else:
                plt.text(0.5, 0.5, 'Data tidak cukup untuk decomposition\n(minimal 24 bulan)', 
                        ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
                plt.title('Seasonal Decomposition', fontsize=14, fontweight='bold')
                
        except Exception as e:
            plt.text(0.5, 0.5, f'Decomposition error:\n{str(e)}', 
                    ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
            plt.title('Seasonal Decomposition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/09_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
            print("✅ 09_seasonal_decomposition.png saved")
        plt.show()
        
        # 10. Data Quality Summary 
        fig, ax = plt.subplots(figsize=(8, 7))

        # Hitung metrik kualitas data
        missing_8888 = self.missing_stats['missing_8888']
        missing_nan = self.missing_stats['missing_nan']
        zero_values = self.missing_stats['zero_values']
        valid_data = len(self.data) - missing_8888 - missing_nan
        total_data = len(self.data)
        missing_data = missing_8888 + missing_nan

        # Donut chart setup
        donut_vals = [valid_data, missing_data, zero_values]
        donut_labels = ['Valid', 'Missing', 'Zero']
        colors = ['#4CAF50', '#FF6B6B', '#FFD700']

        wedges, _, _ = ax.pie(donut_vals, labels=donut_labels, autopct='%1.1f%%',
                            startangle=90, pctdistance=0.85, colors=colors,
                            textprops={'fontsize': 10})
        ax.add_artist(plt.Circle((0, 0), 0.70, fc='white'))  # Buat donut

        ax.set_title('📊 Komposisi Kualitas Data', fontsize=14, fontweight='bold')

        # Ringkasan statistik
        score_valid = valid_data / total_data * 100

        summary_text = f'''Ringkasan:
        • Total: {total_data:,}
        • Valid: {valid_data:,} ({score_valid:.1f}%)
        • Missing: {missing_data:,} ({missing_data/total_data*100:.1f}%)
        • Zero: {zero_values:,} ({zero_values/total_data*100:.1f}%)
        '''

        plt.figtext(0.5, 0.01, summary_text, fontsize=10, fontfamily='monospace',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

        plt.tight_layout()

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/10_data_quality_summary.png', dpi=300, bbox_inches='tight')
            print("✅ 10_data_quality_summary.png saved")
        plt.show()

    def summary_report(self):
        """
        Laporan ringkasan preprocessing
        """
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN PREPROCESSING CURAH HUJAN")
        print("="*60)
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Periode: {self.data['Date'].min()} s/d {self.data['Date'].max()}")
        print(f"   • Rentang tahun: {self.data['Year'].max() - self.data['Year'].min() + 1} tahun")
        
        print(f"\n🔍 DATA QUALITY:")
        print(f"   • Missing values (8888): {self.missing_stats['missing_8888']:,} ({self.missing_stats['missing_8888']/len(self.data)*100:.1f}%)")
        print(f"   • Missing values (NaN): {self.missing_stats['missing_nan']:,}")
        print(f"   • Zero values: {self.missing_stats['zero_values']:,}")
        print(f"   • Valid data: {len(self.data) - self.missing_stats['missing_8888'] - self.missing_stats['missing_nan']:,}")
        
        valid_data = self.data['RR_imputed'].dropna()
        if len(valid_data) > 0:
            print(f"\n📈 STATISTIK DESKRIPTIF (setelah preprocessing):")
            print(f"   • Mean: {valid_data.mean():.2f} mm")
            print(f"   • Median: {valid_data.median():.2f} mm")
            print(f"   • Std Dev: {valid_data.std():.2f} mm")
            print(f"   • Min: {valid_data.min():.2f} mm")
            print(f"   • Max: {valid_data.max():.2f} mm")
            print(f"   • Skewness: {valid_data.skew():.2f}")
        
        if hasattr(self, 'outliers') and 'iqr' in self.outliers:
            print(f"\n⚠️  OUTLIER DETECTION:")
            print(f"   • IQR method: {len(self.outliers['iqr'])} outliers")
            print(f"   • Upper bound: {self.outliers['bounds']['upper']:.2f} mm")
        
        print(f"\n✅ PREPROCESSING STEPS COMPLETED:")
        print(f"   • ✓ Data cleaning & special values handling")
        print(f"   • ✓ Missing values analysis")
        print(f"   • ✓ Outlier detection")
        print(f"   • ✓ Data transformation")
        print(f"   • ✓ Stationarity testing")
        print(f"   • ✓ Comprehensive visualization")
        
        print(f"\n📋 REKOMENDASI UNTUK FORECASTING:")
        missing_pct = self.missing_stats['missing_percentage']
        if missing_pct > 50:
            print(f"   • ⚠️  Missing values tinggi ({missing_pct:.1f}%) - pertimbangkan aggregasi temporal")
        elif missing_pct > 20:
            print(f"   • ⚠️  Missing values sedang ({missing_pct:.1f}%) - gunakan imputasi hati-hati")
        else:
            print(f"   • ✅ Missing values rendah ({missing_pct:.1f}%) - data cukup baik")
        
        if len(valid_data) > 0 and valid_data.skew() > 2:
            print(f"   • ✅ Distribusi sangat skewed - gunakan transformasi log")
        elif len(valid_data) > 0 and valid_data.skew() > 1:
            print(f"   • ✅ Distribusi skewed - pertimbangkan transformasi")
        
        print(f"\n🎯 SIAP UNTUK HOLT-WINTERS FORECASTING!")
        print("="*60)

def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    print("🌧️  PREPROCESSING CURAH HUJAN - DATASET BMKG")
    print("="*50)

    import sys; sys.stdout = open("preprocessing_log.txt", "w")

    
    # Contoh data loading (sesuaikan dengan path file Anda)
    try:
        # Ganti dengan path file data Anda
        data_path = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"  # atau "data_bmkg.xlsx"
        
        # Load data
        print(f"📂 Loading data dari: {data_path}")
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Format file tidak didukung. Gunakan .csv atau .xlsx")
        
        # Validasi kolom yang diperlukan
        required_columns = ['Date', 'RR']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Kolom berikut tidak ditemukan: {missing_columns}")
            print(f"📋 Kolom yang tersedia: {list(df.columns)}")
            return
        
        print(f"✅ Data berhasil dimuat: {len(df)} records")
        print(f"📋 Kolom tersedia: {list(df.columns)}")
        
        # Inisialisasi preprocessor
        preprocessor = RainfallPreprocessor(df)
        
        # FASE 1: Load dan persiapan data
        print("\n🔄 FASE 1: Loading dan Persiapan Data")
        preprocessor.load_and_prepare_data()
        
        # FASE 2: Analisis missing values
        print("\n🔄 FASE 2: Analisis Missing Values")
        yearly_missing = preprocessor.analyze_missing_values()
        
        # FASE 3: Cleaning data
        print("\n🔄 FASE 3: Data Cleaning")
        preprocessor.clean_rainfall_data()
        
        # FASE 4: Imputasi musiman
        print("\n🔄 FASE 4: Imputasi Berbasis Musiman")
        preprocessor.seasonal_imputation()
        
        # FASE 5: Deteksi outlier
        print("\n🔄 FASE 5: Deteksi Outlier Advanced")
        preprocessor.detect_outliers_advanced()
        
        # FASE 6: Transformasi data
        print("\n🔄 FASE 6: Transformasi Data")
        preprocessor.transform_data()
        
        # FASE 7: Uji stationarity
        print("\n🔄 FASE 7: Uji Stationarity")
        preprocessor.stationarity_tests()
        
        # FASE 8: Visualisasi komprehensif
        print("\n🔄 FASE 8: Visualisasi Komprehensif")
        preprocessor.create_individual_plots()
        
        # FASE 9: Laporan ringkasan
        print("\n🔄 FASE 9: Laporan Ringkasan")
        preprocessor.summary_report()
        
        # Simpan hasil preprocessing
        output_path = "preprocessed_rainfall_data.csv"
        
        # Simpan hanya kolom terkait RR
        rr_columns = [
            'Date', 'Year', 'Month', 'Day',
            'RR_original', 'RR_estimation_method',
            'RR_imputed', 'imputation_method', 'is_outlier',
            'RR_log', 'RR_sqrt', 'RR_boxcox'
        ]
        preprocessor.data[rr_columns].to_csv(output_path, index=False)
        print(f"\n💾 Data hasil preprocessing disimpan ke: {output_path}")
        
        # Informasi kolom hasil preprocessing
        print(f"\n📊 KOLOM HASIL PREPROCESSING:")
        processed_columns = [
            'Date', 'Year', 'Month', 'Day',
            'RR_original', 'RR_imputed', 'RR_log', 'RR_sqrt',
            'is_outlier'
        ]
        
        available_columns = [col for col in processed_columns if col in preprocessor.data.columns]
        for col in available_columns:
            non_null_count = preprocessor.data[col].notna().sum()
            print(f"   • {col}: {non_null_count:,} non-null values")
        
        # Rekomendasi untuk Holt-Winters
        print(f"\n🎯 REKOMENDASI UNTUK HOLT-WINTERS:")
        valid_data = preprocessor.data['RR_imputed'].dropna()
        
        if len(valid_data) > 0:
            # Cek seasonal pattern
            seasonal_strength = preprocessor.data.groupby('Month')['RR_imputed'].std().mean()
            print(f"   • Data tersedia untuk forecasting: {len(valid_data):,} records")
            print(f"   • Seasonal strength: {seasonal_strength:.2f}")
            
            # Rekomendasi parameter Holt-Winters
            if seasonal_strength > 10:
                print(f"   • Gunakan Holt-Winters ADDITIVE (seasonal pattern kuat)")
            else:
                print(f"   • Gunakan Holt-Winters MULTIPLICATIVE (seasonal pattern lemah)")
            
            # Rekomendasi periode seasonal
            print(f"   • Seasonal period: 12 (bulanan) atau 365 (harian)")
            
            # Rekomendasi transformasi
            skewness = valid_data.skew()
            if skewness > 2:
                print(f"   • Gunakan RR_log untuk forecasting (skewness tinggi: {skewness:.2f})")
            elif skewness > 1:
                print(f"   • Pertimbangkan RR_sqrt untuk forecasting (skewness sedang: {skewness:.2f})")
            else:
                print(f"   • Gunakan RR_imputed untuk forecasting (skewness rendah: {skewness:.2f})")
        
        print(f"\n🎉 PREPROCESSING SELESAI!")
        print(f"📁 File output: {output_path}")
        print(f"🚀 Siap untuk implementasi Holt-Winters forecasting!")
        
        return preprocessor
        
    except FileNotFoundError:
        print(f"❌ Error: File {data_path} tidak ditemukan!")
        print("📋 Pastikan file data BMKG tersedia dengan kolom:")
        print("   • Date: Tanggal observasi")
        print("   • RR: Curah hujan (mm)")
        print("   • Kolom lain (opsional): RH_AVG, Temperature, dll.")
        return None
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        print("🔍 Periksa kembali format data dan kolom yang tersedia")
        return None
    
if __name__== "__main__":
    result = main() 