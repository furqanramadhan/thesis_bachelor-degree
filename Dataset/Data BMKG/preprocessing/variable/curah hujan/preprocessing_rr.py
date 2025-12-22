import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
from scipy.stats import boxcox
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import os
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
        self.LOG_CONSTANT = 0.01
        
    def load_and_prepare_data(self):
        """
        Load data dan persiapan awal dengan proper tracking flags
        """
        # Konversi Date ke datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

        # ✅ CREATE TRACKING FLAGS BEFORE ANY MODIFICATION
        # Track original missing data (NaN)
        self._original_nan_mask = self.data['RR'].isna()
        self.data['is_missing_original'] = self._original_nan_mask.copy()
        
        # Track 8888 values
        self.data['is_8888'] = (self.data['RR'] == 8888)
        
        # Track 9999 values  
        self.data['is_9999'] = (self.data['RR'] == 9999)
        
        # Track originally valid data
        self.data['is_original_valid'] = (
            self.data['RR'].notna() & 
            (self.data['RR'] != 8888) & 
            (self.data['RR'] != 9999)
        )

        if 'SS' in self.data.columns:
            self.data['SS'] = self.data['SS'].replace(9999, np.nan)
        
        # Buat kolom tambahan untuk analisis
        self.data['Year'] = self.data['Date'].dt.year
        self.data['month'] = self.data['Date'].dt.month
        self.data['day'] = self.data['Date'].dt.day
        
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
        Membersihkan data curah hujan dengan pendekatan yang disederhanakan
        """
        print("\n=== CLEANING DATA CURAH HUJAN ===")
        
        # Backup data original
        self.data['RR_original'] = self.data['RR'].copy()
        
        # Step 1: Handle missing data codes
        self.data['RR'] = self.data['RR'].replace(9999, np.nan)  # Missing data
        
        # Step 2: Estimate unmeasured rainfall (8888)
        self._estimate_unmeasured_rainfall()
        
        # Step 3: Identify and handle outliers
        self._identify_outliers()
        
        # Step 4: Print summary statistics
        self._print_cleaning_summary()
        
        print("✅ Data cleaning selesai\n")
        
    def transform_to_log_domain(self):
        """
        Transform RR ke log domain SEBELUM imputasi
        Handles estimated 8888 values correctly
        Menggunakan log(RR + 0.01) sesuai best practice
        """
        print("\n=== TRANSFORMASI KE LOG DOMAIN ===")
        
        # Backup data sebelum transformasi
        self.data['RR_mm'] = self.data['RR_original'].copy() if 'RR_original' in self.data.columns else self.data['RR'].copy()
        
        # Hitung statisik data asli
        original_mask = ~self.data.get('is_8888', pd.Series(False, index=self.data.index))
        valid_data = self.data.loc[original_mask, 'RR'].dropna()
        
        if len(valid_data) > 0:
            print(f"Statistik SEBELUM transformasi (domain mm):")
            print(f"  Mean: {valid_data.mean():.2f} mm")
            print(f"  Median: {valid_data.median():.2f} mm")
            print(f"  Std: {valid_data.std():.2f} mm")
            print(f"  Skewness: {valid_data.skew():.2f}")
            print(f"  Min: {valid_data.min():.2f} mm")
            print(f"  Max: {valid_data.max():.2f} mm")
        
        # Step 1: Transform original data (including NaN where 8888 was replaced)
        self.data['RR_log'] = np.log(self.data['RR'] + self.LOG_CONSTANT)
        
        # Step 2: Use pre-estimated log values for 8888 locations
        if 'RR_log_estimated' in self.data.columns:
            mask_8888 = self.data.get('is_8888', pd.Series(False, index=self.data.index))
            estimated_count = mask_8888.sum()
            if estimated_count > 0:
                self.data.loc[mask_8888, 'RR_log'] = self.data.loc[mask_8888, 'RR_log_estimated']
                print(f"  ✅ Used pre-estimated log values for {estimated_count} data (8888)")
        
        # Statistik setelah transformasi
        valid_log = self.data['RR_log'].dropna()
        if len(valid_log) > 0:
            print(f"\nStatistik SETELAH transformasi (domain log):")
            print(f"  Mean: {valid_log.mean():.3f}")
            print(f"  Median: {valid_log.median():.3f}")
            print(f"  Std: {valid_log.std():.3f}")
            print(f"  Skewness: {valid_log.skew():.3f}")
            print(f"  Min: {valid_log.min():.3f}")
            print(f"  Max: {valid_log.max():.3f}")
            
            # ✅ Compare skewness on ORIGINAL data only (fair comparison)
            if len(valid_data) > 0:
                # Calculate skewness for original data only (exclude estimated)
                original_log = self.data.loc[original_mask, 'RR_log'].dropna()
                if len(original_log) > 0:
                    skew_reduction = abs(valid_data.skew()) - abs(original_log.skew())
                    print(f"\n✅ Skewness reduction: {skew_reduction:.3f}")
                    print(f"   ({abs(valid_data.skew()):.3f} → {abs(original_log.skew()):.3f})")
    
        
        # Analisis distribusi zero values
        zero_count = (valid_data == 0).sum() if len(valid_data) > 0 else 0
        if zero_count > 0:
            log_zero_value = np.log(0 + self.LOG_CONSTANT)
            print(f"\n📊 Zero-inflation analysis:")
            print(f"  Zero values: {zero_count} ({zero_count/len(self.data)*100:.1f}%)")
            print(f"  log(0 + {self.LOG_CONSTANT}) = {log_zero_value:.3f}")
            if len(valid_log) > 0:
                print(f"  Range preserved: {log_zero_value:.3f} to {valid_log.max():.3f}")
        
        print("\n✅ Transformasi selesai. Imputasi akan dilakukan di log-space.")
        
    def inverse_transform_from_log(self, log_values):  
        """
        Inverse transform dengan safety checks - preserves distribution
        """
        mm_values = np.exp(log_values) - self.LOG_CONSTANT
        
        # ✅ FIX: Use quantile-based cap to preserve distribution shape
        if not hasattr(self, '_max_cap_mm'):
            # Calculate once from original data
            original_valid = self.data['RR_original'].replace([8888, 9999], np.nan).dropna()
            if len(original_valid) > 0:
                # Use P99.9 instead of hard 200mm - preserves extreme events
                self._max_cap_mm = max(200, original_valid.quantile(0.999))
                print(f"🔧 Dynamic cap calculated: {self._max_cap_mm:.1f}mm (P99.9 of original data)")
            else:
                self._max_cap_mm = 200
                print(f"🔧 Fallback cap used: {self._max_cap_mm}mm")
        
        # Clip using dynamic threshold
        return np.clip(mm_values, 0, self._max_cap_mm)
        
    def _safe_log_value(self, log_value):
        """
        Ensure log values stay in realistic range
        """
        # Based on physical limits:
        # Min: 0 mm → log(0.01) ≈ -4.6
        # Max: 200 mm → log(200.01) ≈ 5.3
        return np.clip(log_value, -4.6, 5.3)

    def _calculate_log_ss_adjustment(self, ss):
        """
        Hitung adjustment di log domain (additive)
        """
        if pd.isna(ss):
            return 0.0
        # Adjustment dalam log space
        if ss < 1.5:
            return np.random.uniform(0.25, 0.35)  # boost
        elif ss < 3.0:
            return np.random.uniform(0.10, 0.20)  # slight boost
        elif ss < 6.0:
            return np.random.uniform(-0.05, 0.05)  # neutral
        elif ss < 8.0:
            return np.random.uniform(-0.20, -0.10)  # reduce
        else:
            return np.random.uniform(-0.35, -0.25)  # strong reduction
        
    def seasonal_imputation_log_domain(self):
        """
        Imputasi di log domain dengan metode advanced
        """
        print("\n=== IMPUTASI DI LOG DOMAIN ===")
        
        # Initialize imputed column di log domain
        self.data['RR_log_imputed'] = self.data['RR_log'].copy()
        missing_count = self.data['RR_log_imputed'].isna().sum()
        
        if missing_count == 0:
            print("Tidak ada missing values untuk diimputasi")
            return True
        
        print(f"Memproses {missing_count} missing values di log domain...")
        
        # Calculate monthly statistics di LOG domain
        monthly_stats_log = self._calculate_monthly_stats_log()
        
        # Identify missing patterns
        missing_patterns = self._identify_missing_patterns_log()
        
        # Apply imputation strategy
        total_imputed = 0
        
        for pattern in missing_patterns:
            gap_length = pattern['length']
            indices = pattern['indices']
            
            if gap_length <= 3:
                # Short gaps: cubic spline di log-space
                imputed_count = self._impute_short_gaps_cubic_spline(indices)
                method = "cubic_spline_log"
            elif gap_length <= 14:
                # Medium gaps: monthly pattern + smooth
                imputed_count = self._impute_medium_gaps_log(indices, monthly_stats_log)
                method = "monthly_smooth_log"
            else:
                # Long gaps: LOWESS smoothing
                imputed_count = self._impute_long_gaps_lowess(indices, monthly_stats_log)
                method = "lowess_log"
            
            total_imputed += imputed_count
            print(f"  → Gap length {gap_length}: {imputed_count} values imputed using {method}")
        
        print(f"\n✅ {total_imputed} values berhasil diimputasi di log domain")
        
        # Convert back to mm domain untuk interpretasi
        self.data['RR_imputed'] = self.inverse_transform_from_log(
            self.data['RR_log_imputed']
        )
        
        # ✅ FIX: Use pre-created flags for accurate data source tracking
        self.data['RR_source'] = 'original'
        
        # Use flags created during data loading/processing
        if 'is_8888' in self.data.columns:
            self.data.loc[self.data['is_8888'], 'RR_source'] = 'estimated_8888'
        
        # Track originally missing data (NaN before any processing)
        if not hasattr(self, '_original_nan_mask'):
            # Fallback: try to identify from current state
            self._original_nan_mask = self.data['RR_original'].isna()
        
        if hasattr(self, '_original_nan_mask'):
            self.data.loc[self._original_nan_mask, 'RR_source'] = 'imputed_missing'
        
        # Override with more specific imputation method tracking
        if 'imputation_method' in self.data.columns:
            imputed_mask = self.data['imputation_method'].notna()
            # Create more descriptive source labels
            self.data.loc[imputed_mask, 'RR_source'] = 'imputed_' + self.data.loc[imputed_mask, 'imputation_method']
        
        # Print summary
        self._print_imputation_summary_log()
        
        return self.data['RR_log_imputed'].isna().sum() == 0  
    
    def _calculate_monthly_stats_log(self):
        """
        Hitung statistik bulanan di LOG domain
        """
        monthly_stats = {}
        
        for month in range(1, 13):
            month_data = self.data[
                (self.data['month'] == month) & 
                (self.data['RR_log'].notna())
            ]['RR_log']
            
            if len(month_data) > 0:
                monthly_stats[month] = {
                    'mean': month_data.mean(),
                    'median': month_data.median(),
                    'std': month_data.std(),
                    'q25': month_data.quantile(0.25),
                    'q75': month_data.quantile(0.75),
                    'count': len(month_data)
                }
            else:
                # Fallback
                overall_data = self.data['RR_log'].dropna()
                monthly_stats[month] = {
                    'mean': overall_data.mean() if len(overall_data) > 0 else np.log(2 + self.LOG_CONSTANT),
                    'median': overall_data.median() if len(overall_data) > 0 else np.log(1 + self.LOG_CONSTANT),
                    'std': overall_data.std() if len(overall_data) > 0 else 1.0,
                    'q25': overall_data.quantile(0.25) if len(overall_data) > 0 else np.log(0.5 + self.LOG_CONSTANT),
                    'q75': overall_data.quantile(0.75) if len(overall_data) > 0 else np.log(5 + self.LOG_CONSTANT),
                    'count': 0
                }
        
        return monthly_stats
    
    def _identify_missing_patterns_log(self):
        """
        Identifikasi pola missing data di log domain
        """
        missing_mask = self.data['RR_log_imputed'].isna()
        
        if not missing_mask.any():
            return []
        
        groups = []
        current_group = []
        
        for idx, is_missing in missing_mask.items():
            if is_missing:
                current_group.append(idx)
            else:
                if current_group:
                    groups.append({
                        'indices': current_group,
                        'length': len(current_group),
                        'start': current_group[0],
                        'end': current_group[-1]
                    })
                    current_group = []
        
        if current_group:
            groups.append({
                'indices': current_group,
                'length': len(current_group),
                'start': current_group[0],
                'end': current_group[-1]
            })
        
        return groups

    def _impute_short_gaps_cubic_spline(self, indices):
        """
        SHORT GAPS (≤3 hari): Cubic spline interpolation di log-space
        """
        if not indices:
            return 0
                
        # Get context window (10 days before and after)
        start_idx = max(0, indices[0] - 10)
        end_idx = min(len(self.data) - 1, indices[-1] + 10)
        
        # Extract valid data points around the gap
        context_data = self.data.loc[start_idx:end_idx].copy()
        valid_mask = context_data['RR_log_imputed'].notna()
        
        if valid_mask.sum() < 4:  # Need at least 4 points for cubic spline
            # Fallback to linear interpolation
            temp_series = self.data['RR_log_imputed'].copy()
            interpolated = temp_series.interpolate(method='linear', limit_direction='both')
            
            for idx in indices:
                if pd.isna(self.data.loc[idx, 'RR_log_imputed']):
                    imputed_value = interpolated.loc[idx]
                    imputed_value = self._safe_log_value(imputed_value)  # ✅ ADD SAFETY
                    self.data.loc[idx, 'RR_log_imputed'] = imputed_value
                    self.data.loc[idx, 'imputation_method'] = 'linear_interpolation_log'
        else:
            # Use cubic spline
            x_valid = context_data[valid_mask].index.values
            y_valid = context_data.loc[valid_mask, 'RR_log_imputed'].values
            
            # Create cubic spline
            cs = CubicSpline(x_valid, y_valid)
            
            # Interpolate missing values
            for idx in indices:
                if pd.isna(self.data.loc[idx, 'RR_log_imputed']):
                    imputed_value = cs(idx)
                    imputed_value = self._safe_log_value(imputed_value)  # ✅ ADD SAFETY
                    self.data.loc[idx, 'RR_log_imputed'] = imputed_value
                    self.data.loc[idx, 'imputation_method'] = 'cubic_spline_log'
        
        return len(indices)

    def _impute_medium_gaps_log(self, indices, monthly_stats_log):
        """
        MEDIUM GAPS (4-14 hari): Monthly pattern + CONSERVATIVE noise
        """        
        for idx in indices:
            if pd.isna(self.data.loc[idx, 'RR_log_imputed']):
                month = self.data.loc[idx, 'month']
                stats = monthly_stats_log[month]
                
                # Base value: monthly median di log domain
                base_log = stats['median']
                
                # ✅ FIX: Reduce noise significantly
                # 0.2 in log domain = ~1.22x multiplier (more conservative)
                noise = np.random.normal(0, min(stats['std'] * 0.15, 0.2))
                log_value = base_log + noise
                
                # Apply SS adjustment (additive di log domain)
                ss = self.data.loc[idx, 'SS'] if 'SS' in self.data.columns else 4.0
                if not pd.isna(ss):
                    ss_adjustment = self._calculate_log_ss_adjustment(ss)
                    # ✅ FIX: Scale down SS adjustment too
                    log_value += ss_adjustment * 0.5  # Reduce SS impact
                
                # ✅ APPLY SAFETY BOUNDS
                log_value = self._safe_log_value(log_value)
                
                self.data.loc[idx, 'RR_log_imputed'] = log_value
                self.data.loc[idx, 'imputation_method'] = 'monthly_smooth_log'
        
        # Apply Savitzky-Golay filter untuk smoothing
        gap_start = indices[0]
        gap_end = indices[-1]
        window_start = max(0, gap_start - 7)
        window_end = min(len(self.data) - 1, gap_end + 7)
        
        window_data = self.data.loc[window_start:window_end, 'RR_log_imputed'].copy()
        
        # Fill any remaining NaN in window before smoothing
        window_data = window_data.interpolate(method='linear', limit_direction='both')
        
        if len(window_data) >= 5:  # Minimum window for Savitzky-Golay
            try:
                smoothed = savgol_filter(window_data.values, 
                                window_length=min(5, len(window_data) if len(window_data) % 2 == 1 else len(window_data)-1),
                                polyorder=2)
                
                # Apply smoothed values only to imputed indices with safety bounds
                for i, idx in enumerate(range(window_start, window_end + 1)):
                    if idx in indices:
                        smoothed_value = self._safe_log_value(smoothed[i])  # ✅ APPLY SAFETY
                        self.data.loc[idx, 'RR_log_imputed'] = smoothed_value
            except:
                pass  # Keep original imputed values if smoothing fails
        
        return len(indices)

    def _impute_long_gaps_lowess(self, indices, monthly_stats_log):
        """
        LONG GAPS (>14 hari): LOWESS smoothing with safe boundaries
        """
        
        if not indices:
            return 0
        
        # Get extended context (30 days before and after)
        gap_start = indices[0]
        gap_end = indices[-1]
        context_start = max(0, gap_start - 30)
        context_end = min(len(self.data) - 1, gap_end + 30)
        
        # Extract context data
        context_data = self.data.loc[context_start:context_end].copy()
        valid_mask = context_data['RR_log_imputed'].notna()
        
        if valid_mask.sum() < 10:  # Need sufficient points for LOWESS
            # Fallback: use monthly median
            for idx in indices:
                if pd.isna(self.data.loc[idx, 'RR_log_imputed']):
                    month = self.data.loc[idx, 'month']
                    median_value = self._safe_log_value(monthly_stats_log[month]['median'])  # ✅ SAFETY
                    self.data.loc[idx, 'RR_log_imputed'] = median_value
                    self.data.loc[idx, 'imputation_method'] = 'monthly_median_log'
        else:
            # Prepare data for LOWESS
            x_valid = np.arange(len(context_data))[valid_mask]
            y_valid = context_data.loc[valid_mask, 'RR_log_imputed'].values
            
            # Apply LOWESS smoothing
            try:
                smoothed = lowess(y_valid, x_valid, frac=0.1, return_sorted=False)
                
                # ✅ SAFE INTERPOLATION - use bounded fill values
                f_interp = interp1d(x_valid, smoothed, kind='linear', 
                                bounds_error=False,
                                fill_value=(smoothed[0], smoothed[-1]))  # Use edge values
                
                # Impute missing values
                for idx in indices:
                    if pd.isna(self.data.loc[idx, 'RR_log_imputed']):
                        relative_idx = idx - context_start
                        imputed_value = f_interp(relative_idx)
                        imputed_value = self._safe_log_value(imputed_value)  # ✅ APPLY SAFETY
                        self.data.loc[idx, 'RR_log_imputed'] = imputed_value
                        self.data.loc[idx, 'imputation_method'] = 'lowess_log'
            except:
                # Fallback if LOWESS fails
                for idx in indices:
                    if pd.isna(self.data.loc[idx, 'RR_log_imputed']):
                        month = self.data.loc[idx, 'month']
                        median_value = self._safe_log_value(monthly_stats_log[month]['median'])  # ✅ SAFETY
                        self.data.loc[idx, 'RR_log_imputed'] = median_value
                        self.data.loc[idx, 'imputation_method'] = 'monthly_median_log_fallback'
        
        return len(indices)
    
    def _print_imputation_summary_log(self):
        """
        Print ringkasan hasil imputasi di log domain
        """
        imputed_data = self.data[self.data.get('imputation_method', '').str.len() > 0]
        
        if len(imputed_data) > 0:
            print("\n📊 Metode imputasi yang digunakan:")
            method_counts = imputed_data['imputation_method'].value_counts()
            for method, count in method_counts.items():
                print(f"  • {method}: {count} values")
        
        # Statistik di log domain
        final_log = self.data['RR_log_imputed'].dropna()
        if len(final_log) > 0:
            print(f"\n📉 Statistik final (LOG domain):")
            print(f"  • Total data: {len(final_log)}")
            print(f"  • Mean: {final_log.mean():.3f}")
            print(f"  • Median: {final_log.median():.3f}")
            print(f"  • Std: {final_log.std():.3f}")
            print(f"  • Missing: {self.data['RR_log_imputed'].isna().sum()}")
        
        # Statistik di mm domain
        final_mm = self.data['RR_imputed'].dropna()
        if len(final_mm) > 0:
            print(f"\n📉 Statistik final (MM domain):")
            print(f"  • Total data: {len(final_mm)}")
            print(f"  • Mean: {final_mm.mean():.2f} mm")
            print(f"  • Median: {final_mm.median():.2f} mm")
            print(f"  • Std: {final_mm.std():.2f} mm")
            
            # Compare with original
            if 'RR_original' in self.data.columns:
                original_clean = self.data['RR_original'].replace([8888, 9999], np.nan).dropna()
                if len(original_clean) > 0:
                    mean_diff = abs(final_mm.mean() - original_clean.mean())
                    print(f"  • Deviasi dari original: {mean_diff:.2f} mm")       
                    
    def detect_outliers_log_domain(self):
        """
        Deteksi outlier di log domain dengan final safety checks
        """
        print("\n=== DETEKSI OUTLIER DI LOG DOMAIN ===")
        
        valid_log = self.data['RR_log_imputed'].dropna()
        
        if valid_log.empty:
            print("Tidak ada data valid untuk deteksi outlier.")
            return
        
        # ✅ APPLY FINAL SAFETY CLIPPING to all log values
        self.data['RR_log_imputed'] = self.data['RR_log_imputed'].apply(
            lambda x: self._safe_log_value(x) if pd.notna(x) else x
        )
        
        # Recalculate after safety clipping
        valid_log = self.data['RR_log_imputed'].dropna()
        
        # Use P95 instead of P99 for more conservative outlier detection
        p95_log = valid_log.quantile(0.95)
        outlier_mask_log = self.data['RR_log_imputed'] > p95_log
        
        # Convert threshold ke mm untuk interpretasi
        p95_mm = self.inverse_transform_from_log(p95_log)
        
        # Flag outliers
        self.data['is_outlier_log'] = outlier_mask_log
        self.data['is_outlier'] = outlier_mask_log  # For compatibility
        
        # ✅ FINAL BACK-TRANSFORMATION WITH SAFETY
        self.data['RR_imputed'] = self.inverse_transform_from_log(
            self.data['RR_log_imputed']
        )
        
        # Statistics
        outlier_count = outlier_mask_log.sum()
        
        print(f"📊 Threshold P95 (log domain): {p95_log:.3f}")
        print(f"📊 Threshold P95 (mm domain): {p95_mm:.2f} mm")
        print(f"✅ Total outliers flagged: {outlier_count} dari {len(valid_log)} data ({outlier_count/len(valid_log)*100:.2f}%)")
        
        # ✅ CHECK FINAL STATISTICS
        final_mm = self.data['RR_imputed'].dropna()
        if len(final_mm) > 0:
            print(f"\n📊 FINAL MM STATISTICS (after safety fixes):")
            print(f"   • Mean: {final_mm.mean():.2f} mm")
            print(f"   • Median: {final_mm.median():.2f} mm") 
            print(f"   • Max: {final_mm.max():.2f} mm")
            print(f"   • Skewness: {final_mm.skew():.2f}")
            
            if final_mm.skew() > 10:
                print(f"   ⚠️  Skewness still high - additional fixes may be needed")
            else:
                print(f"   ✅ Skewness significantly improved!")
        
        # Show examples
        if outlier_count > 0:
            outliers_log = self.data[outlier_mask_log][['Date', 'RR_log_imputed', 'RR_imputed']].head(5)
            print(f"\n📋 Contoh outliers terdeteksi:")
            for _, row in outliers_log.iterrows():
                print(f"  • {row['Date']}: log={row['RR_log_imputed']:.3f}, mm={row['RR_imputed']:.2f}")
        
        # Store outliers
        self.outliers['log_domain'] = {
            'threshold_log': p95_log,
            'threshold_mm': p95_mm,
            'count': outlier_count,
            'indices': self.data[outlier_mask_log].index.tolist()
        }
        self.data['RR_log_transformed'] = self.data['RR_imputed'] 
        
        print(f"\n💡 Outlier detection di log domain dengan safety bounds applied")
        
    def _estimate_unmeasured_rainfall(self):
        """
        Estimasi data 8888 (tidak terukur) directly di log domain
        """
        mask_8888 = self.data['RR'] == 8888
        count_8888 = mask_8888.sum()
        
        if count_8888 == 0:
            return

        print(f"Estimasi {count_8888} data tidak terukur (8888) di LOG DOMAIN...")
        
        self.data['is_8888'] = mask_8888.copy()
        
        # Mapping kondisi meteorologi ke kategori hujan (mm)
        conditions = {
            'heavy_rain': {'rh_min': 90, 'tn_max': 23, 'rain_range': (7, 20)},
            'moderate_rain': {'rh_min': 80, 'tn_max': 24, 'rain_range': (2, 10)},
            'light_rain': {'rh_min': 70, 'tn_max': 24, 'rain_range': (0.5, 5)},
            'dry': {'rh_min': 0, 'tn_max': 100, 'rain_range': (0, 1)}
        }
        
        estimated_values_log = []
        
        for idx in self.data[mask_8888].index:
            row = self.data.loc[idx]
            rh = row.get('RH_AVG', 75)
            tn = row.get('TN', 25)
            ss = row.get('SS', 4.0)
            
            # Determine rainfall category
            category = 'dry'
            for cat, cond in conditions.items():
                if cat != 'dry' and rh >= cond['rh_min'] and tn <= cond['tn_max']:
                    category = cat
                    break
            
            # Get mm range first - will convert to log later
            min_val, max_val = conditions[category]['rain_range']
            
            # Convert range to log domain
            min_log = np.log(min_val + self.LOG_CONSTANT)
            max_log = np.log(max_val + self.LOG_CONSTANT)
            
            # Generate base log value directly
            base_value_log = np.random.uniform(min_log, max_log)
            
            # Apply SS adjustment (additive in log domain)
            ss_adjustment = self._calculate_log_ss_adjustment(ss)
            estimated_log = base_value_log + ss_adjustment
            
            # Apply safety bounds
            estimated_log = self._safe_log_value(estimated_log)
            estimated_values_log.append(estimated_log)      
            
        # ✅ FIX: Store ONLY log values, don't touch RR yet
        # RR_log column will be created in transform_to_log_domain()
        # For now, just replace 8888 with NaN to be handled later
        self.data.loc[mask_8888, 'RR'] = np.nan
        # Store estimated log values for later use
        self.data.loc[mask_8888, 'RR_log_estimated'] = estimated_values_log
        self.data.loc[mask_8888, 'RR_estimation_method'] = 'meteorological_log_domain'
        
        print(f"  → {len(estimated_values_log)} nilai berhasil diestimasi di log domain")
        print(f"  → Range (log): {min(estimated_values_log):.3f} - {max(estimated_values_log):.3f}")

    
    def _identify_outliers(self):
        """
        Identifikasi outlier ekstrem (>150mm/hari)
        """
        outlier_threshold = 150
        outliers = self.data[self.data['RR'] > outlier_threshold]
        
        print(f"Outlier ekstrem (>{outlier_threshold}mm): {len(outliers)} data")
        
        if len(outliers) > 0:
            print("  Dates dengan outlier:")
            for idx, row in outliers.head(5).iterrows():  # Show max 5
                print(f"    {row['Date']}: {row['RR']:.1f}mm")
        
        # Store outliers for analysis
        self.outliers = outliers
    
    def _print_cleaning_summary(self):
        """
        Print ringkasan hasil cleaning
        """
        valid_data = self.data['RR'].dropna()
        
        if len(valid_data) == 0:
            print("⚠️  Tidak ada data valid setelah cleaning")
            return
        
        print(f"\nRingkasan data setelah cleaning:")
        print(f"  Total data valid: {len(valid_data)}")
        print(f"  Mean: {valid_data.mean():.2f}mm")
        print(f"  Median: {valid_data.median():.2f}mm")
        print(f"  Min: {valid_data.min():.2f}mm")
        print(f"  Max: {valid_data.max():.2f}mm")
        
        # Rainfall categories distribution
        categories = self._categorize_rainfall(valid_data)
        print(f"\nDistribusi kategori hujan:")
        for cat, count in categories.items():
            pct = count / len(valid_data) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")

    def _categorize_rainfall(self, data):
        """
        Kategorisasi curah hujan berdasarkan intensitas harian sesuai standar BMKG.
        """
        return {
            'Berawan (0–0.4 mm)': ((data >= 0) & (data <= 0.4)).sum(),
            'Hujan ringan (0.5–19.9 mm)': ((data > 0.4) & (data <= 19.9)).sum(),
            'Hujan sedang (20–49.9 mm)': ((data >= 20) & (data <= 49.9)).sum(),
            'Hujan lebat (50–99.9 mm)': ((data >= 50) & (data <= 99.9)).sum(),
            'Hujan sangat lebat (100–150 mm)': ((data >= 100) & (data <= 150)).sum(),
            'Hujan ekstrem (>150 mm)': (data > 150).sum()
        }  
        

    def create_individual_plots(self, output_dir="rainfall_plots", save_plots=True):
        """
        Membuat plots individual untuk analisis dan menyimpannya ke direktori terpisah
        
        Parameters:
        output_dir (str): Direktori untuk menyimpan plots
        save_plots (bool): Apakah akan menyimpan plots ke file
        """
        
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Direktori {output_dir} dibuat")
        
        print("\n=== MEMBUAT PLOTS INDIVIDUAL ===")

    
        # PLOT 01 - TIME SERIES BEFORE & AFTER
        print("🔄 Plot #1: Time Series Before & After Preprocessing")

        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

        # Prepare data
        valid_data = self.data[self.data['Date'].notna()].sort_values('Date')

        # SUBPLOT 1: BEFORE (RR_original)
        if not valid_data.empty:
            # Data normal (bukan 8888, bukan NaN, bukan 9999)
            normal_mask = (valid_data['RR_original'] != 8888) & (valid_data['RR_original'].notna()) & (valid_data['RR_original'] != 9999)
            normal_data = valid_data[normal_mask]
            
            # Plot garis utama (hanya data normal)
            ax1.plot(normal_data['Date'], normal_data['RR_original'], 
                    color='blue', alpha=0.7, linewidth=2.0, label='Data Original')
            
            # Marker untuk nilai 8888 (tidak terukur)
            mask_8888 = valid_data['RR_original'] == 8888
            if mask_8888.any():
                ax1.scatter(valid_data[mask_8888]['Date'], 
                        [0] * mask_8888.sum(),  # Plot di y=0 untuk visibility
                        color='red', marker='d', s=50, alpha=0.8, 
                        label=f'Tidak Terukur (8888): {mask_8888.sum()} data', zorder=5)
            
            # Marker untuk NaN
            mask_nan = valid_data['RR_original'].isna()
            if mask_nan.any():
                ax1.scatter(valid_data[mask_nan]['Date'], 
                        [0] * mask_nan.sum(),  # Plot di y=0 untuk visibility
                        color='gray', marker='x', s=50, alpha=0.8, 
                        label=f'Missing (NaN): {mask_nan.sum()} data', zorder=5)
            
            # Kustomisasi subplot 1
            ax1.set_ylabel('Curah Hujan (mm)', fontsize=12)
            ax1.set_title('Before Preprocessing (RR_original)', fontsize=13, fontweight='bold', pad=10)
            ax1.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            ax1.set_ylim(0, 200)
            ax1.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, 
                    shadow=True, framealpha=0.9)
            ax1.set_facecolor('white')

        # SUBPLOT 2: AFTER (RR_imputed)
        valid_imputed = self.data[self.data['RR_imputed'].notna()].sort_values('Date')

        if not valid_imputed.empty:
            # Plot garis utama
            ax2.plot(valid_imputed['Date'], valid_imputed['RR_imputed'], 
                    color='blue', alpha=0.7, linewidth=2.0, label='Data Imputed')
            
            # Highlight kategori hujan
            categories = {
                'Hujan Sedang (20–49.9 mm)': ((valid_imputed['RR_imputed'] >= 20) & (valid_imputed['RR_imputed'] <= 49.9)),
                'Hujan Lebat (50–99.9 mm)': ((valid_imputed['RR_imputed'] >= 50) & (valid_imputed['RR_imputed'] <= 99.9)),
                'Hujan Sangat Lebat (100–150 mm)': ((valid_imputed['RR_imputed'] >= 100) & (valid_imputed['RR_imputed'] <= 150)),
                'Hujan Ekstrem (>150 mm)': (valid_imputed['RR_imputed'] > 150)
            }
            colors = {
                'Hujan Sedang (20–49.9 mm)': '#06923E',
                'Hujan Lebat (50–99.9 mm)': '#FFDE63',
                'Hujan Sangat Lebat (100–150 mm)': '#DC2525',
                'Hujan Ekstrem (>150 mm)': '#222831'
            }
            
            for label, mask in categories.items():
                subset = valid_imputed[mask]
                if not subset.empty:
                    ax2.scatter(subset['Date'], subset['RR_imputed'], label=label,
                            color=colors[label], s=45, alpha=0.9)
            
            # Set ticks untuk sumbu X
            start_date, end_date = valid_imputed['Date'].min(), valid_imputed['Date'].max()
            date_ticks = pd.date_range(start=start_date, end=end_date, freq='YS')
            ax2.set_xticks(date_ticks)
            ax2.set_xticklabels([d.year for d in date_ticks], rotation=0, ha='center', fontsize=12)
            ax2.xaxis.set_minor_locator(plt.matplotlib.dates.MonthLocator())
            
            # Kustomisasi subplot 2
            ax2.set_xlabel('Tahun', fontsize=12)
            ax2.set_ylabel('Curah Hujan (mm)', fontsize=12)
            ax2.set_title('After Preprocessing (RR_imputed)', fontsize=13, fontweight='bold', pad=10)
            ax2.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            ax2.set_ylim(0, 200)
            ax2.legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, 
                    shadow=True, framealpha=0.9)
            ax2.set_facecolor('white')

        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/01_time_series_before_after.png', dpi=300, 
                        bbox_inches='tight', facecolor='white', edgecolor='none')
            print("✅ 01_time_series_before_after.png saved")

        plt.show()
        
        # PLOT 02 - BOXPLOT BULANAN BEFORE & AFTER OUTLIER TREATMENT
        print("🔄 Plot #2: Boxplot Bulanan Before & After Outlier Treatment")

        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        months = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        month_labels = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei',
                        'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov']

        # ========================
        # SUBPLOT 1: BEFORE (RR_imputed tanpa treatment)
        # ========================
        monthly_data_before = []
        for month in months:
            month_rr = self.data[self.data['month'] == month]['RR_imputed'].dropna()
            if not month_rr.empty:
                monthly_data_before.append(month_rr)
            else:
                monthly_data_before.append(pd.Series(dtype=float))

        if any(len(m) > 0 for m in monthly_data_before):
            box1 = ax1.boxplot(monthly_data_before, labels=month_labels,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='o', markerfacecolor='red', 
                                            markersize=4, linestyle='none', 
                                            markeredgecolor='red', alpha=0.6))
            
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            for patch, color in zip(box1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Statistik untuk subplot 1
            all_data_before = pd.concat(monthly_data_before, ignore_index=True).dropna()
            if not all_data_before.empty:
                outlier_count = (all_data_before > all_data_before.quantile(0.99)).sum()
                ax1.text(0.02, 0.98, 
                        f'Outliers (P99): {outlier_count} | Skewness: {all_data_before.skew():.2f}',
                        transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax1.set_ylabel('Curah Hujan (mm)', fontsize=12)
            ax1.set_title('Before Treatment (RR_imputed)', fontsize=13, fontweight='bold', pad=10)
            ax1.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            ax1.set_ylim(0, 200)
            ax1.set_facecolor('white')

       # PLOT 02 - BOXPLOT BULANAN: ORIGINAL vs LOG-TRANSFORMED DATA
        print("🔄 Plot #2: Boxplot Bulanan Original vs Log-Transformed Data")

        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

        months = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        month_labels = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei',
                        'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov']

        # ========================
        # SUBPLOT 1: ORIGINAL DATA (RR_original - cleaned but not log-transformed)
        # ========================
        monthly_data_original = []
        for month in months:
            # Use RR_original but exclude 8888, 9999, and NaN
            month_data = self.data[self.data['month'] == month]['RR_original']
            month_rr = month_data[(month_data != 8888) & (month_data != 9999) & (month_data.notna())]
            
            if not month_rr.empty:
                monthly_data_original.append(month_rr)
            else:
                monthly_data_original.append(pd.Series(dtype=float))

        if any(len(m) > 0 for m in monthly_data_original):
            box1 = ax1.boxplot(monthly_data_original, labels=month_labels,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='o', markerfacecolor='red', 
                                            markersize=4, linestyle='none', 
                                            markeredgecolor='red', alpha=0.6))
            
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            for patch, color in zip(box1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Statistik untuk subplot 1 (Original data)
            all_data_original = pd.concat(monthly_data_original, ignore_index=True).dropna()
            if not all_data_original.empty:
                outlier_count_orig = (all_data_original > all_data_original.quantile(0.99)).sum()
                ax1.text(0.02, 0.98, 
                        f'Data: {len(all_data_original)} | P99 Outliers: {outlier_count_orig} | Skewness: {all_data_original.skew():.2f}',
                        transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            ax1.set_ylabel('Curah Hujan (mm)', fontsize=12)
            ax1.set_title('Original Data (RR_original - Raw Scale)', fontsize=13, fontweight='bold', pad=10)
            ax1.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            ax1.set_ylim(0, min(200, all_data_original.quantile(0.95) * 1.2 if not all_data_original.empty else 200))
            ax1.set_facecolor('white')

        # ========================
        # SUBPLOT 2: LOG-TRANSFORMED DATA (RR_log_imputed converted back to mm)
        # ========================
        monthly_data_log = []
        for month in months:
            # Show the final processed data in mm scale (converted from log)
            month_rr = self.data[self.data['month'] == month]['RR_imputed'].dropna()
            if not month_rr.empty:
                monthly_data_log.append(month_rr)
            else:
                monthly_data_log.append(pd.Series(dtype=float))

        if any(len(m) > 0 for m in monthly_data_log):
            box2 = ax2.boxplot(monthly_data_log, labels=month_labels,
                            patch_artist=True, showfliers=True,
                            flierprops=dict(marker='o', markerfacecolor='blue', 
                                            markersize=4, linestyle='none', 
                                            markeredgecolor='blue', alpha=0.6))
            
            for patch, color in zip(box2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            # Statistik untuk subplot 2 (Log-processed data)
            all_data_log = pd.concat(monthly_data_log, ignore_index=True).dropna()
            if not all_data_log.empty:
                # Use log domain threshold converted to mm
                log_threshold = self.data['RR_log_imputed'].quantile(0.99)
                mm_threshold = self.inverse_transform_from_log(log_threshold)
                outlier_count_log = (all_data_log > mm_threshold).sum()
                
                ax2.text(0.02, 0.98, 
                        f'Data: {len(all_data_log)} | Log P99 Outliers: {outlier_count_log} | Skewness: {all_data_log.skew():.2f}',
                        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            ax2.set_xlabel('Bulan', fontsize=12)
            ax2.set_ylabel('Curah Hujan (mm)', fontsize=12)
            ax2.set_title('Log-Processed Data (RR_imputed - After Log+0.01 Processing)', fontsize=13, fontweight='bold', pad=10)
            ax2.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            ax2.set_xticks(range(1, 13))
            ax2.set_xticklabels(month_labels, rotation=0, ha='center')
            ax2.set_ylim(0, min(200, all_data_log.quantile(0.95) * 1.2 if not all_data_log.empty else 200))
            ax2.set_facecolor('white')

        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/02_boxplot_original_vs_log_processed.png', dpi=300, 
                        bbox_inches='tight', facecolor='white', edgecolor='none')
            print("✅ 02_boxplot_original_vs_log_processed.png saved")

        plt.show()

         # 3. Pergeseran Pola Curah Hujan Bulanan (Des–Nov) - MODIFIED
        print("🔄 Plot #3: Pergeseran Pola Curah Hujan Bulanan (Des–Nov)")

        # Set background menjadi putih
        plt.style.use('default')  # Reset ke style default
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        plt.figure(figsize=(14, 8))

        # Periode yang diinginkan: tahun individual + gabungan
        periods = {
            '2020': [2020],
            '2021': [2021],
            '2022': [2022],
            '2023': [2023],
            '2024': [2024],
            '2025': [2025],  # Hingga Juni 2025
            '2005-2025': list(range(2005, 2026))
        }

        month_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        month_names = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 
                    'Jun', 'Jul', 'Ags', 'Sep', 'Okt', 'Nov']

        # Warna solid pekat sesuai permintaan
        colors = {
            '2020': '#A16D28',    # Coklat
            '2021': '#DC143C',    # Merah
            '2022': '#228B22',    # Hijau
            '2023': '#0000FF',    # Biru
            '2024': '#FF8C00',    # Jingga
            '2025': '#FFD700',    # Ungu untuk 2025
            '2005-2025': '#000000'  # Hitam
        }

        custom_labels = {
            '2020': 'Total Curah Hujan 2020',
            '2021': 'Total Curah Hujan 2021',
            '2022': 'Total Curah Hujan 2022',
            '2023': 'Total Curah Hujan 2023',
            '2024': 'Total Curah Hujan 2024',
            '2025': 'Total Curah Hujan 2025 (Jan–Juni)',
            '2005-2025': 'Rata-Rata Curah Hujan 2005–2025'
        }
        

        line_styles = ['-'] * len(periods)  

        shift_summary = {}

        for i, (label, years) in enumerate(periods.items()):
            period_data = self.data[self.data['Year'].isin(years)]
            
            # Untuk 2025, filter hanya sampai Juni
            if label == '2025':
                period_data = period_data[period_data['month'] <= 6]
            
            if period_data.empty:
                continue
            
            # Hitung TOTAL curah hujan bulanan (bukan rata-rata)
            monthly_total = period_data.groupby('month')['RR_imputed'].sum()
            
            # Untuk periode gabungan 2005-2025, hitung rata-rata dari total tahunan
            if label == '2005-2025':
                # Hitung total per tahun per bulan, lalu rata-rata
                yearly_monthly_total = period_data.groupby(['Year', 'month'])['RR_imputed'].sum().reset_index()
                monthly_avg_total = yearly_monthly_total.groupby('month')['RR_imputed'].mean()
                monthly_values = [monthly_avg_total.get(m, 0) for m in month_order]
            else:
                monthly_values = [monthly_total.get(m, 0) for m in month_order]
            
            # Untuk 2025, hanya tampilkan Jan-Jun (tidak termasuk Des)
            if label == '2025':
                # Buat array dengan NaN untuk semua bulan
                full_monthly_values = [np.nan] * 12
                for idx, month in enumerate(month_order):
                    if 1 <= month <= 6:  # Hanya isi Jan-Jun untuk 2025
                        full_monthly_values[idx] = monthly_values[idx] if idx < len(monthly_values) else np.nan
                monthly_values = full_monthly_values
            
            # Plot garis untuk semua periode
            if label == '2005-2025':
                plt.plot(month_names, monthly_values, marker='o', linewidth=3.5, markersize=8,
                        color=colors[label], linestyle=line_styles[0], label=custom_labels[label], alpha=0.9)
            elif label == '2025':
                # Untuk 2025, plot hanya bagian yang memiliki data (Jan-Jun)
                jan_idx = month_names.index('Jan')
                jun_idx = month_names.index('Jun')
                plt.plot(month_names[jan_idx:jun_idx+1], monthly_values[jan_idx:jun_idx+1], 
                        marker='o', linewidth=2.5, markersize=6,
                        color=colors[label], linestyle=line_styles[0], label=custom_labels[label], alpha=0.8)
            else:
                plt.plot(month_names, monthly_values, marker='o', linewidth=2.5, markersize=6,
                        color=colors[label], linestyle=line_styles[0], label=custom_labels[label], alpha=0.8)
            
            max_idx = np.argmax(monthly_values)
            min_idx = np.argmin(monthly_values)            
           
            # Simpan ringkasan untuk analisis pergeseran
            shift_summary[label] = {
                'max_month': month_order[max_idx],
                'max_value': monthly_values[max_idx],
                'min_month': month_order[min_idx],
                'min_value': monthly_values[min_idx],
                'total': sum(monthly_values)
            }

        # Kustomisasi grafik
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Curah Hujan (mm)', fontsize=12)

        # Grid 
        plt.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
        plt.grid(True, axis='y', which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='lightgray')

        # Urutkan legend berdasarkan urutan yang diinginkan
        legend_order = ['2005-2025','2020','2021','2022','2023','2024','2025']
        custom_legend_order = [custom_labels[k] for k in legend_order]  # Konversi ke label final

        handles, labels = plt.gca().get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))

        ordered_handles = [legend_dict[label] for label in custom_legend_order if label in legend_dict]
        ordered_labels = [label for label in custom_legend_order if label in legend_dict]


        plt.legend(ordered_handles, ordered_labels, loc='upper right', fontsize=10, 
                frameon=True, fancybox=True, shadow=True, framealpha=0.9)

        # Tulisan horizontal
        plt.xticks(rotation=0, ha='center')

        plt.ylim(bottom=0)

        # Set background axes menjadi putih
        plt.gca().set_facecolor('white')

        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/03_pergeseran_curah_hujan.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print("✅ 03_pergeseran_curah_hujan.png saved")

        plt.show()
    def summary_report(self):
        """
        Laporan ringkasan preprocessing dengan perbandingan yang fair
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
        
        # Original data statistics
        print(f"\n📉 STATISTIK RR_original (sebelum imputasi):")
        original_rr = self.data['RR_original'].replace([8888, 9999], np.nan).dropna()
        if len(original_rr) > 0:
            print(f"   • Count: {len(original_rr):,}")
            print(f"   • Mean: {original_rr.mean():.2f} mm")
            print(f"   • Std Dev: {original_rr.std():.2f} mm")
            print(f"   • Min: {original_rr.min():.2f} mm")
            print(f"   • Q1 (25%): {original_rr.quantile(0.25):.2f} mm")
            print(f"   • Median (Q2): {original_rr.median():.2f} mm")
            print(f"   • Q3 (75%): {original_rr.quantile(0.75):.2f} mm")
            print(f"   • Max: {original_rr.max():.2f} mm")
            print(f"   • Skewness: {original_rr.skew():.2f}")

        print(f"\n📉 STATISTIK RR_imputed (setelah imputasi):")
        valid_data = self.data['RR_imputed'].dropna()
        if len(valid_data) > 0:
            print(f"   • Count: {len(valid_data):,}")
            print(f"   • Mean: {valid_data.mean():.2f} mm")
            print(f"   • Std Dev: {valid_data.std():.2f} mm")
            print(f"   • Min: {valid_data.min():.2f} mm")
            print(f"   • Q1 (25%): {valid_data.quantile(0.25):.2f} mm")
            print(f"   • Median (Q2): {valid_data.median():.2f} mm")
            print(f"   • Q3 (75%): {valid_data.quantile(0.75):.2f} mm")
            print(f"   • Max: {valid_data.max():.2f} mm")
            print(f"   • Skewness: {valid_data.skew():.2f}")
        else:
            print("   ⚠️ Tidak ada data valid dalam RR_imputed")

        # Fair comparison
        print(f"\n📊 PERBANDINGAN FAIR (Same Data Points):")
        original_indices = (self.data['RR_original'].notna()) & \
                        (self.data['RR_original'] != 8888) & \
                        (self.data['RR_original'] != 9999)
        
        if original_indices.any():
            original_subset = self.data.loc[original_indices, 'RR_original']
            imputed_subset = self.data.loc[original_indices, 'RR_imputed']
            
            skew_before = original_subset.skew()
            skew_after = imputed_subset.skew()
            mean_before = original_subset.mean() 
            mean_after = imputed_subset.mean()
            std_before = original_subset.std()
            std_after = imputed_subset.std()
            
            print(f"   • Sample size: {original_indices.sum():,} (identical data points)")
            print(f"   • Mean: {mean_before:.2f} → {mean_after:.2f} mm (Δ: {abs(mean_after-mean_before):.2f})")
            print(f"   • Std Dev: {std_before:.2f} → {std_after:.2f} mm") 
            print(f"   • Skewness: {skew_before:.2f} → {skew_after:.2f}")
            
            if abs(skew_before) > 0.1:
                if abs(skew_after) < abs(skew_before):
                    improvement = ((abs(skew_before) - abs(skew_after)) / abs(skew_before)) * 100
                    print(f"   • ✅ Skewness improved by {improvement:.1f}%")
                else:
                    degradation = ((abs(skew_after) - abs(skew_before)) / abs(skew_before)) * 100
                    print(f"   • ⚠️ Skewness degraded by {degradation:.1f}%")
            else:
                print(f"   • ℹ️ Original skewness already low, minimal change expected")
            
            mean_change_pct = abs(mean_after - mean_before) / mean_before * 100
            if mean_change_pct < 5:
                print(f"   • ✅ Mean well preserved (change: {mean_change_pct:.1f}%)")
            else:
                print(f"   • ⚠️ Mean significantly changed (change: {mean_change_pct:.1f}%)")
        else:
            print("   • ⚠️ No valid original data points for comparison")

        # Imputation impact
        print(f"\n📈 IMPACT ANALISIS IMPUTASI:")
        estimated_8888_count = self.data.get('is_8888', pd.Series(False, index=self.data.index)).sum()
        imputed_nan_count = len(valid_data) - len(original_rr) - estimated_8888_count
        
        if estimated_8888_count > 0:
            print(f"   • Data 8888 yang diestimasi: {estimated_8888_count:,}")
        if imputed_nan_count > 0:
            print(f"   • Missing values yang diimputasi: {imputed_nan_count:,}")
        
        total_added = estimated_8888_count + imputed_nan_count
        if total_added > 0:
            coverage_improvement = (total_added / len(self.data)) * 100
            print(f"   • Total data coverage improvement: {coverage_improvement:.1f}%")
            print(f"   • Final data completeness: {len(valid_data)/len(self.data)*100:.1f}%")

        # ✅ FIX: Outlier detection results - safer access
        print(f"\n⚠️  OUTLIER DETECTION:")
        if hasattr(self, 'outliers') and 'log_domain' in self.outliers:
            outlier_info = self.outliers['log_domain']
            
            # Safe access to outlier count
            outlier_count = outlier_info.get('count', 0)
            threshold_mm = outlier_info.get('threshold_mm', 0)
            
            print(f"   • Log domain P95 method: {outlier_count} outliers")
            print(f"   • Threshold (mm): {threshold_mm:.2f} mm")
            if len(valid_data) > 0:
                print(f"   • Outlier percentage: {outlier_count/len(valid_data)*100:.2f}%")
        else:
            # Calculate outlier info from data if not available
            if 'is_outlier' in self.data.columns:
                outlier_count = self.data['is_outlier'].sum()
                print(f"   • Detected outliers: {outlier_count}")
                if len(valid_data) > 0:
                    print(f"   • Outlier percentage: {outlier_count/len(valid_data)*100:.2f}%")
            else:
                print("   • No outlier detection performed")
        
        print(f"\n✅ PREPROCESSING STEPS COMPLETED:")
        print(f"   • ✓ Data cleaning & special values handling")  
        print(f"   • ✓ Log+0.01 transformation for skewness reduction")
        print(f"   • ✓ Advanced imputation in log domain")
        print(f"   • ✓ Outlier detection with safety bounds")
        print(f"   • ✓ Visualisasi komprehensif")
        
        print(f"\n📋 REKOMENDASI UNTUK FORECASTING:")
        missing_pct = self.missing_stats['missing_percentage']
        if missing_pct > 50:
            print(f"   • ⚠️  Missing values tinggi ({missing_pct:.1f}%) - pertimbangkan aggregasi temporal")
        elif missing_pct > 20:
            print(f"   • ⚠️  Missing values sedang ({missing_pct:.1f}%) - gunakan imputasi hati-hati")
        else:
            print(f"   • ✅ Missing values rendah ({missing_pct:.1f}%) - data cukup baik")
        
        # Final quality assessment
        if len(valid_data) > 0:
            final_skewness = valid_data.skew()
            if final_skewness <= 3:
                print(f"   • ✅ Skewness baik ({final_skewness:.2f}) - siap untuk modeling")
            elif final_skewness <= 8:
                print(f"   • ✅ Skewness acceptable ({final_skewness:.2f}) - cocok untuk rainfall data")
            else:
                print(f"   • ⚠️ Skewness tinggi ({final_skewness:.2f}) - pertimbangkan transformasi tambahan")
        
        print(f"\n🎯 SIAP UNTUK HOLT-WINTERS FORECASTING!")
        print("="*60)

def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    print("🌧️  PREPROCESSING CURAH HUJAN - DATASET BMKG")
    print("="*50)

    # Setup direktori output
    output_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Utara/Stasiun Meteorologi Malikussaleh/CSV CLEANED/curah hujan"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file akan disimpan di direktori output
    sys.stdout = open(os.path.join(output_dir, "preprocessing_log_rainfall.txt"), "w")

    try:
        data_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Utara/Stasiun Meteorologi Malikussaleh/CSV/BMKG_Data_All.csv"
        
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
        preprocessor.analyze_missing_values()
        
        # FASE 3: Cleaning data
        print("\n🔄 FASE 3: Data Cleaning")
        preprocessor.clean_rainfall_data()
        
        # FASE 4: Transform to Log Domain (NEW)
        print("\n🔄 FASE 4: Transformasi ke Log Domain")
        preprocessor.transform_to_log_domain()
        
        # FASE 5: Imputasi di Log Domain (NEW) 
        print("\n🔄 FASE 5: Imputasi di Log Domain")
        preprocessor.seasonal_imputation_log_domain()
        
        # FASE 6: Deteksi Outlier di Log Domain (NEW)
        print("\n🔄 FASE 6: Deteksi Outlier di Log Domain") 
        preprocessor.detect_outliers_log_domain()
        
        # FASE 7: Visualisasi komprehensif
        print("\n🔄 FASE 7: Visualisasi Komprehensif")
        plots_dir = os.path.join(output_dir, "rainfall_plots")
        preprocessor.create_individual_plots(output_dir=plots_dir, save_plots=True)
        
        # FASE 8: Laporan ringkasan
        print("\n🔄 FASE 8: Laporan Ringkasan")
        preprocessor.summary_report()
        
        # Simpan hasil preprocessing
        output_path = os.path.join(output_dir, "preprocessed_rainfall_data.csv")
        
        # Simpan hanya kolom terkait RR
        rr_columns = [
            'Date', 'Year', 'month', 'day',
            'RR_original', 'RR_estimation_method',
            'RR_imputed', 'imputation_method', 'is_outlier'
        ]
        preprocessor.data[rr_columns].to_csv(output_path, index=False)
                
        # Informasi kolom hasil preprocessing
        print(f"\n📊 KOLOM HASIL PREPROCESSING:")
        processed_columns = [
            'Date', 'Year', 'month', 'day',
            'RR_original', 'RR_imputed', 'is_outlier'
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
            seasonal_strength = preprocessor.data.groupby('month')['RR_imputed'].std().mean()
            print(f"   • Data tersedia untuk forecasting: {len(valid_data):,} records")
            print(f"   • Seasonal strength: {seasonal_strength:.2f}")
            
            
        print(f"\n🎉 PREPROCESSING SELESAI!")
        print(f"📁 File output: {output_path}")
        
        return preprocessor
        
    except FileNotFoundError:
        print(f"❌ Error: File {data_path} tidak ditemukan!")
        print("📋 Pastikan file data BMKG tersedia dengan kolom:")
        print("   • Date: Tanggal observasi")
        print("   • RR: Curah hujan (mm)")

        return None
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        return None
    
if __name__== "__main__":
    result = main() 