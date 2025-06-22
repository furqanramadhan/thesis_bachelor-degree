import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.interpolate import CubicSpline
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BuoyDataPreprocessor:
    """
    Enhanced preprocessing class for oceanographic buoy data with physics-aware imputation
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None
        self.location_ranges = {
            '4N90E': {'start': '2005-01-01', 'end': '2018-12-31'},
            '0N90E': {'start': '2005-01-01', 'end': '2020-06-07'},
            '8N90E': {'start': '2005-01-01', 'end': '2020-02-21'}
        }
        
        # Physical constraints for variables
        self.constraints = {
            'SST': {'min': 20.0, 'max': 35.0, 'name': 'Sea Surface Temperature (°C)'},
            'RAD': {'min': 0.0, 'max': 500.0, 'name': 'Solar Radiation (W/m²)'}
        }
        
        self.imputation_log = {}
        
    def load_and_prepare(self):
        """📋 Tahap 1: Load data dan seleksi kolom dengan validasi"""
        print("🔄 Loading and preparing data...")
        
        # Load data
        self.raw_data = pd.read_csv(self.filepath)
        print(f"📊 Raw data shape: {self.raw_data.shape}")
        
        # Convert date dan seleksi kolom
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        selected_cols = ['Date', 'Location', 'SST', 'RAD']
        self.raw_data = self.raw_data[selected_cols].copy()
        
        # Filter rentang waktu global
        start_date = '2005-01-01'
        end_date = '2020-12-31'
        mask = (self.raw_data['Date'] >= start_date) & (self.raw_data['Date'] <= end_date)
        self.raw_data = self.raw_data[mask].copy()
        
        print(f"✅ Data loaded: {len(self.raw_data)} records")
        print(f"📊 Locations: {self.raw_data['Location'].unique()}")
        
        return self
    
    def apply_location_cutoffs(self):
        """⚠️ Tahap 2: Penanganan khusus per lokasi"""
        print("🔄 Applying location-specific cutoffs...")
        
        filtered_data = []
        
        for location, date_range in self.location_ranges.items():
            if location in self.raw_data['Location'].values:
                loc_data = self.raw_data[self.raw_data['Location'] == location].copy()
                
                mask = (loc_data['Date'] >= date_range['start']) & (loc_data['Date'] <= date_range['end'])
                loc_data = loc_data[mask]
                
                filtered_data.append(loc_data)
                print(f"📍 {location}: {len(loc_data)} records ({date_range['start']} to {date_range['end']})")
        
        self.raw_data = pd.concat(filtered_data, ignore_index=True)
        return self
    
    def initial_cleaning(self):
        """🧹 Tahap 3: Pembersihan awal dengan constraint preservation"""
        print("🔄 Initial data cleaning with physical constraints...")
        
        # Store original stats
        print("📊 Original data statistics:")
        for var in ['SST', 'RAD']:
            series = self.raw_data[var].dropna()
            if len(series) > 0:
                print(f"   {var}: min={series.min():.2f}, max={series.max():.2f}, "
                      f"mean={series.mean():.2f}, violations_min={sum(series < self.constraints[var]['min'])}, "
                      f"violations_max={sum(series > self.constraints[var]['max'])}")
        
        # Apply physical constraints - but preserve NaN for legitimate missing data
        for var in ['SST', 'RAD']:
            constraint = self.constraints[var]
            before_clip = (~self.raw_data[var].isnull()).sum()
            
            # Clip extreme values (likely measurement errors)
            self.raw_data[var] = self.raw_data[var].clip(lower=constraint['min'], upper=constraint['max'])
            
            after_clip = (~self.raw_data[var].isnull()).sum()
            if before_clip != after_clip:
                print(f"⚠️ {var}: {before_clip - after_clip} values clipped to physical constraints")
        
        # Remove rows where both SST and RAD are NaN
        initial_len = len(self.raw_data)
        self.raw_data = self.raw_data.dropna(subset=['SST', 'RAD'], how='all')
        final_len = len(self.raw_data)
        
        print(f"✅ Removed {initial_len - final_len} completely empty records")
        
        # Sort data
        self.raw_data = self.raw_data.sort_values(['Location', 'Date']).reset_index(drop=True)
        
        return self
    
    def create_daily_grid(self):
        """🔄 Tahap 4: Resampling harian dan create complete date grid"""
        print("🔄 Creating daily grid...")
        
        complete_data = []
        
        for location, date_range in self.location_ranges.items():
            if location in self.raw_data['Location'].values:
                # Create complete date range for this location
                date_range_complete = pd.date_range(
                    start=date_range['start'], 
                    end=date_range['end'], 
                    freq='D'
                )
                
                # Create complete grid
                complete_grid = pd.DataFrame({
                    'Date': date_range_complete,
                    'Location': location
                })
                
                # Get existing data for this location
                loc_data = self.raw_data[self.raw_data['Location'] == location]
                
                # Merge to create complete dataset with NaNs where missing
                merged = complete_grid.merge(loc_data, on=['Date', 'Location'], how='left')
                complete_data.append(merged)
                
                missing_count = merged[['SST', 'RAD']].isnull().sum().sum()
                total_possible = len(merged) * 2
                print(f"📍 {location}: {missing_count}/{total_possible} missing values ({missing_count/total_possible*100:.1f}%)")
        
        self.processed_data = pd.concat(complete_data, ignore_index=True)
        return self
    
    def constrained_interpolation(self, series, method='linear', var_name='Unknown'):
        """
        Physics-aware interpolation that respects physical constraints
        """
        if series.isnull().sum() == 0:
            return series
        
        constraint = self.constraints.get(var_name, {'min': -np.inf, 'max': np.inf})
        
        # Standard interpolation
        if method == 'linear':
            interpolated = series.interpolate(method='linear')
        elif method == 'cubic':
            interpolated = series.interpolate(method='cubic')
        else:
            interpolated = series.interpolate(method=method)
        
        # Apply constraints to interpolated values only
        original_mask = ~series.isnull()  # Original non-NaN values
        interpolated_mask = ~interpolated.isnull() & series.isnull()  # Newly interpolated values
        
        # Clip only the interpolated values
        clipped_interpolated = interpolated.copy()
        clipped_interpolated[interpolated_mask] = np.clip(
            interpolated[interpolated_mask], 
            constraint['min'], 
            constraint['max']
        )
        
        violations = sum((interpolated[interpolated_mask] < constraint['min']) | 
                        (interpolated[interpolated_mask] > constraint['max']))
        
        if violations > 0:
            print(f"      ⚠️ Clipped {violations} interpolated {var_name} values to physical constraints")
        
        return clipped_interpolated
    
    def seasonal_interpolation_with_constraints(self, series, location, var_name):
        """
        Seasonal interpolation with physical constraints
        """
        available_data = series.dropna()
        if len(available_data) < 730:  # Less than 2 years
            print(f"    ⚠️ Not enough data for seasonal decomposition, using constrained cubic")
            return self.constrained_interpolation(series, 'cubic', var_name)
        
        try:
            # Pre-fill with constrained linear interpolation
            temp_series = self.constrained_interpolation(series, 'linear', var_name)
            
            # Apply STL decomposition
            stl = STL(temp_series, seasonal=365, robust=True, period=365)
            decomposition = stl.fit()
            
            # Use decomposed components to fill original missing values
            filled_series = series.copy()
            missing_mask = series.isnull()
            
            # Combine trend + seasonal for missing values
            seasonal_trend = decomposition.trend + decomposition.seasonal
            filled_series[missing_mask] = seasonal_trend[missing_mask]
            
            # Apply constraints to the seasonally imputed values
            constraint = self.constraints.get(var_name, {'min': -np.inf, 'max': np.inf})
            filled_series[missing_mask] = np.clip(
                filled_series[missing_mask],
                constraint['min'],
                constraint['max']
            )
            
            return filled_series
        
        except Exception as e:
            print(f"    ⚠️ STL failed for {location}, falling back to constrained cubic: {str(e)}")
            return self.constrained_interpolation(series, 'cubic', var_name)
    
    def smart_gap_filling_with_constraints(self, series, location, variable):
        """
        Smart gap filling strategy with physical constraints
        """
        if series.isnull().sum() == 0:
            return series
        
        print(f"    📊 Analyzing gaps for {variable}...")
        
        # Identify individual gaps
        is_missing = series.isnull()
        gaps = []
        gap_start = None
        
        for i, missing in enumerate(is_missing):
            if missing and gap_start is None:
                gap_start = i
            elif not missing and gap_start is not None:
                gaps.append((gap_start, i-1, i-gap_start))
                gap_start = None
        
        if gap_start is not None:
            gaps.append((gap_start, len(series)-1, len(series)-gap_start))
        
        if not gaps:
            return series
        
        print(f"    📊 Found {len(gaps)} gaps, sizes: {[g[2] for g in gaps]}")
        
        filled_series = series.copy()
        available_years = len(series) / 365.25
        
        # Process each gap individually with constraints
        for gap_start, gap_end, gap_length in gaps:
            print(f"    🔧 Filling gap: days {gap_start}-{gap_end} (length: {gap_length})")
            
            if gap_length <= 3:
                method = "constrained_linear"
            elif gap_length <= 7:
                method = "constrained_cubic"
            elif gap_length <= 30 and available_years >= 2:
                method = "constrained_seasonal"
            else:
                method = "constrained_cubic"
            
            # Extract gap region with buffer
            buffer_size = min(30, max(gap_length, 7))
            gap_region = filled_series.iloc[max(0, gap_start-buffer_size):min(len(filled_series), gap_end+buffer_size)]
            
            if method == "constrained_linear":
                gap_region_filled = self.constrained_interpolation(gap_region, 'linear', variable)
            elif method == "constrained_cubic":
                gap_region_filled = self.constrained_interpolation(gap_region, 'cubic', variable)
            elif method == "constrained_seasonal":
                gap_region_filled = self.seasonal_interpolation_with_constraints(gap_region, location, variable)
            
            # Update the filled series
            filled_series.iloc[max(0, gap_start-buffer_size):min(len(filled_series), gap_end+buffer_size)] = gap_region_filled
            
            print(f"    ✅ Gap filled using {method}")
        
        return filled_series
    
    def hybrid_imputation(self):
        """🧩 Tahap 5: Physics-aware hybrid imputation"""
        print("🔄 Applying physics-aware hybrid imputation strategy...")
        
        imputed_data = []
        
        for location in self.processed_data['Location'].unique():
            print(f"📍 Processing {location}...")
            
            loc_data = self.processed_data[self.processed_data['Location'] == location].copy()
            loc_data = loc_data.set_index('Date').sort_index()
            
            available_years = (loc_data.index.max() - loc_data.index.min()).days / 365.25
            print(f"    📊 Available timespan: {available_years:.1f} years")
            
            # Process each variable with constraints
            for var in ['SST', 'RAD']:
                print(f"  🔄 Imputing {var} with physical constraints...")
                
                series = loc_data[var].copy()
                missing_pct = series.isnull().sum() / len(series) * 100
                print(f"    📊 Missing: {series.isnull().sum()}/{len(series)} ({missing_pct:.1f}%)")
                
                if series.isnull().sum() == 0:
                    print(f"    ✅ No missing values for {var}")
                    continue
                
                # Use constrained gap filling
                filled_series = self.smart_gap_filling_with_constraints(series, location, var)
                
                # Final constraint check and cleanup
                if filled_series.isnull().sum() > 0:
                    print(f"    🔧 Final cleanup: {filled_series.isnull().sum()} remaining NaNs")
                    filled_series = self.constrained_interpolation(filled_series, 'linear', var)
                    
                    # Forward/backward fill if still needed
                    if filled_series.isnull().sum() > 0:
                        filled_series = filled_series.fillna(method='ffill').fillna(method='bfill')
                
                # Final constraint enforcement
                constraint = self.constraints[var]
                violations_before = sum((filled_series < constraint['min']) | (filled_series > constraint['max']))
                filled_series = np.clip(filled_series, constraint['min'], constraint['max'])
                violations_after = sum((filled_series < constraint['min']) | (filled_series > constraint['max']))
                
                if violations_before > 0:
                    print(f"    🔧 Final constraint enforcement: {violations_before} violations corrected")
                
                loc_data[var] = filled_series
                final_missing = loc_data[var].isnull().sum()
                print(f"    ✅ {var} imputation complete. Remaining NaNs: {final_missing}")
            
            # Reset index and add location back
            loc_data = loc_data.reset_index()
            loc_data['Location'] = location
            imputed_data.append(loc_data)
        
        self.processed_data = pd.concat(imputed_data, ignore_index=True)
        print("✅ Physics-aware hybrid imputation completed!")
        return self
    
    def feature_engineering(self):
        """📊 Tahap 6: Feature engineering"""
        print("🔄 Adding temporal features...")
        
        # Add temporal features
        self.processed_data['DayOfYear'] = self.processed_data['Date'].dt.dayofyear
        self.processed_data['Year'] = self.processed_data['Date'].dt.year
        self.processed_data['Month'] = self.processed_data['Date'].dt.month
        self.processed_data['Quarter'] = self.processed_data['Date'].dt.quarter
        
        # Add seasonal indicators
        self.processed_data['Season'] = self.processed_data['Month'].map({
            12: 'DJF', 1: 'DJF', 2: 'DJF',
            3: 'MAM', 4: 'MAM', 5: 'MAM',  
            6: 'JJA', 7: 'JJA', 8: 'JJA',
            9: 'SON', 10: 'SON', 11: 'SON'
        })
        
        print("✅ Temporal features added")
        return self
    
    def validate_physical_constraints(self):
        """🔍 Comprehensive physical constraint validation"""
        print("🔄 Validating physical constraints...")
        
        validation_passed = True
        
        for var in ['SST', 'RAD']:
            constraint = self.constraints[var]
            series = self.processed_data[var]
            
            # Check for constraint violations
            below_min = sum(series < constraint['min'])
            above_max = sum(series > constraint['max'])
            nan_count = series.isnull().sum()
            
            print(f"📊 {var} ({constraint['name']}):")
            print(f"    Range: [{constraint['min']}, {constraint['max']}]")
            print(f"    Data range: [{series.min():.2f}, {series.max():.2f}]")
            print(f"    Below minimum: {below_min}")
            print(f"    Above maximum: {above_max}")
            print(f"    Missing values: {nan_count}")
            
            if below_min > 0 or above_max > 0 or nan_count > 0:
                validation_passed = False
                
            # Special validation for RAD (should never be negative)
            if var == 'RAD':
                negative_count = sum(series < 0)
                print(f"    ❌ CRITICAL - Negative values: {negative_count}")
                if negative_count > 0:
                    print(f"    🔧 Fixing negative RAD values...")
                    self.processed_data[var] = np.clip(self.processed_data[var], 0, None)
                    print(f"    ✅ Negative RAD values corrected")
        
        if validation_passed:
            print("✅ All physical constraints validated successfully!")
        else:
            print("⚠️ Some constraint violations found and corrected")
        
        return self
    
    def quality_control(self):
        """🔍 Enhanced post-imputation quality control"""
        print("🔄 Performing enhanced quality control...")
        
        # Validate physical constraints first
        self.validate_physical_constraints()
        
        # Check for any remaining missing values
        missing_counts = self.processed_data[['SST', 'RAD']].isnull().sum()
        print(f"📊 Remaining missing values: SST={missing_counts['SST']}, RAD={missing_counts['RAD']}")
        
        # Statistical outlier detection (but don't automatically remove)
        for var in ['SST', 'RAD']:
            series = self.processed_data[var]
            q1, q99 = series.quantile([0.01, 0.99])
            outliers = ((series < q1) | (series > q99)).sum()
            print(f"📊 {var} statistical outliers (1st-99th percentile): {outliers}")
        
        return self
    
    def generate_summary_report(self):
        """📋 Generate comprehensive summary report"""
        print("\n" + "="*70)
        print("📋 PHYSICS-AWARE PREPROCESSING SUMMARY REPORT")
        print("="*70)
        
        # Basic statistics
        print("\n📊 FINAL DATASET OVERVIEW:")
        print(f"Total records: {len(self.processed_data):,}")
        print(f"Date range: {self.processed_data['Date'].min()} to {self.processed_data['Date'].max()}")
        print(f"Locations: {', '.join(self.processed_data['Location'].unique())}")
        
        # Physical constraint validation summary
        print("\n🔍 PHYSICAL CONSTRAINT VALIDATION:")
        for var in ['SST', 'RAD']:
            constraint = self.constraints[var]
            series = self.processed_data[var]
            violations = sum((series < constraint['min']) | (series > constraint['max']))
            print(f"{var}: {violations} constraint violations (should be 0)")
            
            # Special check for RAD negative values
            if var == 'RAD':
                negative = sum(series < 0)
                print(f"RAD negative values: {negative} (CRITICAL - should be 0)")
        
        # Missing value status
        print("\n🔍 MISSING VALUE STATUS:")
        for location in self.processed_data['Location'].unique():
            loc_data = self.processed_data[self.processed_data['Location'] == location]
            sst_missing = loc_data['SST'].isnull().sum()
            rad_missing = loc_data['RAD'].isnull().sum()
            total_records = len(loc_data)
            print(f"{location}: SST={sst_missing}/{total_records} ({sst_missing/total_records*100:.2f}%), "
                  f"RAD={rad_missing}/{total_records} ({rad_missing/total_records*100:.2f}%)")
        
        # Statistical summary
        print("\n📈 STATISTICAL SUMMARY:")
        summary_stats = self.processed_data.groupby('Location')[['SST', 'RAD']].describe()
        print(summary_stats.round(3))
        
        return self
    
    def save_processed_data(self, output_path='buoys_preprocessed_fixed.csv'):
        """💾 Tahap 7: Save processed data dengan final validation"""
        print(f"🔄 Saving processed data to {output_path}...")
        
        # Final validation before saving
        print("🔍 Final pre-save validation...")
        rad_negative = sum(self.processed_data['RAD'] < 0)
        if rad_negative > 0:
            print(f"❌ CRITICAL ERROR: Found {rad_negative} negative RAD values before saving!")
            print("🔧 Applying emergency correction...")
            self.processed_data['RAD'] = np.clip(self.processed_data['RAD'], 0, None)
            print("✅ Emergency correction applied")
        
        # Select final columns
        final_columns = ['Date', 'Location', 'SST', 'RAD', 'DayOfYear', 'Year', 'Month', 'Quarter', 'Season']
        final_data = self.processed_data[final_columns].copy()
        
        # Save to CSV
        final_data.to_csv(output_path, index=False)
        print(f"✅ Data saved: {len(final_data)} records")
        
        # Post-save validation
        print("🔍 Post-save validation...")
        saved_data = pd.read_csv(output_path)
        rad_negative_saved = sum(saved_data['RAD'] < 0)
        print(f"✅ Saved file validation: {rad_negative_saved} negative RAD values (should be 0)")
        
        return final_data
    
    def run_full_pipeline(self, output_path='buoys_preprocessed_fixed.csv'):
        """🚀 Run complete preprocessing pipeline dengan physics-aware processing"""
        print("🚀 Starting physics-aware buoy data preprocessing...")
        print("="*70)
        
        try:
            # Execute all steps
            (self.load_and_prepare()
             .apply_location_cutoffs()
             .initial_cleaning()
             .create_daily_grid()
             .hybrid_imputation()
             .feature_engineering()
             .quality_control()
             .generate_summary_report())
            
            # Save final data
            final_data = self.save_processed_data(output_path)
            
            print("\n🎉 PHYSICS-AWARE PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*70)
            
            return final_data
            
        except Exception as e:
            print(f"❌ Error during preprocessing: {str(e)}")
            raise e

# Example usage with enhanced validation
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = BuoyDataPreprocessor('Buoys_Data_All.csv')
    
    # Run full pipeline
    processed_data = preprocessor.run_full_pipeline('buoys_preprocessed_fixed.csv')
    
    # Additional validation
    print("\n🔍 FINAL VALIDATION CHECK:")
    print(f"RAD negative values: {sum(processed_data['RAD'] < 0)}")
    print(f"RAD range: [{processed_data['RAD'].min():.3f}, {processed_data['RAD'].max():.3f}]")
    print(f"SST range: [{processed_data['SST'].min():.3f}, {processed_data['SST'].max():.3f}]")
    
    # Create validation plot
    print("\n📊 Creating validation visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Time series plots
    for i, var in enumerate(['SST', 'RAD']):
        for location in processed_data['Location'].unique():
            loc_data = processed_data[processed_data['Location'] == location]
            axes[i, 0].plot(pd.to_datetime(loc_data['Date']), loc_data[var], 
                           label=location, alpha=0.7)
        
        axes[i, 0].set_title(f'{var} Time Series - All Locations')
        axes[i, 0].set_ylabel(var)
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Add constraint lines
        if var in preprocessor.constraints:
            constraint = preprocessor.constraints[var]
            axes[i, 0].axhline(y=constraint['min'], color='red', linestyle='--', alpha=0.5, label=f'Min ({constraint["min"]})')
            axes[i, 0].axhline(y=constraint['max'], color='red', linestyle='--', alpha=0.5, label=f'Max ({constraint["max"]})')
    
    # Distribution plots
    for i, var in enumerate(['SST', 'RAD']):
        processed_data.boxplot(column=var, by='Location', ax=axes[i, 1])
        axes[i, 1].set_title(f'{var} Distribution by Location')
        axes[i, 1].set_xlabel('Location')
        axes[i, 1].set_ylabel(var)
        
        # Add constraint lines
        if var in preprocessor.constraints:
            constraint = preprocessor.constraints[var]
            axes[i, 1].axhline(y=constraint['min'], color='red', linestyle='--', alpha=0.5)
            axes[i, 1].axhline(y=constraint['max'], color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('buoys_physics_aware_validation.png', dpi=300, bbox_inches='tight')
    print("✅ Validation visualization saved as 'buoys_physics_aware_validation.png'")
    
    plt.show()