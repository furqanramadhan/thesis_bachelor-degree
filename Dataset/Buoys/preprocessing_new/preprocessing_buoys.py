import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from datetime import datetime
import glob

def load_and_combine_buoy_data(location_code, data_dir):
    """
    Load and combine all buoy data files for a specific location.
    
    Parameters:
    -----------
    location_code : str
        Location identifier (e.g., '0N90E')
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames for each variable type
    """
    print(f"Loading data for buoy location {location_code}...")
    
    # Define variable types and their corresponding filenames
    var_types = {
        'radiation': f'rad{location_code.lower()}',
        'rainfall': f'rain{location_code.lower()}',
        'humidity': f'rh{location_code.lower()}',
        'sst': f'sst{location_code.lower()}',
        'temperature': f't{location_code.lower()}',
        'wind': f'w{location_code.lower()}'
    }
    
    data_dict = {}
    
    # Load each variable type
    for var_type, file_prefix in var_types.items():
        # Look for matching files (could be .csv, .txt, etc.)
        file_pattern = os.path.join(data_dir, f"{file_prefix}*")
        matching_files = glob.glob(file_pattern)
        
        if matching_files:
            file_path = matching_files[0]
            try:
                # Load the file
                df = pd.read_csv(file_path)
                print(f"Successfully loaded {var_type} data from {file_path}")
                
                # Convert date column to datetime
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                
                # Store in dictionary
                data_dict[var_type] = df
                
            except Exception as e:
                print(f"Error loading {var_type} data: {e}")
        else:
            print(f"No {var_type} data file found matching pattern: {file_pattern}")
    
    return data_dict

def preprocess_buoy_data(data_dict, location_code, output_dir="output", cleaned_dir="cleaned"):
    """
    Preprocess all buoy data variables.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing DataFrames for each variable type
    location_code : str
        Location identifier (e.g., '0N90E')
    output_dir : str
        Directory for saving visualization outputs
    cleaned_dir : str
        Directory for saving cleaned data
        
    Returns:
    --------
    dict
        Dictionary containing cleaned DataFrames
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    loc_output_dir = os.path.join(output_dir, location_code)
    os.makedirs(loc_output_dir, exist_ok=True)
    
    cleaned_data = {}
    
    # Process radiation data
    if 'radiation' in data_dict:
        print("\nProcessing Short Wave Radiation data...")
        df_rad = data_dict['radiation'].copy()
        
        # Apply quality filter
        if 'Q' in df_rad.columns:
            df_rad_clean = df_rad[df_rad['Q'] > 0].copy()
            print(f"Filtered radiation data by quality: {len(df_rad_clean)}/{len(df_rad)} rows kept")
        else:
            df_rad_clean = df_rad.copy()
            print("No quality column found in radiation data")
        
        # Process SWRad variable
        if 'SWRad' in df_rad_clean.columns:
            # Handle outliers
            df_rad_clean = handle_outliers(df_rad_clean, 'SWRad')
            
            # Visualize
            plot_time_series(df_rad_clean, 'SWRad', 'Short Wave Radiation', 'W/m²', location_code, loc_output_dir)
            plot_seasonal_patterns(df_rad_clean, 'SWRad', 'Short Wave Radiation', 'W/m²', location_code, loc_output_dir)
            plot_annual_trends(df_rad_clean, 'SWRad', 'Short Wave Radiation', 'W/m²', location_code, loc_output_dir)
            
            # Store cleaned data
            cleaned_data['SWRad'] = df_rad_clean[['SWRad']].copy()
            
            # Save to CSV
            df_rad_clean.to_csv(f"{cleaned_dir}/{location_code}_SWRad_clean.csv")
            print(f"Saved cleaned radiation data to {cleaned_dir}/{location_code}_SWRad_clean.csv")
    
    # Process rainfall data
    if 'rainfall' in data_dict:
        print("\nProcessing Rainfall data...")
        df_rain = data_dict['rainfall'].copy()
        
        # Apply quality filter
        if 'Q' in df_rain.columns:
            df_rain_clean = df_rain[df_rain['Q'] > 0].copy()
            print(f"Filtered rainfall data by quality: {len(df_rain_clean)}/{len(df_rain)} rows kept")
        else:
            df_rain_clean = df_rain.copy()
            print("No quality column found in rainfall data")
        
        # Process Prec variable
        if 'Prec' in df_rain_clean.columns:
            # Handle outliers
            df_rain_clean = handle_outliers(df_rain_clean, 'Prec')
            
            # Visualize
            plot_time_series(df_rain_clean, 'Prec', 'Rainfall', 'mm', location_code, loc_output_dir)
            plot_seasonal_patterns(df_rain_clean, 'Prec', 'Rainfall', 'mm', location_code, loc_output_dir)
            plot_annual_trends(df_rain_clean, 'Prec', 'Rainfall', 'mm', location_code, loc_output_dir)
            
            # Store cleaned data
            cleaned_data['Prec'] = df_rain_clean[['Prec']].copy()
            
            # Save to CSV
            df_rain_clean.to_csv(f"{cleaned_dir}/{location_code}_Prec_clean.csv")
            print(f"Saved cleaned rainfall data to {cleaned_dir}/{location_code}_Prec_clean.csv")
    
    # Process humidity data
    if 'humidity' in data_dict:
        print("\nProcessing Relative Humidity data...")
        df_rh = data_dict['humidity'].copy()
        
        # Apply quality filter
        if 'Q' in df_rh.columns:
            df_rh_clean = df_rh[df_rh['Q'] > 0].copy()
            print(f"Filtered humidity data by quality: {len(df_rh_clean)}/{len(df_rh)} rows kept")
        else:
            df_rh_clean = df_rh.copy()
            print("No quality column found in humidity data")
        
        # Process RH variable
        if 'RH' in df_rh_clean.columns:
            # Handle outliers
            df_rh_clean = handle_outliers(df_rh_clean, 'RH')
            
            # Visualize
            plot_time_series(df_rh_clean, 'RH', 'Relative Humidity', '%', location_code, loc_output_dir)
            plot_seasonal_patterns(df_rh_clean, 'RH', 'Relative Humidity', '%', location_code, loc_output_dir)
            plot_annual_trends(df_rh_clean, 'RH', 'Relative Humidity', '%', location_code, loc_output_dir)
            
            # Store cleaned data
            cleaned_data['RH'] = df_rh_clean[['RH']].copy()
            
            # Save to CSV
            df_rh_clean.to_csv(f"{cleaned_dir}/{location_code}_RH_clean.csv")
            print(f"Saved cleaned humidity data to {cleaned_dir}/{location_code}_RH_clean.csv")
    
    # Process SST data
    if 'sst' in data_dict:
        print("\nProcessing Sea Surface Temperature data...")
        df_sst = data_dict['sst'].copy()
        
        # Apply quality filter
        if 'Q' in df_sst.columns:
            df_sst_clean = df_sst[df_sst['Q'] > 0].copy()
            print(f"Filtered SST data by quality: {len(df_sst_clean)}/{len(df_sst)} rows kept")
        else:
            df_sst_clean = df_sst.copy()
            print("No quality column found in SST data")
        
        # Process SST variable
        if 'SST' in df_sst_clean.columns:
            # Handle outliers
            df_sst_clean = handle_outliers(df_sst_clean, 'SST')
            
            # Visualize
            plot_time_series(df_sst_clean, 'SST', 'Sea Surface Temperature', '°C', location_code, loc_output_dir)
            plot_seasonal_patterns(df_sst_clean, 'SST', 'Sea Surface Temperature', '°C', location_code, loc_output_dir)
            plot_annual_trends(df_sst_clean, 'SST', 'Sea Surface Temperature', '°C', location_code, loc_output_dir)
            
            # Store cleaned data
            cleaned_data['SST'] = df_sst_clean[['SST']].copy()
            
            # Save to CSV
            df_sst_clean.to_csv(f"{cleaned_dir}/{location_code}_SST_clean.csv")
            print(f"Saved cleaned SST data to {cleaned_dir}/{location_code}_SST_clean.csv")
    
    # Process temperature profile data
    if 'temperature' in data_dict:
        print("\nProcessing Temperature Profile data...")
        df_temp = data_dict['temperature'].copy()
        
        # Find temperature columns (look for columns with 'TEMP' prefix)
        temp_cols = [col for col in df_temp.columns if col.startswith('TEMP_')]
        
        if temp_cols:
            print(f"Found {len(temp_cols)} temperature depth measurements")
            
            # Process each depth
            for col in temp_cols:
                # Extract depth from column name
                depth = col.split('_')[1].replace('m', '')
                print(f"Processing temperature at depth {depth}m")
                
                if pd.api.types.is_numeric_dtype(df_temp[col]):
                    # Handle outliers
                    df_temp = handle_outliers(df_temp, col)
                    
                    # Visualize
                    plot_time_series(df_temp, col, f'Water Temperature {depth}m', '°C', location_code, loc_output_dir)
                    
                    # Store cleaned data
                    cleaned_data[f'TEMP_{depth}'] = df_temp[[col]].copy()
            
            # Create a simplified dataframe with selected depths (if needed)
            selected_depths = ['10.0m', '100.0m', '300.0m'] if len(temp_cols) > 3 else temp_cols
            selected_cols = [f'TEMP_{depth}' for depth in selected_depths if f'TEMP_{depth}' in df_temp.columns]
            
            if selected_cols:
                # Plot temperature profiles
                plot_temperature_profile(df_temp, selected_cols, location_code, loc_output_dir)
            
            # Save to CSV
            df_temp_clean = df_temp[temp_cols].copy()
            df_temp_clean.to_csv(f"{cleaned_dir}/{location_code}_TEMP_clean.csv")
            print(f"Saved cleaned temperature profile data to {cleaned_dir}/{location_code}_TEMP_clean.csv")
    
    # Process wind data
    if 'wind' in data_dict:
        print("\nProcessing Wind data...")
        df_wind = data_dict['wind'].copy()
        
        # Process wind components
        for col in ['UWND', 'VWND', 'WSPD', 'WDIR']:
            if col in df_wind.columns:
                # Handle missing values
                missing_count = df_wind[col].isna().sum()
                if missing_count > 0:
                    print(f"Found {missing_count} missing values in {col} ({missing_count/len(df_wind)*100:.2f}%)")
                
                # Handle outliers
                df_wind = handle_outliers(df_wind, col)
        
        # Visualize wind speed
        if 'WSPD' in df_wind.columns:
            plot_time_series(df_wind, 'WSPD', 'Wind Speed', 'm/s', location_code, loc_output_dir)
            plot_seasonal_patterns(df_wind, 'WSPD', 'Wind Speed', 'm/s', location_code, loc_output_dir)
            plot_annual_trends(df_wind, 'WSPD', 'Wind Speed', 'm/s', location_code, loc_output_dir)
            
            # Store cleaned data
            cleaned_data['WSPD'] = df_wind[['WSPD']].copy()
        
        # Wind direction visualization (if both components available)
        if 'UWND' in df_wind.columns and 'VWND' in df_wind.columns:
            plot_wind_rose(df_wind, location_code, loc_output_dir)
        
        # Save to CSV
        wind_cols = [col for col in ['UWND', 'VWND', 'WSPD', 'WDIR'] if col in df_wind.columns]
        df_wind_clean = df_wind[wind_cols].copy()
        df_wind_clean.to_csv(f"{cleaned_dir}/{location_code}_WIND_clean.csv")
        print(f"Saved cleaned wind data to {cleaned_dir}/{location_code}_WIND_clean.csv")
    
    # Create a combined dataset with key variables
    print("\nCreating combined dataset...")
    combine_key_variables(cleaned_data, location_code, cleaned_dir)
    
    return cleaned_data

def handle_outliers(df, variable, method='zscore', threshold=3):
    """Handle outliers in the specified variable."""
    df_result = df.copy()
    
    # Skip if variable doesn't exist or is non-numeric
    if variable not in df_result.columns:
        return df_result
        
    if not pd.api.types.is_numeric_dtype(df_result[variable]):
        print(f"Skipping outlier detection for non-numeric variable: {variable}")
        return df_result
    
    # Skip if too many NaN values
    nan_count = df_result[variable].isna().sum()
    if nan_count > len(df_result) * 0.5:
        print(f"Skipping outlier detection for {variable}: too many NaN values ({nan_count} / {len(df_result)})")
        return df_result
    
    # Get original count
    valid_data = df_result[variable].dropna()
    original_count = len(valid_data)
    
    if original_count == 0:
        return df_result
    
    # Detect outliers
    try:
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(valid_data))
            outliers = z_scores > threshold
            outlier_indices = valid_data.index[outliers]
        elif method == 'iqr':
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_indices = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)].index
        else:
            print(f"Unknown outlier detection method: {method}")
            return df_result
    except Exception as e:
        print(f"Error detecting outliers for {variable}: {e}")
        return df_result
    
    # Mark outliers
    if len(outlier_indices) > 0:
        print(f"Detected {len(outlier_indices)} outliers in {variable} ({len(outlier_indices)/original_count*100:.2f}%)")
        
        # Create an 'is_outlier_[var]' column
        outlier_col = f"is_outlier_{variable.replace('(', '').replace(')', '').replace('.', '_')}"
        df_result[outlier_col] = False
        df_result.loc[outlier_indices, outlier_col] = True
        
        # For modeling preparation, we might want to replace outliers with NaN
        # rather than removing them, so the time series structure is preserved
        df_result.loc[outlier_indices, variable] = np.nan
        
        print(f"Marked outliers in column '{outlier_col}' and replaced values with NaN")
    else:
        print(f"No outliers detected in {variable}")
    
    return df_result

def plot_time_series(df, variable, var_name, unit, location_code, output_dir):
    """Plot time series for a variable."""
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[variable], 'o-', alpha=0.5, markersize=2)
    
    plt.title(f'{var_name} at {location_code}')
    plt.ylabel(f'{var_name} ({unit})')
    plt.xlabel('Date')
    plt.grid(True)
    
    # Add a 30-day rolling average to show trend
    if len(df) > 30:
        valid_data = df[variable].dropna()
        if len(valid_data) > 30:
            rolling_avg = valid_data.rolling(window=30, center=True).mean()
            plt.plot(valid_data.index, rolling_avg, 'r-', linewidth=2, label='30-day Rolling Average')
            plt.legend()
    
    plt.tight_layout()
    var_file = variable.replace('(', '').replace(')', '').replace('.', '_')
    plt.savefig(f'{output_dir}/{var_file}_time_series.png')
    plt.close()

def plot_seasonal_patterns(df, variable, var_name, unit, location_code, output_dir):
    """Plot seasonal patterns for a variable."""
    # Skip if not enough data
    if len(df) < 30:
        print(f"Skipping seasonal analysis for {variable}: insufficient data")
        return
    
    # Resample to monthly data
    try:
        monthly_data = df[variable].resample('M').mean()
        
        # Create month and year columns
        monthly_df = pd.DataFrame(monthly_data)
        monthly_df['month'] = monthly_df.index.month
        monthly_df['year'] = monthly_df.index.year
        
        # Plot monthly patterns
        monthly_pattern = monthly_df.groupby('month')[variable].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        plt.figure(figsize=(12, 6))
        monthly_pattern.plot(kind='bar')
        plt.title(f'Monthly {var_name} Pattern at {location_code}')
        plt.ylabel(f'{var_name} ({unit})')
        plt.xlabel('Month')
        plt.xticks(np.arange(12), months, rotation=45)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        var_file = variable.replace('(', '').replace(')', '').replace('.', '_')
        plt.savefig(f'{output_dir}/{var_file}_monthly_pattern.png')
        plt.close()
        
        # Boxplot of monthly values (showing variation within each month)
        if len(monthly_df) >= 12:  # Only if we have enough data
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='month', y=variable, data=monthly_df)
            plt.title(f'Monthly {var_name} Distribution at {location_code}')
            plt.ylabel(f'{var_name} ({unit})')
            plt.xlabel('Month')
            plt.xticks(np.arange(12), months, rotation=45)
            plt.grid(True, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{var_file}_monthly_boxplot.png')
            plt.close()
    except Exception as e:
        print(f"Error in seasonal analysis for {variable}: {e}")

def plot_annual_trends(df, variable, var_name, unit, location_code, output_dir):
    """Plot annual trends for a variable."""
    # Skip if not enough data
    if len(df) < 365:
        print(f"Skipping annual trend analysis for {variable}: insufficient data")
        return
    
    try:
        # Resample to annual data
        annual_data = df[variable].resample('Y').mean()
        
        # Skip if we have too few years
        if len(annual_data) < 3:
            print(f"Skipping annual trend analysis for {variable}: less than 3 years of data")
            return
        
        # Plot annual trend
        plt.figure(figsize=(14, 6))
        annual_data.plot()
        
        plt.title(f'Annual {var_name} Trend at {location_code}')
        plt.ylabel(f'{var_name} ({unit})')
        plt.xlabel('Year')
        plt.grid(True)
        
        # Add trend line
        years_numeric = np.arange(len(annual_data))
        
        # Only calculate trend if we have enough valid data points
        valid_data = annual_data.dropna()
        if len(valid_data) >= 3:
            numeric_idx = np.arange(len(valid_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(numeric_idx, valid_data)
            
            trend_line = intercept + slope * numeric_idx
            plt.plot(valid_data.index, trend_line, 'r--', 
                    label=f'Trend: {slope:.4f} per year (p={p_value:.4f}, R²={r_value**2:.4f})')
            plt.legend()
        
        plt.tight_layout()
        var_file = variable.replace('(', '').replace(')', '').replace('.', '_')
        plt.savefig(f'{output_dir}/{var_file}_annual_trend.png')
        plt.close()
    except Exception as e:
        print(f"Error in annual trend analysis for {variable}: {e}")

def plot_temperature_profile(df, temp_cols, location_code, output_dir):
    """Plot temperature profile at different depths."""
    try:
        # Get average temperature at each depth
        avg_temps = df[temp_cols].mean()
        depths = [float(col.split('_')[1].replace('m', '')) for col in temp_cols]
        
        # Plot temperature profile
        plt.figure(figsize=(8, 10))
        plt.plot(avg_temps, depths, 'o-', linewidth=2)
        plt.title(f'Average Temperature Profile at {location_code}')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Depth (m)')
        plt.grid(True)
        plt.gca().invert_yaxis()  # Invert y-axis to show depth increasing downward
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temperature_profile.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting temperature profile: {e}")

def plot_wind_rose(df, location_code, output_dir):
    """Plot wind rose diagram using wind components."""
    try:
        # Check if required libraries are installed
        try:
            from windrose import WindroseAxes
        except ImportError:
            print("windrose package not found. Skipping wind rose plot.")
            return
            
        # Skip if not enough data
        if len(df) < 30:
            print("Skipping wind rose plot: insufficient data")
            return
            
        # Calculate wind speed and direction if not available
        if 'WSPD' not in df.columns or 'WDIR' not in df.columns:
            if 'UWND' in df.columns and 'VWND' in df.columns:
                # Calculate wind speed and direction from U and V components
                uwnd = df['UWND'].values
                vwnd = df['VWND'].values
                
                # Skip rows with missing values
                mask = ~(np.isnan(uwnd) | np.isnan(vwnd))
                uwnd = uwnd[mask]
                vwnd = vwnd[mask]
                
                if len(uwnd) < 30:
                    print("Skipping wind rose plot: insufficient valid data")
                    return
                
                wspd = np.sqrt(uwnd**2 + vwnd**2)
                wdir = (270 - np.arctan2(vwnd, uwnd) * 180 / np.pi) % 360
                
                # Create temporary DataFrame with calculated values
                temp_df = pd.DataFrame({
                    'wspd': wspd,
                    'wdir': wdir
                })
            else:
                print("Skipping wind rose plot: required wind components not available")
                return
        else:
            # Use available wind speed and direction
            wspd = df['WSPD'].dropna().values
            wdir = df['WDIR'].dropna().values
            
            # Create temporary DataFrame with values
            temp_df = pd.DataFrame({
                'wspd': wspd,
                'wdir': wdir
            })
        
        # Create wind rose
        plt.figure(figsize=(10, 10))
        ax = WindroseAxes.from_ax()
        ax.bar(temp_df['wdir'], temp_df['wspd'], normed=True, opening=0.8, edgecolor='white')
        ax.set_legend(title='Wind Speed (m/s)')
        plt.title(f'Wind Rose at {location_code}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wind_rose.png')
        plt.close()
    except Exception as e:
        print(f"Error creating wind rose plot: {e}")

def combine_key_variables(cleaned_data, location_code, cleaned_dir):
    """Combine key variables into a single dataset."""
    try:
        # Get key variables
        key_vars = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad']
        available_vars = [var for var in key_vars if var in cleaned_data]
        
        if len(available_vars) <= 1:
            print("Not enough variables available to create combined dataset")
            return
        
        # Combine into single DataFrame
        combined_df = pd.DataFrame()
        
        for var in available_vars:
            if combined_df.empty:
                combined_df = cleaned_data[var].copy()
            else:
                combined_df = combined_df.join(cleaned_data[var], how='outer')
        
        # Save combined dataset
        combined_file = f"{cleaned_dir}/{location_code}_combined_clean.csv"
        combined_df.to_csv(combined_file)
        print(f"Saved combined dataset with {len(available_vars)} variables to {combined_file}")
        
        # Create correlation matrix if we have enough variables
        if len(available_vars) >= 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = combined_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Correlation Matrix for {location_code}')
            plt.tight_layout()
            plt.savefig(f"{cleaned_dir}/{location_code}_correlation_matrix.png")
            plt.close()
            
            print("Created correlation matrix visualization")
    except Exception as e:
        print(f"Error creating combined dataset: {e}")

if __name__ == "__main__":
    # Define data directories for each location
    data_dirs = {
        '0N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/0N90E/CSV',
        '4N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/4N90E/CSV',
        '8N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Buoys/8N90E/CSV'
    }
    
    # Define variable information (name and unit)
    variable_info = {
        'SST': ('Sea Surface Temperature', '°C'),
        'RH': ('Relative Humidity', '%'),
        'Prec': ('Rainfall', 'mm'),
        'WSPD': ('Wind Speed', 'm/s'),
        'SWRad': ('Short Wave Radiation', 'W/m²'),
        'UWND': ('Zonal Wind', 'm/s'),
        'VWND': ('Meridional Wind', 'm/s'),
        'WDIR': ('Wind Direction', '°')
        # Add other variables as needed
    }
    
    # Define temperature columns for profile plotting
    temp_cols = [
        'TEMP_10.0m', 'TEMP_20.0m', 'TEMP_40.0m', 'TEMP_60.0m', 'TEMP_80.0m', 
        'TEMP_100.0m', 'TEMP_120.0m', 'TEMP_140.0m', 'TEMP_180.0m', 
        'TEMP_300.0m', 'TEMP_500.0m'
    ]
    
    # Create timestamp for this preprocessing run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting preprocessing run at {timestamp}")
    
    # Process each location
    for location, data_dir in data_dirs.items():
        print(f"\n{'='*50}")
        print(f"Processing location: {location}")
        print(f"{'='*50}")
        
        # Load data for the location
        data_dict = load_and_combine_buoy_data(location, data_dir)
        
        # Skip if no data was loaded
        if not data_dict:
            print(f"No data found for location {location}. Skipping...")
            continue
        
        # Define output and cleaned directories for this location
        output_dir = os.path.join(data_dir, "../CSV CLEANED")
        cleaned_dir = os.path.join(data_dir, "../CSV CLEANED")
        
        # Ensure the directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cleaned_dir, exist_ok=True)
        
        # Preprocess the data
        cleaned_data = preprocess_buoy_data(data_dict, location, output_dir, cleaned_dir)
        
        # Generate and save plots
        for var_type, df in cleaned_data.items():
            # Check if this DataFrame has variables we can plot
            for variable in df.columns:
                if variable in variable_info:
                    var_name, unit = variable_info[variable]
                    
                    # Generate time series plot
                    plot_time_series(df, variable, var_name, unit, location, output_dir)
                    
                    # Generate seasonal patterns plot
                    plot_seasonal_patterns(df, variable, var_name, unit, location, output_dir)
                    
                    # Generate annual trends plot
                    plot_annual_trends(df, variable, var_name, unit, location, output_dir)
            
            # Check if this is the temperature dataframe and has the needed columns
            if all(col in df.columns for col in temp_cols):
                plot_temperature_profile(df, temp_cols, location, output_dir)
            
            # Check if this is the wind dataframe with required columns
            if all(col in df.columns for col in ['UWND', 'VWND', 'WDIR']):
                plot_wind_rose(df, location, output_dir)
        
        # Check if preprocessing was successful
        if cleaned_data:
            print(f"Successfully processed data for location {location}")
        else:
            print(f"Failed to process data for location {location}")
    
    print(f"\nPreprocessing run completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}")

