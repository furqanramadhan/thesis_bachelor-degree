import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from windrose import WindroseAxes
from datetime import datetime
import matplotlib.dates as mdates
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

def apply_quality_filtering(df, variable_type):
    """
    Filter data based on quality codes (Q) and source codes (S).
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    variable_type : str
        Type of variable (for logging purposes)
        
    Returns:
    --------
    DataFrame
        Filtered DataFrame
    """
    if df is None or df.empty:
        return df
    
    original_length = len(df)
    
    # Make a copy to avoid modifying the original
    df_filtered = df.copy()
    
    # Filter based on quality codes
    if 'Q' in df_filtered.columns:
        # Create a boolean mask for quality filtering
        # Q=0: Missing data - remove
        # Q=1: Highest quality - keep
        # Q=2: Default quality - keep
        # Q=3: Adjusted data - keep but flag
        # Q=4: Lower quality - keep but flag
        # Q=5: Sensor failed - remove
        
        # Remove missing data and failed sensors
        quality_mask = (df_filtered['Q'] > 0) & (df_filtered['Q'] < 5)
        df_filtered = df_filtered[quality_mask].copy()
        
        # Add flag for adjusted data and lower quality
        if 'data_quality' not in df_filtered.columns:
            df_filtered['data_quality'] = 'high'
        
        # Mark adjusted data
        if 'Q' in df_filtered.columns:
            df_filtered.loc[df_filtered['Q'] == 3, 'data_quality'] = 'adjusted'
            df_filtered.loc[df_filtered['Q'] == 4, 'data_quality'] = 'low'
        
        quality_filtered_length = len(df_filtered)
        print(f"Quality filtering for {variable_type}: Kept {quality_filtered_length}/{original_length} rows ({quality_filtered_length/original_length*100:.2f}%)")
    
    # Filter based on source codes
    # Focus on source codes 5 (Recovered from Instrument RAM) as it seems to be the most common and reliable
    if 'S' in df_filtered.columns:
        # Prioritize data from RAM (delayed mode) over telemetry
        df_filtered['source_priority'] = 1  # Default priority
        df_filtered.loc[df_filtered['S'] == 5, 'source_priority'] = 5  # Recovered from RAM (highest)
        df_filtered.loc[df_filtered['S'] == 6, 'source_priority'] = 4  # Derived from RAM
        df_filtered.loc[df_filtered['S'] == 7, 'source_priority'] = 3  # Temporally interpolated from RAM
        df_filtered.loc[df_filtered['S'] == 8, 'source_priority'] = 2  # Spatially interpolated from RAM
        
        # Keep this information for reference but don't filter based on source yet
        # We'll use it when merging/combining data sources
    
    return df_filtered

def handle_missing_values(df, variable_type, variables=None):
    """
    Handle missing values using appropriate strategies for each variable type.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    variable_type : str
        Type of variable ('radiation', 'rainfall', etc.)
    variables : list
        List of specific variables to process (optional)
        
    Returns:
    --------
    DataFrame
        DataFrame with imputed values
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Get the variables to process
    if variables is None:
        # Process all numeric columns except Q and S
        variables = [col for col in df_imputed.columns if col not in ['Q', 'S', 'data_quality', 'source_priority'] 
                    and pd.api.types.is_numeric_dtype(df_imputed[col])]
    
    print(f"Handling missing values for {variable_type} variables: {variables}")
    
    # Apply specific imputation strategies based on variable type
    if variable_type == 'radiation':
        # For radiation data (SWRad, StDev, Max) - use time-based interpolation
        for var in variables:
            missing_count = df_imputed[var].isna().sum()
            if missing_count > 0:
                print(f"  Imputing {missing_count} missing values for {var}")
                
                # First try time-based interpolation (for short gaps)
                df_imputed[var] = df_imputed[var].interpolate(method='time', limit=3)
                
                # For remaining gaps, use forward fill with a limit
                remaining_missing = df_imputed[var].isna().sum()
                if remaining_missing > 0:
                    df_imputed[var] = df_imputed[var].fillna(method='ffill', limit=2)
                    
                # Calculate how many were filled
                final_missing = df_imputed[var].isna().sum()
                filled_count = missing_count - final_missing
                print(f"    Filled {filled_count}/{missing_count} values. {final_missing} remain missing.")
    
    elif variable_type == 'rainfall':
        # For rainfall (Prec) - missing values often mean no rain
        for var in variables:
            if var == 'Prec':
                missing_count = df_imputed[var].isna().sum()
                if missing_count > 0:
                    print(f"  Imputing {missing_count} missing values for {var} with zeros")
                    df_imputed[var] = df_imputed[var].fillna(0)
            else:
                # For other rainfall-related variables (StDev, %Time)
                missing_count = df_imputed[var].isna().sum()
                if missing_count > 0:
                    print(f"  Imputing {missing_count} missing values for {var}")
                    df_imputed[var] = df_imputed[var].interpolate(method='linear', limit=2)
    
    elif variable_type == 'humidity':
        # For humidity (RH) - tends to be stable, use ffill+bfill
        for var in variables:
            missing_count = df_imputed[var].isna().sum()
            if missing_count > 0:
                print(f"  Imputing {missing_count} missing values for {var}")
                df_imputed[var] = df_imputed[var].fillna(method='ffill', limit=2)
                remaining_missing = df_imputed[var].isna().sum()
                if remaining_missing > 0:
                    df_imputed[var] = df_imputed[var].fillna(method='bfill', limit=2)
                final_missing = df_imputed[var].isna().sum()
                filled_count = missing_count - final_missing
                print(f"    Filled {filled_count}/{missing_count} values. {final_missing} remain missing.")
    
    elif variable_type in ['sst', 'temperature']:
        # For temperature data - linear interpolation in both directions
        for var in variables:
            missing_count = df_imputed[var].isna().sum()
            if missing_count > 0:
                print(f"  Imputing {missing_count} missing values for {var}")
                df_imputed[var] = df_imputed[var].interpolate(method='linear', limit_direction='both', limit=3)
                final_missing = df_imputed[var].isna().sum()
                filled_count = missing_count - final_missing
                print(f"    Filled {filled_count}/{missing_count} values. {final_missing} remain missing.")
    
    elif variable_type == 'wind':
        # For wind data - linear interpolation
        for var in variables:
            missing_count = df_imputed[var].isna().sum()
            if missing_count > 0:
                print(f"  Imputing {missing_count} missing values for {var}")
                df_imputed[var] = df_imputed[var].interpolate(method='linear', limit_direction='both', limit=2)
                final_missing = df_imputed[var].isna().sum()
                filled_count = missing_count - final_missing
                print(f"    Filled {filled_count}/{missing_count} values. {final_missing} remain missing.")
    
    return df_imputed

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
        
        # Apply quality filtering
        df_rad_filtered = apply_quality_filtering(df_rad, "radiation")
        
        # Apply missing value handling
        df_rad_clean = handle_missing_values(df_rad_filtered, "radiation", ['SWRad', 'StDev', 'Max'])
        
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
        
        # Apply quality filtering
        df_rain_filtered = apply_quality_filtering(df_rain, "rainfall")
        
        # Apply missing value handling
        df_rain_clean = handle_missing_values(df_rain_filtered, "rainfall", ['Prec', 'StDev', '%Time'])
        
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
        
        # Apply quality filtering
        df_rh_filtered = apply_quality_filtering(df_rh, "humidity")
        
        # Apply missing value handling
        df_rh_clean = handle_missing_values(df_rh_filtered, "humidity", ['RH'])
        
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
    
    # Process SST data - prioritize sst0n90e.csv over t0n90e.csv for SST
    if 'sst' in data_dict:
        print("\nProcessing Sea Surface Temperature data...")
        df_sst = data_dict['sst'].copy()
        
        # Apply quality filtering
        df_sst_filtered = apply_quality_filtering(df_sst, "sst")
        
        # Apply missing value handling
        df_sst_clean = handle_missing_values(df_sst_filtered, "sst", ['SST'])
        
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
        
        # Remove the SST column since we're using the one from sst0n90e.csv
        if 'SST' in df_temp.columns and 'SST' in cleaned_data:
            print("  Removing duplicate SST column from temperature profile data (using the one from SST data)")
            df_temp = df_temp.drop(columns=['SST'])
        
        # Find temperature columns (look for columns with 'TEMP' prefix)
        temp_cols = [col for col in df_temp.columns if col.startswith('TEMP_')]
        
        if temp_cols:
            print(f"Found {len(temp_cols)} temperature depth measurements")
            
            # Apply quality filtering (if quality columns exist)
            df_temp_filtered = apply_quality_filtering(df_temp, "temperature")
            
            # Apply missing value handling for all temperature columns
            df_temp_clean = handle_missing_values(df_temp_filtered, "temperature", temp_cols)
            
            # Process each depth
            for col in temp_cols:
                # Extract depth from column name
                depth = col.split('_')[1].replace('m', '')
                print(f"Processing temperature at depth {depth}m")
                
                if pd.api.types.is_numeric_dtype(df_temp_clean[col]):
                    # Handle outliers
                    df_temp_clean = handle_outliers(df_temp_clean, col)
                    
                    # Visualize
                    plot_time_series(df_temp_clean, col, f'Water Temperature {depth}m', '°C', location_code, loc_output_dir)
                    
                    # Store cleaned data
                    cleaned_data[f'TEMP_{depth}'] = df_temp_clean[[col]].copy()
            
            # Create a simplified dataframe with selected depths (if needed)
            selected_depths = ['10.0m', '100.0m', '300.0m'] if len(temp_cols) > 3 else temp_cols
            selected_cols = [f'TEMP_{depth}' for depth in selected_depths if f'TEMP_{depth}' in df_temp_clean.columns]
            
            if selected_cols:
                # Plot temperature profiles
                plot_temperature_profile(df_temp_clean, selected_cols, location_code, loc_output_dir)
            
            # Save to CSV
            df_temp_clean.to_csv(f"{cleaned_dir}/{location_code}_TEMP_clean.csv")
            print(f"Saved cleaned temperature profile data to {cleaned_dir}/{location_code}_TEMP_clean.csv")
    
    # Process wind data
    if 'wind' in data_dict:
        print("\nProcessing Wind data...")
        df_wind = data_dict['wind'].copy()
        
        # Apply quality filtering
        # Note: Wind data sometimes lacks Q and S columns
        if 'Q' in df_wind.columns or 'S' in df_wind.columns:
            df_wind_filtered = apply_quality_filtering(df_wind, "wind")
        else:
            print("  No quality or source columns found in wind data")
            df_wind_filtered = df_wind.copy()
        
        # Apply missing value handling
        wind_cols = [col for col in ['UWND', 'VWND', 'WSPD', 'WDIR'] if col in df_wind_filtered.columns]
        df_wind_clean = handle_missing_values(df_wind_filtered, "wind", wind_cols)
        
        # Process wind components
        for col in wind_cols:
            if col in df_wind_clean.columns:
                # Handle outliers
                df_wind_clean = handle_outliers(df_wind_clean, col)
        
        # Visualize wind speed
        if 'WSPD' in df_wind_clean.columns:
            plot_time_series(df_wind_clean, 'WSPD', 'Wind Speed', 'm/s', location_code, loc_output_dir)
            plot_seasonal_patterns(df_wind_clean, 'WSPD', 'Wind Speed', 'm/s', location_code, loc_output_dir)
            plot_annual_trends(df_wind_clean, 'WSPD', 'Wind Speed', 'm/s', location_code, loc_output_dir)
            
            # Store cleaned data
            cleaned_data['WSPD'] = df_wind_clean[['WSPD']].copy()
        
        # Wind direction visualization (if both components available)
        if 'UWND' in df_wind_clean.columns and 'VWND' in df_wind_clean.columns:
            plot_wind_rose(df_wind_clean, location_code, loc_output_dir)
        
        # Save to CSV
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
        monthly_data = df[variable].resample('ME').mean()
        
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
    if len(df) < 365:
        print(f"Skipping annual trend analysis for {variable}: insufficient data")
        return
    
    try:
        annual_data = df[variable].resample('YE').mean()
        if len(annual_data) < 3:
            print(f"Skipping annual trend analysis for {variable}: less than 3 years of data")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Clear any existing formatters
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        
        # Set date formatter explicitly
        date_formatter = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(date_formatter)
        
        # Plot annual data
        ax.plot(annual_data.index, annual_data.values, 'b-', label='Annual Average')
        
        valid_data = annual_data.dropna()
        if len(valid_data) >= 3:
            numeric_idx = np.arange(len(valid_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(numeric_idx, valid_data)
            
            # Calculate trend line
            trend_line = intercept + slope * numeric_idx
            
            # Plot trend line in one go instead of segments
            ax.plot(valid_data.index, trend_line, 
                   'r--', linewidth=2,
                   label=f'Trend: {slope:.4f} per year (p={p_value:.4f}, R²={r_value**2:.4f})')
        
        ax.set_title(f'Annual {var_name} Trend at {location_code}')
        ax.set_ylabel(f'{var_name} ({unit})')
        ax.set_xlabel('Year')
        ax.grid(True)
        ax.legend()
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'annual_trend_{variable}_{location_code}.png'))
        
    except ValueError as ve:
        print(f"Error processing {variable}: {ve}")
    except RuntimeError as re:
        print(f"Runtime error while plotting {variable}: {re}")
    except Exception as e:
        print(f"Unexpected error processing {variable}: {e}")
    finally:
        plt.close()

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
                # First ensure we only work with rows that have both components
                valid_mask = ~(df['UWND'].isna() | df['VWND'].isna())
                if valid_mask.sum() < 30:
                    print("Skipping wind rose plot: insufficient valid wind component data")
                    return
                
                uwnd = df.loc[valid_mask, 'UWND'].values
                vwnd = df.loc[valid_mask, 'VWND'].values
                
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
            # For existing wind speed and direction, only use rows that have both values
            valid_mask = ~(df['WSPD'].isna() | df['WDIR'].isna())
            if valid_mask.sum() < 30:
                print("Skipping wind rose plot: insufficient valid wind data")
                return
            
            wspd = df.loc[valid_mask, 'WSPD'].values
            wdir = df.loc[valid_mask, 'WDIR'].values
            
            # Create temporary DataFrame with values
            temp_df = pd.DataFrame({
                'wspd': wspd,
                'wdir': wdir
            })
        
        # Verify we have data to plot
        if len(temp_df) < 30:
            print("Skipping wind rose plot: insufficient valid data after filtering")
            return
            
        # Create figure with more space around it
        fig = plt.figure(figsize=(12, 10))
        
        # Create wind rose with adjusted position to make room for labels
        rect = [0.1, 0.1, 0.8, 0.8]  # [left, bottom, width, height]
        ax = WindroseAxes(fig, rect)
        fig.add_axes(ax)
        
        # Define custom wind speed bins for better visualization
        bins = np.array([0, 2.2, 4.2, 6.3, 8.3, 10.4, 15])
        
        # Generate a better color palette
        cmap = plt.cm.viridis_r  # Or try: plt.cm.turbo, plt.cm.jet
        
        # Plot the wind rose with improved parameters
        ax.bar(
            temp_df['wdir'], 
            temp_df['wspd'], 
            normed=True, 
            opening=0.8, 
            edgecolor='white',
            nsector=16,  # Use 16 sectors for smoother appearance
            bins=bins,
            cmap=cmap
        )
        
        # Improve legend appearance and position
        legend = ax.set_legend(
            title='Wind Speed (m/s)', 
            loc='lower left',
            bbox_to_anchor=(-0.1, -0.15),  # Position legend outside the plot
            ncol=3,  # Use multiple columns for more compact legend
            fontsize=9,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        legend.get_title().set_fontsize(10)  # Adjust title font size
        
        # Improve title positioning and appearance
        ax.set_title(f'Wind Rose at {location_code}', y=1.08, fontsize=14, fontweight='bold')
        
        # Make directional labels (N, S, E, W) bolder and larger
        ax.set_rgrids([3.0, 6.0, 9.0, 12.0, 15.0], angle=0, fontsize=9, fontweight='bold')
        ax.set_radii_angle(angle=45)  # Angle for radius labels
        
        # Adjust cardinal direction text size and weight
        for text in ax.get_xticklabels():
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        # Add tight layout but with padding to prevent cutting off labels
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Save figure with higher DPI for clearer labels
        plt.savefig(f'{output_dir}/wind_rose.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Wind rose plot created successfully for {location_code}")
        
    except Exception as e:
        print(f"Error creating wind rose plot: {e}")
        import traceback
        traceback.print_exc()  # This will print the detailed error stack

def combine_key_variables(cleaned_data, location_code, cleaned_dir):
    """
    Combine key meteorological variables into a single dataset with improved join strategy.
    This function ensures that all data points from individual cleaned files are properly
    preserved in the combined dataset.
    
    Parameters:
    -----------
    cleaned_data : dict
        Dictionary containing DataFrames for each variable type
    location_code : str
        Location identifier (e.g., '0N90E')
    cleaned_dir : str
        Directory for saving cleaned data
    """
    try:
        key_vars = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad', 'UWND', 'VWND']
        available_vars = [var for var in key_vars if var in cleaned_data]
        
        if len(available_vars) <= 1:
            print("Not enough variables available to create combined dataset")
            return
        
        print(f"\nCombining data for {len(available_vars)} variables: {', '.join(available_vars)}")
        
        # STEP 1: Build a universal index from all variables
        all_timestamps = set()
        for var in available_vars:
            all_timestamps.update(cleaned_data[var].index)
        
        # Create a sorted master index
        master_index = pd.DatetimeIndex(sorted(list(all_timestamps)))
        print(f"Created master index with {len(master_index)} unique timestamps")
        
        # STEP 2: Initialize the combined DataFrame with the master index
        combined_df = pd.DataFrame(index=master_index)
        combined_df.index.name = 'Date'
        
        # Add date component columns for convenience
        combined_df['year'] = combined_df.index.year
        combined_df['month'] = combined_df.index.month
        combined_df['day'] = combined_df.index.day
        
        # STEP 3: Add each variable with appropriate handling of quality and source info
        for var in available_vars:
            # Get the variable's DataFrame
            df_var = cleaned_data[var]
            
            # Only use the actual data column, not any metadata columns
            if var in df_var.columns:
                print(f"  Adding variable {var}...")
                
                # First add the main data column
                combined_df[var] = df_var[var]
                
                # Check for quality information
                if 'data_quality' in df_var.columns:
                    combined_df[f'{var}_quality'] = df_var['data_quality']
                
                # Check for source code information
                if 'source_priority' in df_var.columns:
                    combined_df[f'{var}_priority'] = df_var['source_priority']
                
                # Add original source code if available
                if 'S' in df_var.columns:
                    combined_df[f'{var}_source'] = df_var['S']
                
                # Add original quality code if available
                if 'Q' in df_var.columns:
                    combined_df[f'{var}_Q'] = df_var['Q']
                    
                print(f"    Added {df_var[var].count()} data points out of {len(df_var)} records")
            else:
                print(f"  Warning: Variable {var} not found in its DataFrame")
        
        # STEP 4: Save the combined data
        combined_file = f"{cleaned_dir}/{location_code}_combined_clean.csv" 
        combined_df.to_csv(combined_file)
        print(f"Saved combined dataset to {combined_file}")
        
        # STEP 5: Create a detailed coverage report
        coverage_report = pd.DataFrame(index=available_vars, 
                                      columns=['Total_Records', 'Available_Records', 'Coverage_Pct'])
        
        for var in available_vars:
            if var in combined_df.columns:
                total = len(combined_df)
                available = combined_df[var].count()
                coverage = (available / total) * 100
                
                coverage_report.loc[var, 'Total_Records'] = total
                coverage_report.loc[var, 'Available_Records'] = available
                coverage_report.loc[var, 'Coverage_Pct'] = coverage
                
                print(f"  {var}: {available}/{total} records ({coverage:.2f}% coverage)")
        
        # Save the coverage report
        coverage_file = f"{cleaned_dir}/{location_code}_coverage_report.csv"
        coverage_report.to_csv(coverage_file)
        print(f"Saved coverage report to {coverage_file}")
        
        # STEP 6: Create a correlation matrix if we have enough variables
        numeric_vars = [var for var in available_vars if var in combined_df.columns]
        
        if len(numeric_vars) >= 2:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Only include rows that have data for at least 2 variables
                valid_rows = combined_df[numeric_vars].dropna(thresh=2)
                
                if len(valid_rows) > 10:  # Ensure we have enough data for a meaningful correlation
                    plt.figure(figsize=(10, 8))
                    corr_matrix = valid_rows.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title(f'Correlation Matrix for {location_code}')
                    plt.tight_layout()
                    plt.savefig(f"{cleaned_dir}/{location_code}_correlation_matrix.png")
                    plt.close()
                    
                    print("Created correlation matrix visualization")
                else:
                    print("Not enough overlapping data for correlation analysis")
            except Exception as e:
                print(f"Error creating correlation visualization: {e}")
        
        return combined_df
        
    except Exception as e:
        print(f"Error combining variables: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":

    
    # Define data directories for each location
    data_dirs = {
        '0N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/0N90E/CSV',
        '4N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/4N90E/CSV',
        '8N90E': '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys/8N90E/CSV'
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