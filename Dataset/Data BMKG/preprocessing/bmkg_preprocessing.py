import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from scipy.interpolate import CubicSpline
from datetime import datetime
import math
import warnings

# Ignore future warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def preprocess_bmkg_data(input_file, output_file):
    print("Loading data...")
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    # Convert Date to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        # Create Date from Year, Month, Day if Date column doesn't exist
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df.set_index('Date', inplace=True)
    
    # Sort data by date
    df = df.sort_index()
    
    # Define missing value codes
    missing_codes = [8888.0, 9999.0]
    
    # Create season column (Indonesia: wet/dry season)
    # Wet season: October-April, Dry season: May-September
    df['Season'] = df.index.month.map(lambda m: 'Wet' if m >= 10 or m <= 4 else 'Dry')
    
    # Display initial information
    print("\nInitial data info:")
    print(f"Dataset shape: {df.shape}")
    
    # Replace missing value codes with NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        for code in missing_codes:
            mask = df[col] == code
            if mask.sum() > 0:
                print(f"Replaced {mask.sum()} instances of {code} with NaN in column {col}")
                df.loc[mask, col] = np.nan
    
    # Display missing values before preprocessing
    missing_before = df.isnull().sum() / len(df) * 100
    print("\nMissing values percentage before preprocessing:")
    print(missing_before)
    
    # 1. Preprocess Rainfall (RR) - 36.75% Missing
    print("\nProcessing Rainfall (RR)...")
    
    # Create flag for missing rainfall
    df['is_RR_missing'] = df['RR'].isna().astype(int)
    
    # Specific handling for dry season - set missing values to 0
    dry_season_mask = (df['Season'] == 'Dry') & df['RR'].isna()
    df.loc[dry_season_mask, 'RR'] = 0
    
    # For remaining missing values, try simple but effective methods
    # First, check if there are still missing values
    if df['RR'].isna().any():
        # Try moving average first (30-day window)
        print("Using moving average for remaining RR missing values...")
        # Temporarily fill with 0 for calculation purposes
        temp_df = df.copy()
        temp_df['RR'] = temp_df['RR'].fillna(0)
        
        # Calculate 30-day moving average
        window_size = 30
        moving_avg = temp_df['RR'].rolling(window=window_size, center=True, min_periods=1).mean()
        
        # Fill missing values with moving average (which takes seasonality into account)
        missing_mask = df['RR'].isna()
        df.loc[missing_mask, 'RR'] = moving_avg[missing_mask]
        
        # For any remaining NaNs, use forward fill followed by backward fill
        if df['RR'].isna().any():
            print(f"Using forward/backward fill for {df['RR'].isna().sum()} remaining RR values...")
            df['RR'] = df['RR'].ffill().bfill()
    
    # Ensure no negative values for rainfall
    df['RR'] = df['RR'].clip(lower=0)
    
    # 2. Wind Direction at Maximum Speed (DDD_X) - 4.95% Missing
    print("Processing Wind Direction (DDD_X)...")
    
    # Function to calculate circular mean for wind direction
    def circular_mean(directions):
        directions_rad = np.radians(directions)
        x_mean = np.nanmean(np.cos(directions_rad))
        y_mean = np.nanmean(np.sin(directions_rad))
        mean_direction = np.degrees(np.arctan2(y_mean, x_mean))
        return (mean_direction + 360) % 360 if not np.isnan(mean_direction) else np.nan
    
    # Group by month/season for mode calculation
    monthly_wind_direction = df.groupby([df.index.month])['DDD_X'].apply(
        lambda x: circular_mean(x.dropna()) if x.count() > 0 else np.nan
    ).to_dict()
    
    seasonal_wind_direction = df.groupby('Season')['DDD_X'].apply(
        lambda x: circular_mean(x.dropna()) if x.count() > 0 else np.nan
    ).to_dict()
    
    # Fill missing values based on FF_X and closest valid DDD_X
    for idx in df.index[df['DDD_X'].isna()]:
        if not pd.isna(df.loc[idx, 'FF_X']):
            # Look for closest date with valid DDD_X
            closest_dates = df[~df['DDD_X'].isna()].index
            if len(closest_dates) > 0:
                closest_date = min(closest_dates, key=lambda date: abs((date - idx).total_seconds()))
                if abs((closest_date - idx).total_seconds()) < 7 * 24 * 3600:  # Within 7 days
                    df.loc[idx, 'DDD_X'] = df.loc[closest_date, 'DDD_X']
                    continue
        
        # If still NaN, use monthly or seasonal mode
        month = idx.month
        season = df.loc[idx, 'Season']
        
        if month in monthly_wind_direction and not np.isnan(monthly_wind_direction[month]):
            df.loc[idx, 'DDD_X'] = monthly_wind_direction[month]
        elif season in seasonal_wind_direction and not np.isnan(seasonal_wind_direction[season]):
            df.loc[idx, 'DDD_X'] = seasonal_wind_direction[season]
    
    # 3. Average Wind Speed (FF_AVG) - 7.36% Missing
    print("Processing Average Wind Speed (FF_AVG)...")
    
    # Check correlation between FF_X and FF_AVG to see if regression is appropriate
    valid_mask = ~df['FF_X'].isna() & ~df['FF_AVG'].isna()
    correlation = df.loc[valid_mask, ['FF_X', 'FF_AVG']].corr().iloc[0, 1]
    print(f"Correlation between FF_X and FF_AVG: {correlation:.4f}")
    
    # For missing FF_AVG where FF_X is available, use linear regression
    if correlation > 0.5:  # Sufficient correlation for regression
        from sklearn.linear_model import LinearRegression
        
        X = df.loc[valid_mask, 'FF_X'].values.reshape(-1, 1)
        y = df.loc[valid_mask, 'FF_AVG'].values
        
        model = LinearRegression().fit(X, y)
        print(f"Linear model: FF_AVG = {model.coef_[0]:.4f} * FF_X + {model.intercept_:.4f}")
        
        # Apply regression model where FF_X is available but FF_AVG is missing
        ff_x_valid_mask = df['FF_X'].notna() & df['FF_AVG'].isna()
        df.loc[ff_x_valid_mask, 'FF_AVG'] = model.predict(df.loc[ff_x_valid_mask, 'FF_X'].values.reshape(-1, 1))
    
    # For remaining missing values, use weighted window linear interpolation
    df['FF_AVG'] = df['FF_AVG'].interpolate(method='linear', limit_direction='both')
    
    # 4. Temperature (TX, TN, TAVG) & Humidity (RH_AVG) - < 5% Missing
    print("Processing Temperature and Humidity...")
    
    for col in ['TX', 'TN', 'TAVG', 'RH_AVG']:
        try:
            # Use time-based interpolation first
            df[col] = df[col].interpolate(method='time', limit_direction='both')
            
            # Additional smoothing with rolling window if needed
            if df[col].isna().any():
                window_size = 7  # 7-day window
                df[col] = df[col].fillna(df[col].rolling(window=window_size, center=True, min_periods=1).mean())
            
            # Any remaining NaNs, use forward fill and then backward fill
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
        except Exception as e:
            print(f"Interpolation failed for {col}: {e}")
            # Simple fallback
            df[col] = df[col].ffill().bfill()
    
    # 5. Sunshine Duration (SS) - 0.74% Missing
    print("Processing Sunshine Duration (SS)...")
    
    # Interpolate with time method first
    df['SS'] = df['SS'].interpolate(method='time', limit_direction='both')
    
    # Apply physical constraints
    # Sunshine duration cannot be negative
    df['SS'] = df['SS'].clip(lower=0)
    
    # Calculate approximate max daylight hours (very rough estimate based on latitude)
    # This would be more accurate with actual location data
    # For Indonesia (equatorial), daylight hours range from ~11.5 to ~12.5 hours
    df['max_daylight'] = 12.5  # Simplified constant for Indonesia
    
    # Ensure sunshine doesn't exceed max daylight
    df['SS'] = df[['SS', 'max_daylight']].min(axis=1)
    df.drop('max_daylight', axis=1, inplace=True)
    
    # 6. Most Common Wind Direction (DDD_CAR) - 1.96% Missing
    print("Processing Cardinal Wind Direction (DDD_CAR)...")
    
    # Convert to string type to handle categorical data properly
    df['DDD_CAR'] = df['DDD_CAR'].astype(str)
    
    # Replace 'nan' string with np.nan
    df.loc[df['DDD_CAR'] == 'nan', 'DDD_CAR'] = np.nan
    
    # Fill missing DDD_CAR with monthly mode
    for month in range(1, 13):
        # Get data for this month
        month_data = df[df.index.month == month]
        if month_data.empty or month_data['DDD_CAR'].isna().all():
            continue
            
        # Get most common value excluding NaN
        month_mode = month_data['DDD_CAR'].value_counts().idxmax() if not month_data['DDD_CAR'].dropna().empty else None
        
        if month_mode:
            # Fill missing values for this month
            month_mask = (df.index.month == month) & df['DDD_CAR'].isna()
            df.loc[month_mask, 'DDD_CAR'] = month_mode
    
    # For any remaining NaNs, try season-based filling
    for season in df['Season'].unique():
        if pd.isna(season):
            continue
            
        # Get data for this season
        season_data = df[df['Season'] == season]
        if season_data.empty or season_data['DDD_CAR'].isna().all():
            continue
            
        # Get most common value excluding NaN
        season_mode = season_data['DDD_CAR'].value_counts().idxmax() if not season_data['DDD_CAR'].dropna().empty else None
        
        if season_mode:
            # Fill missing values for this season
            season_mask = (df['Season'] == season) & df['DDD_CAR'].isna()
            df.loc[season_mask, 'DDD_CAR'] = season_mode
    
    # Any remaining NaNs, fill with the overall most common direction
    if df['DDD_CAR'].isna().any():
        overall_mode = df['DDD_CAR'].value_counts().idxmax() if not df['DDD_CAR'].dropna().empty else 'N'
        df['DDD_CAR'] = df['DDD_CAR'].fillna(overall_mode)
    
    # Final check for any remaining missing values
    missing_after = df.isnull().sum() / len(df) * 100
    print("\nMissing values percentage after preprocessing:")
    print(missing_after)
    
    # Drop the temporary column
    if 'max_daylight' in df.columns:
        df.drop('max_daylight', axis=1, inplace=True)
    
    # Save preprocessed data
    df.to_csv(output_file)
    print(f"\nPreprocessed data saved to {output_file}")
    
    return df

# Main execution
if __name__ == "__main__":
    input_file = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
    output_file = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV CLEANED/BMKG_Data_Cleaned.csv"
    
    try:
        processed_df = preprocess_bmkg_data(input_file, output_file)
        
        if processed_df is not None:
            # Generate some diagnostic plots
            plt.figure(figsize=(15, 10))
            
            # Plot rainfall data
            plt.subplot(2, 2, 1)
            processed_df['RR'].plot()
            plt.title('Rainfall After Preprocessing')
            plt.ylabel('Rainfall (mm)')
            
            # Plot temperature data
            plt.subplot(2, 2, 2)
            processed_df[['TN', 'TX', 'TAVG']].plot()
            plt.title('Temperature After Preprocessing')
            plt.ylabel('Temperature (°C)')
            
            # Plot humidity
            plt.subplot(2, 2, 3)
            processed_df['RH_AVG'].plot()
            plt.title('Relative Humidity After Preprocessing')
            plt.ylabel('Humidity (%)')
            
            # Plot sunshine hours
            plt.subplot(2, 2, 4)
            processed_df['SS'].plot()
            plt.title('Sunshine Hours After Preprocessing')
            plt.ylabel('Sunshine Duration (hours)')
            
            plt.tight_layout()
            plt.savefig('BMKG_Data_Diagnostic_Plots.png')
            print("Diagnostic plots saved to 'BMKG_Data_Diagnostic_Plots.png'")
        else:
            print("Preprocessing failed, cannot generate diagnostic plots.")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
