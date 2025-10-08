import pandas as pd
import numpy as np
from datetime import datetime

print("🔄 MERGING BMKG DATASETS")
print("=" * 60)

# Load the datasets
print("📂 Loading datasets...")

# Load the rainfall data into a DataFrame
rainfall_data = pd.read_csv("/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/preprocessing/variable/curah hujan/preprocessed_rainfall_data.csv")
print(f"✅ Rainfall data loaded: {len(rainfall_data)} records")

# Load the humidity data into a DataFrame  
humidity_data = pd.read_csv("/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/preprocessing/variable/kelembaban/preprocessed_humidity_data.csv")
print(f"✅ Humidity data loaded: {len(humidity_data)} records")

# Load the temperature data into a DataFrame
temperature_data = pd.read_csv("/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/preprocessing/variable/suhu/preprocessed_temp_data.csv")
print(f"✅ Temperature data loaded: {len(temperature_data)} records")

# Convert Date columns to datetime for proper merging
print("\n🔄 Converting Date columns...")
rainfall_data['Date'] = pd.to_datetime(rainfall_data['Date'])
humidity_data['Date'] = pd.to_datetime(humidity_data['Date']) 
temperature_data['Date'] = pd.to_datetime(temperature_data['Date'])

# Display basic info about each dataset
print("\n📊 Dataset Info:")
print(f"Rainfall columns: {list(rainfall_data.columns)}")
print(f"Humidity columns: {list(humidity_data.columns)}")
print(f"Temperature columns: {list(temperature_data.columns)}")

# Check date ranges for each dataset
print(f"\nRainfall date range: {rainfall_data['Date'].min()} to {rainfall_data['Date'].max()}")
print(f"Humidity date range: {humidity_data['Date'].min()} to {humidity_data['Date'].max()}")  
print(f"Temperature date range: {temperature_data['Date'].min()} to {temperature_data['Date'].max()}")

# Start merging process
print("\n🔗 Merging datasets...")

# Start with rainfall data as base (it has the most preprocessing metadata)
merged_data = rainfall_data.copy()
print(f"Base dataset (rainfall): {len(merged_data)} records")

# Merge with humidity data
merged_data = merged_data.merge(
    humidity_data[['Date', 'RH_AVG_preprocessed']], 
    on='Date', 
    how='outer',
    suffixes=('', '_humidity')
)
print(f"After humidity merge: {len(merged_data)} records")

# Merge with temperature data  
merged_data = merged_data.merge(
    temperature_data[['Date', 'Year', 'Month', 'Day', 'TN', 'TX', 'TAVG']], 
    on=['Date', 'Year', 'Month', 'Day'], 
    how='outer',
    suffixes=('', '_temp')
)
# Setelah temperature merge (sekitar line 51):
print(f"Columns after temperature merge: {list(merged_data.columns)}")
print(f"Sample row:\n{merged_data.iloc[0]}")

# Setelah drop columns:
print(f"Columns after drop: {list(merged_data.columns)}")

# Sort by Date to ensure chronological order
merged_data = merged_data.sort_values('Date').reset_index(drop=True)

merged_data['Year'] = merged_data['Date'].dt.year
merged_data['Month'] = merged_data['Date'].dt.month  
merged_data['Day'] = merged_data['Date'].dt.day

# Create data quality flag based on missing/imputed data
print("\n🏷️ Creating data quality flags...")

def create_quality_flag(row):
    """Create quality flag based on data completeness and imputation"""
    flags = []
    
    # Check rainfall quality
    if pd.isna(row['RR_original']):
        if row['RR_estimation_method'] == 'meteorological_estimate_with_ss':
            flags.append('RR_estimated')
        elif row['imputation_method'] in ['linear_interpolation', 'monthly_average_with_ss']:
            flags.append('RR_imputed')
    
    # Check outlier flag
    if row['is_outlier'] == True:
        flags.append('RR_outlier')
    
    # Check humidity (should be minimal issues)
    if pd.isna(row['RH_AVG_preprocessed']):
        flags.append('RH_missing')
    
    # Check temperature
    if pd.isna(row['TN']) or pd.isna(row['TX']) or pd.isna(row['TAVG']):
        flags.append('TEMP_missing')
    
    # Return combined flags or 'clean' if no issues
    return '|'.join(flags) if flags else 'clean'

merged_data['data_quality_flag'] = merged_data.apply(create_quality_flag, axis=1)

# Display quality summary
print("\n📋 Data Quality Summary:")
quality_counts = merged_data['data_quality_flag'].value_counts()
for flag, count in quality_counts.head(10).items():
    percentage = (count / len(merged_data)) * 100
    print(f"   {flag}: {count} records ({percentage:.1f}%)")

# Check for any remaining missing values
print("\n🔍 Missing Values Check:")
missing_summary = merged_data.isnull().sum()
for col, missing in missing_summary.items():
    if missing > 0:
        percentage = (missing / len(merged_data)) * 100
        print(f"   {col}: {missing} missing ({percentage:.2f}%)")

merged_data = merged_data.rename(columns={
    'RR_imputed': 'RR',
    'RH_AVG_preprocessed': 'RH_AVG'
})

# Reorder columns for better organization (with updated column names and removed columns)
column_order = [
    'Date', 'Year', 'Month', 'Day',
    'TN', 'TX', 'TAVG',  # Temperature variables (highest quality)
    'RH_AVG',  # Humidity (renamed from RH_AVG_preprocessed)  
    'RR',  # Rainfall (renamed from RR_imputed)
    'is_outlier', 'data_quality_flag'  # Quality flags only
]

# Ensure all columns exist before reordering
available_columns = [col for col in column_order if col in merged_data.columns]
other_columns = [col for col in merged_data.columns if col not in column_order]
final_column_order = available_columns + other_columns

# Apply the new column order
columns_to_drop = ['RR_original', 'RR_estimation_method', 'imputation_method']
merged_data = merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns])

# Display final dataset info
print(f"\n📊 Final Merged Dataset Info:")
print(f"   Total records: {len(merged_data)}")
print(f"   Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
print(f"   Total columns: {len(merged_data.columns)}")
print(f"   Columns: {list(merged_data.columns)}")

# Save the merged dataset
output_path = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/preprocessing/cleaned_bmkg.csv"
merged_data.to_csv(output_path, index=False)

print(f"\n💾 Merged dataset saved to: {output_path}")

# Final statistics
print(f"\n📈 Final Statistics:")
print(f"   Clean records: {(merged_data['data_quality_flag'] == 'clean').sum()} ({(merged_data['data_quality_flag'] == 'clean').sum()/len(merged_data)*100:.1f}%)")
print(f"   Records with some imputation: {(merged_data['data_quality_flag'] != 'clean').sum()} ({(merged_data['data_quality_flag'] != 'clean').sum()/len(merged_data)*100:.1f}%)")

print("\n🎉 MERGING COMPLETED SUCCESSFULLY!")
print("=" * 60)