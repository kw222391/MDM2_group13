import pandas as pd

# Load the CSV files
dogdata = pd.read_csv('dogdata.csv')
pet_profile = pd.read_csv('pet_profile.csv')

# Standardize breed names in both datasets
dogdata.rename(columns={'Unnamed: 0': 'Breed'}, inplace=True)
dogdata['Breed'] = dogdata['Breed'].str.strip().str.lower()
pet_profile['Breed'] = pet_profile['Breed'].str.strip().str.lower()

# Merge datasets on Breed to combine pet profile data with breed characteristics
merged_data = pet_profile.merge(dogdata, on='Breed', how='left')

# Fill missing values with reasonable defaults
merged_data['energy_level_value'] = merged_data['energy_level_value'].fillna(0.5)
merged_data['max_height'] = merged_data['max_height'].fillna(50.0)

# Save the merged dataset to a CSV file
merged_data.to_csv('merged.csv', index=False)
print("Merged data saved as 'merged.csv'")

# Function to match dogs based on user preferences
def match_dogs():
    preferred_temperament = input("Enter preferred temperament (e.g., friendly, loyal, active): ").strip()
    min_energy = input("Enter minimum energy level (0 to 1, or leave blank): ").strip()
    max_size = input("Enter maximum height in cm (or leave blank): ").strip()
    
    # Convert input values
    min_energy = float(min_energy) if min_energy else None
    max_size = float(max_size) if max_size else None
    
    filtered_data = merged_data.copy()
    
    # Apply filters
    if preferred_temperament:
        filtered_data = filtered_data[
            filtered_data['temperament'].str.contains(preferred_temperament, case=False, na=False)
        ]
    if min_energy is not None:
        filtered_data = filtered_data[filtered_data['energy_level_value'] >= min_energy]
    if max_size is not None:
        filtered_data = filtered_data[filtered_data['max_height'] <= max_size]
    
    # Display results
    columns_to_show = ['Animal Name', 'Breed', 'Age', 'temperament', 'energy_level_category', 'max_height']
    existing_columns = [col for col in columns_to_show if col in filtered_data.columns]
    
    results = filtered_data[existing_columns].head(5)
    print("\nMatching Dogs:")
    print(results if not results.empty else "No matches found.")

# Run the function
match_dogs()