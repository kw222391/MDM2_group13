import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
csv_file = "pet_profile_encoded.csv"
df = pd.read_csv(csv_file)

# ---- DATA CLEANING ----
def preprocess_data(df):
    """Convert categorical and text-based data into numerical format for comparison."""
    
    # Convert Age to numeric
    df["Age_numeric"] = df["Age"].str.extract(r'(\d+)').astype(float)

    # Convert Colour column to numerical encoding
    color_map = {"black": 1, "white": 2, "tan": 3, "brindle": 4, "brown": 5, "tri colour": 6}
    df["Colour_encoded"] = df["Colour"].apply(lambda x: [color_map[c.strip().lower()] for c in str(x).split(",") if c.strip().lower() in color_map])

    # Convert household type to binary
    df["Adult_Only"] = df["Adult only household"].apply(lambda x: 1 if "adult only" in str(x).lower() else 0)
    df["Secondary_Age_Children"] = df["Adult only household"].apply(lambda x: 0 if "adult only" in str(x).lower() else 1)
    df["Primary_Age_Children"] = df["Adult only household"].apply(lambda x: 0 if "adult only" in str(x).lower() else 1)

    return df

df = preprocess_data(df)

# ---- USER PREFERENCES ----
def get_user_preferences():
    """Collects user preferences and converts them into a numerical feature vector."""
    
    color_map = {"black": 1, "white": 2, "tan": 3, "brindle": 4, "brown": 5, "tri colour": 6}
    
    # Get color preferences
    color_options = list(color_map.keys())
    selected_colors = input(f"Choose preferred colors (comma-separated from {', '.join(color_options)}): ").lower().split(",")
    selected_color_codes = [color_map[c.strip()] for c in selected_colors if c.strip() in color_map]

    # Age preference
    age_preference = input("Enter preferred age (numeric): ")
    age_preference = float(age_preference) if age_preference.isnumeric() else None

    # Household preference
    household_options = ["Adult_Only", "Secondary_Age_Children", "Primary_Age_Children"]
    household_choice = input(f"Choose a household preference ({', '.join(household_options)}): ").strip()

    # Training preference
    training_options = ["Needs_Training", "Lead_Training", "Knows_Basic_Commands", "Requires_Training_Classes"]
    training_choice = input(f"Choose a training preference ({', '.join(training_options)}): ").strip()

    # Dog compatibility
    dog_options = ["Prefers_Solo", "May_Live_With_Dog", "Good_With_Dogs"]
    dog_choice = input(f"Choose dog compatibility ({', '.join(dog_options)}): ").strip()

    # Cat compatibility
    cat_options = ["Avoids_Cats", "Prefers_Solo_Animal", "May_Live_With_Cat"]
    cat_choice = input(f"Choose cat compatibility ({', '.join(cat_options)}): ").strip()

    # Personality traits
    personality_options = ["Bright_Spark", "Loves_Snoozing", "Energetic", "Young_at_Heart", "Quiet_Loving", "Small_Fun", "Gentle_Giant"]
    personality_choice = input(f"Choose a personality trait ({', '.join(personality_options)}): ").strip()

    # Playfulness preference
    playfulness_options = ["Playful", "Affectionate", "Food_Motivated", "Needs_Play_Training"]
    playfulness_choice = input(f"Choose a playfulness trait ({', '.join(playfulness_options)}): ").strip()

    # Convert to feature vector
    user_vector = {
        "Age_numeric": age_preference,
        "Colour_encoded": selected_color_codes,
        household_choice: 1,
        training_choice: 1,
        dog_choice: 1,
        cat_choice: 1,
        personality_choice: 1,
        playfulness_choice: 1
    }

    return user_vector

user_preferences = get_user_preferences()

# ---- FINDING THE CLOSEST MATCHES ----
def find_best_matches(df, user_prefs, k=5):
    """Finds the top K closest dogs based on user preferences using KNN."""
    
    # Extract numerical feature columns for comparison
    feature_columns = ["Age_numeric", "Adult_Only", "Secondary_Age_Children", "Primary_Age_Children",
                       "Needs_Training", "Lead_Training", "Knows_Basic_Commands", "Requires_Training_Classes",
                       "Prefers_Solo", "May_Live_With_Dog", "Good_With_Dogs",
                       "Avoids_Cats", "Prefers_Solo_Animal", "May_Live_With_Cat",
                       "Bright_Spark", "Loves_Snoozing", "Energetic", "Young_at_Heart",
                       "Quiet_Loving", "Small_Fun", "Gentle_Giant",
                       "Playful", "Affectionate", "Food_Motivated", "Needs_Play_Training"]

    # Fill NaN values with 0
    df_features = df[feature_columns].fillna(0)
    
    # Convert user preferences to a vector matching the feature set
    user_vector = np.zeros(len(feature_columns))
    for i, feature in enumerate(feature_columns):
        if feature in user_prefs:
            user_vector[i] = user_prefs[feature]

    # Standardize data to normalize distances
    scaler = StandardScaler()
    df_features_scaled = scaler.fit_transform(df_features)
    user_vector_scaled = scaler.transform([user_vector])

    # Fit KNN model
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(df_features_scaled)
    
    # Find closest matches
    distances, indices = knn.kneighbors(user_vector_scaled)

    # Get matching dogs
    best_matches = df.iloc[indices[0]]

    return best_matches

# Get top matches
top_matches = find_best_matches(df, user_preferences)

# ---- DISPLAY RESULTS ----
if not top_matches.empty:
    print("\nHere are the top recommended dogs based on your preferences:")
    print(top_matches[["Breed", "Age_numeric", "Colour_encoded"]])
else:
    print("No close matches found.")

