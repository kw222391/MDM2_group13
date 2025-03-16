import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

csv_file = "pet_profile_encoded.csv"
df = pd.read_csv(csv_file)

def preprocess_data(df):
    df["Age_numeric"] = df["Age"].str.extract(r'(\d+)').astype(float)
    color_map = {"black": 1, "white": 2, "tan": 3, "brindle": 4, "brown": 5, "tri colour": 6}
    df["Colour_encoded"] = df["Colour"].apply(lambda x: [color_map[c.strip().lower()] for c in str(x).split(",") if c.strip().lower() in color_map])
    df["Adult_Only"] = df["Adult only household"].apply(lambda x: 1 if "adult only" in str(x).lower() else 0)
    df["Secondary_Age_Children"] = df["Adult only household"].apply(lambda x: 0 if "adult only" in str(x).lower() else 1)
    df["Primary_Age_Children"] = df["Adult only household"].apply(lambda x: 0 if "adult only" in str(x).lower() else 1)
    return df

df = preprocess_data(df)

def get_user_preferences():
    color_map = {"black": 1, "white": 2, "tan": 3, "brindle": 4, "brown": 5, "tri colour": 6}
    color_options = list(color_map.keys())
    selected_colors = input(f"Choose preferred colors (comma-separated from {', '.join(color_options)}): ").lower().split(",")
    selected_color_codes = [color_map[c.strip()] for c in selected_colors if c.strip() in color_map]
    age_preference = input("Enter preferred age (numeric): ")
    age_preference = float(age_preference) if age_preference.isnumeric() else None
    household_options = ["Adult_Only", "Secondary_Age_Children", "Primary_Age_Children"]
    household_choice = input(f"Choose a household preference ({', '.join(household_options)}): ").strip()
    training_options = ["Needs_Training", "Lead_Training", "Knows_Basic_Commands", "Requires_Training_Classes"]
    training_choice = input(f"Choose a training preference ({', '.join(training_options)}): ").strip()
    dog_options = ["Prefers_Solo", "May_Live_With_Dog", "Good_With_Dogs"]
    dog_choice = input(f"Choose dog compatibility ({', '.join(dog_options)}): ").strip()
    cat_options = ["Avoids_Cats", "Prefers_Solo_Animal", "May_Live_With_Cat"]
    cat_choice = input(f"Choose cat compatibility ({', '.join(cat_options)}): ").strip()
    personality_options = ["Bright_Spark", "Loves_Snoozing", "Energetic", "Young_at_Heart", "Quiet_Loving", "Small_Fun", "Gentle_Giant"]
    personality_choice = input(f"Choose a personality trait ({', '.join(personality_options)}): ").strip()
    playfulness_options = ["Playful", "Affectionate", "Food_Motivated", "Needs_Play_Training"]
    playfulness_choice = input(f"Choose a playfulness trait ({', '.join(playfulness_options)}): ").strip()
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

def find_best_matches(df, user_prefs, k=5):
    feature_columns = ["Age_numeric", "Adult_Only", "Secondary_Age_Children", "Primary_Age_Children",
                       "Needs_Training", "Lead_Training", "Knows_Basic_Commands", "Requires_Training_Classes",
                       "Prefers_Solo", "May_Live_With_Dog", "Good_With_Dogs",
                       "Avoids_Cats", "Prefers_Solo_Animal", "May_Live_With_Cat",
                       "Bright_Spark", "Loves_Snoozing", "Energetic", "Young_at_Heart",
                       "Quiet_Loving", "Small_Fun", "Gentle_Giant",
                       "Playful", "Affectionate", "Food_Motivated", "Needs_Play_Training"]
    df_features = df[feature_columns].fillna(0)
    user_vector = np.zeros(len(feature_columns))
    for i, feature in enumerate(feature_columns):
        if feature in user_prefs:
            user_vector[i] = user_prefs[feature]
    scaler = StandardScaler()
    df_features_scaled = scaler.fit_transform(df_features)
    user_vector_scaled = scaler.transform([user_vector])
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(df_features_scaled)
    distances, indices = knn.kneighbors(user_vector_scaled)
    best_matches = df.iloc[indices[0]]
    return best_matches

top_matches = find_best_matches(df, user_preferences)

if not top_matches.empty:
    print("\nHere are the top recommended dogs based on your preferences:")
    for _, row in top_matches.iterrows():
        print(f" {row['Animal Name']} - {row['Breed']} ({row['Age_numeric']} years old)")
        print(f" Profile: {row['URL']}\n")
else:
    print("No close matches found.")
