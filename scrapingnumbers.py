import pandas as pd
import re

# Load the CSV file
csv_file = "pet_profile.csv"  # Replace with actual filename
df = pd.read_csv(csv_file)

# Words to ignore
STOP_WORDS = {"and"}

# Colour Encoding Function
def clean_colour(colour):
    if pd.isna(colour):  # Handle missing values
        return ""
    colour = str(colour).lower()
    colour = re.sub(r'[\s,/&-]+', ' ', colour)  # Replace separators with space
    colour = colour.replace("tri colour", "tricolour")  # Standardize spelling
    words = [word for word in colour.split() if word not in STOP_WORDS]  # Remove stop words
    return " ".join(words)

# Apply cleaning function to the 'Colour' column
df["Cleaned_Colour"] = df["Colour"].apply(clean_colour)

# Extract unique color words
all_words = sorted(set(" ".join(df["Cleaned_Colour"]).split()))  # Unique sorted words

# Assign unique IDs to each color word
colour_mapping = {word: i+1 for i, word in enumerate(all_words)}  # Start IDs from 1

# Encode colours as lists of numbers
df["Colour_encoded"] = df["Cleaned_Colour"].apply(lambda x: [colour_mapping[word] for word in x.split() if word in colour_mapping])

# Age Conversion
def convert_age(age_str):
    if pd.isna(age_str):
        return None
    age_str = age_str.lower()
    numbers = list(map(int, re.findall(r'\d+', age_str)))
    if "month" in age_str:
        return round(sum(numbers) / (2 * 12), 2) if len(numbers) == 2 else round(numbers[0] / 12, 2)
    elif "year" in age_str and numbers:
        return int(numbers[0])
    return None

df["Age_numeric"] = df["Age"].apply(convert_age)

# Lead Training
lead_training_columns = ["Lead_Training", "Needs_Training", "Knows_Basic_Commands", "High_Energy", "Requires_Training_Classes"]
def extract_lead_training(text):
    if pd.isna(text): return pd.Series([0] * 5)
    text = text.lower()
    return pd.Series([
        "walk nicely on a lead" in text, "need training" in text, "know how to sit" in text,
        "love to learn agility" in text, "training classes" in text
    ], dtype=int)
df[lead_training_columns] = df["Walk nicely on a lead"].apply(extract_lead_training)

# Household Preference
household_columns = ["Adult_Only", "Secondary_Age_Children", "Primary_Age_Children"]
def extract_household_preference(text):
    if pd.isna(text): return pd.Series([0] * 3)
    text = text.lower()
    return pd.Series(["adult only" in text, "secondary school age" in text, "primary school age" in text], dtype=int)
df[household_columns] = df["Adult only household"].apply(extract_household_preference)

# Companionship Preference
companionship_columns = ["Needs_Constant_Company", "Needs_Training_To_Be_Alone", "Can_Be_Left_Alone"]
def extract_companionship_preference(text):
    if pd.isna(text): return pd.Series([0] * 3)
    text = text.lower()
    return pd.Series([
        "someone with me most of the time" in text,
        "teaching that it's ok to be alone" in text,
        "can be left alone for short periods" in text
    ], dtype=int)
df[companionship_columns] = df["Companionship preference"].apply(extract_companionship_preference)

# Dog Compatibility
dog_compatibility_columns = ["Prefers_Solo", "May_Live_With_Dog", "Good_With_Dogs"]
def extract_dog_compatibility(text):
    if pd.isna(text): return pd.Series([0] * 3)
    text = text.lower()
    return pd.Series([
        "prefer to be the only dog" in text,
        "may be able to live with another dog" in text,
        "get on well with many other dogs" in text
    ], dtype=int)
df[dog_compatibility_columns] = df["Dog compatibility"].apply(extract_dog_compatibility)

# Cat Compatibility
cat_compatibility_columns = ["Avoids_Cats", "Prefers_Solo_Animal", "May_Live_With_Cat"]
def extract_cat_compatibility(text):
    if pd.isna(text): return pd.Series([0] * 3)
    text = text.lower()
    return pd.Series([
        "prefer not to live with a cat" in text,
        "prefer to be the only animal" in text,
        "may be able to live with a friendly cat" in text
    ], dtype=int)
df[cat_compatibility_columns] = df["Cat compatibility"].apply(extract_cat_compatibility)

# Personality Traits
personality_columns = ["Bright_Spark", "Loves_Snoozing", "Energetic", "Young_at_Heart", "Quiet_Loving", "Small_Fun", "Gentle_Giant"]
def extract_personality_traits(text):
    if pd.isna(text): return pd.Series([0] * 7)
    text = text.lower()
    return pd.Series([
        "bright spark" in text, "love to snooze" in text, "always on the go" in text,
        "young at heart" in text, "quiet but full of love" in text,
        "small but full of fun" in text, "gentle giant" in text
    ], dtype=int)
df[personality_columns] = df["Personality trait"].apply(extract_personality_traits)

# Playfulness
playfulness_columns = ["Playful", "Affectionate", "Food_Motivated", "Needs_Play_Training"]
def extract_playfulness_traits(text):
    if pd.isna(text): return pd.Series([0] * 4)
    text = text.lower()
    return pd.Series([
        "toys and games" in text, "fuss and attention" in text,
        "food and treats" in text, "learn how to play" in text
    ], dtype=int)
df[playfulness_columns] = df["Playfulness"].apply(extract_playfulness_traits)

# ✅ Preserve RSPCA Dog URLs for Matching
if "URL" in df.columns:
    df["URL"] = df["URL"]  # Ensure URLs are kept in the final dataset

# Save to CSV
output_file = "pet_profile_encoded.csv"
df.to_csv(output_file, index=False)

print("Data successfully encoded and saved to", output_file)
