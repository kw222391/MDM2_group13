import requests
from bs4 import BeautifulSoup
import csv
import re


def scrape_pet(url):
    """
    Scrapes the pet page at the given URL and returns a dictionary of pet data.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve {url}: Status code {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    # --- Basic Details ---
    # Attempt to get the pet name from a hidden input first, fallback to <h1>
    name_input = soup.find("input", id="animalName")
    if name_input and name_input.get("value"):
        animal_name = name_input["value"].strip().capitalize()
    else:
        h1 = soup.find("h1")
        animal_name = h1.get_text(strip=True) if h1 else "Unknown"

    # Extract details from the "About Me" table (using the first found instance)
    pet_details = {"Breed": "N/A", "Colour": "N/A", "Age": "N/A", "Ref": "N/A", "Distance": ""}
    about_me = soup.find("div", class_=lambda c: c and "aboutMe" in c)
    if about_me:
        for tr in about_me.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if th and td:
                key = th.get_text(strip=True).rstrip(':')
                value = td.get_text(strip=True)
                if key in pet_details:
                    pet_details[key] = value
            elif not tr.find("th") and tr.find("td"):
                # This row may contain extra info (e.g., distance)
                pet_details["Distance"] = tr.find("td").get_text(strip=True)

    # --- Lifestyle Traits ---
    # Map image filenames to standardized lifestyle attributes.
    lifestyle_map = {
        "training.png": "Walk nicely on a lead",
        "family.png": "Adult only household",
        "home.png": "Companionship preference",
        "dog.png": "Dog compatibility",
        "cat.png": "Cat compatibility",
        "personality.png": "Personality trait",
        "play.png": "Playfulness"
    }
    # Initialize all lifestyle attributes to a default value.
    lifestyle_data = {col: "Not specified" for col in lifestyle_map.values()}

    # Find the "My personality" section (div with id "lifeStyle")
    lifestyle_section = soup.find("div", id="lifeStyle")
    if lifestyle_section:
        # Each <li> should contain an <img> (for the trait) and a <span> with the descriptive text.
        for li in lifestyle_section.find_all("li"):
            img = li.find("img")
            span = li.find("span")
            if img and span:
                src = img.get("src", "")
                # Get the filename (e.g., "family.png") and match it to our mapping.
                filename = src.split("/")[-1].lower()
                if filename in lifestyle_map:
                    lifestyle_data[lifestyle_map[filename]] = span.get_text(strip=True)

    # --- Combine All Data ---
    pet_info = {
        "Animal Name": animal_name,
        "Breed": pet_details.get("Breed", "N/A"),
        "Colour": pet_details.get("Colour", "N/A"),
        "Age": pet_details.get("Age", "N/A"),
        "Ref": pet_details.get("Ref", "N/A"),
        "Distance": pet_details.get("Distance", ""),
        "URL": url  # ✅ Added URL to the data
    }
    pet_info.update(lifestyle_data)
    return pet_info


# List of pet URLs for NINA, BILLY, and COCO
pet_urls = [
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/NINA/ref/BSA2137486/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/BILLY/ref/BSA2137315/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/COCO/ref/BSA2137118/rehome"
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/NINA/ref/BSA2137486/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/MEELA/ref/BSA2136787/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/HARRY/ref/BSA2137626/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/BRIAN/ref/BSA2137450/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/PUMPKIN/ref/BSA2137449/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/BRUCE/ref/BSA2137268/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/PUDDLE/ref/BSA2136986/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/ARTHUR/ref/BSA2136910/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/ANNABELLE/ref/BSA2136828/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/NANCY/ref/BSA2136827/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/BARNABY/ref/BSA2136729/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/CINNAMON/ref/BSA2136550/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/SHELBY/ref/BSA2136420/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/ROBIN/ref/BSA2136087/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/SPIRIT/ref/BSA2136006/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/SPIRIT/ref/BSA2136006/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/TOBY/ref/BSA2135736/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/TOBY/ref/BSA2135736/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/MISSY/ref/BSA2135654/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/ROMAN/ref/BSA2132601/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/PEPE/ref/BSA2132518/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/RODGER/ref/BSA2129563/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/TIMMY/ref/BSA2122576/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/NELSON/ref/BSA2117222/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/HAZEL/ref/262261/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/SID/ref/BSA2137554/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/HAZEL/ref/262261/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/LOLLY/ref/260572/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/ARNIE/ref/BSA2137556/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/KILO/ref/BSA2137393/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/SHELDON/ref/BSA2137050/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/DUCHESS/ref/BSA2137054/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/MATILDA/ref/BSA2136826/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/COMET/ref/BSA2136187/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/REMIE/ref/BSA2137481/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/SYLWIA/ref/BSA2136854/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/BLOSSOM/ref/BSA2136856/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/BRYAN/ref/BSA2136752/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/ZEUS/ref/BSA2135894/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/MIKA/ref/BSA2135861/rehome",
    "https://www.rspca.org.uk/findapet/search/details/-/Animal/JESSIE/ref/BSA2135823/rehome"
]

# Scrape each pet's data
pet_data_list = []
for url in pet_urls:
    data = scrape_pet(url)
    if data:
        pet_data_list.append(data)

# --- Write Data to CSV ---
csv_columns = [
    "Animal Name", "Breed", "Colour", "Age", "Ref", "Distance",
    "Walk nicely on a lead",
    "Adult only household",
    "Companionship preference",
    "Dog compatibility",
    "Cat compatibility",
    "Personality trait",
    "Playfulness",
    "URL"  # ✅ Added URL column to CSV
]

csv_file = "pet_profile.csv"
try:
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for pet in pet_data_list:
            writer.writerow(pet)
    print("Data successfully saved to", csv_file)
except Exception as e:
    print("Error writing CSV file:", e)
