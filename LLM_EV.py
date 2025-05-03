import ollama
import requests
import subprocess

GOOGLE_API_KEY = "AIzaSyDXyV7BmpE58FjroyzSJ8eUWs-JGWhs6Os"

def get_google_rating(station_name, latitude, longitude):
    """Fetch Google reviews for a charging station using Place Details API."""
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{latitude},{longitude}",
        "radius": 10000,
        "keyword": station_name
    }
    
    response = requests.get(base_url, params=params).json()
    print("Nearby Search Response:", response)  # Debugging

    if "results" in response and response["results"]:
        place_id = response["results"][0]["place_id"]
        print(f"Found place_id: {place_id}")  # Debugging

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "key": GOOGLE_API_KEY,
            "place_id": place_id,
            "fields": "rating,user_ratings_total"
        }

        details_response = requests.get(details_url, params=details_params).json()
        print("Place Details Response:", details_response)  # Debugging
        
        result = details_response.get("result", {})
        if "rating" in result:
            rating = result["rating"]
            user_ratings_total = result.get("user_ratings_total", 0)
            return f"{rating} ⭐ ({user_ratings_total} reviews)"
    
    return "No rating available"



def get_charging_stations(latitude, longitude, max_results):
    api_url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "key": "193a3fa5-e53f-47ee-b104-4480bef3dd5b", 
        "latitude": latitude,
        "longitude": longitude,
        "maxresults": max_results,
        "distanceunit": "KM"
    }
    
    response = requests.get(api_url, params=params)
    data = response.json()
    
    stations = []
    for station in data:
        connections = station.get("Connections", [])
        num_ports = len(connections)  # Number of charging ports
        power_levels = [conn.get("PowerKW", "Unknown") for conn in connections]  # Power levels
        availability = station.get("StatusType", {}).get("Title", "Unknown")  # Availability status
        cost = station.get("UsageCost", "Not provided")  # Charging cost
        
        stations.append({
            "name": station["AddressInfo"]["Title"],
            "distance": station["AddressInfo"]["Distance"],
            "num_ports": num_ports,
            "power_levels": power_levels,
            "availability": availability,
            "cost": cost,
            "latitude": station["AddressInfo"]["Latitude"],
            "longitude": station["AddressInfo"]["Longitude"],
            "reviews": get_google_rating(station["AddressInfo"]["Title"], station["AddressInfo"]["Latitude"], station["AddressInfo"]["Longitude"]) 
            })
    
    return stations

def recommend_charging_station_with_llama(stations, user_preference):
    """Generate an LLM-based recommendation using a local LLaMA model."""
    
    # Create a prompt with station details
    station_details = "\n".join([
        f"{i+1}. {s['name']} - Distance: {s['distance']} km, Power: {s['power_levels']} kW, "
        f"Rating: {s['reviews']}, Cost: {s['cost']}, Status: {s['availability']}"
        for i, s in enumerate(stations)
    ])

    prompt = f"""
    You are an AI assistant specializing in electric vehicle charging station recommendations.
    
    Based on the following stations:
    
    {station_details}

    The user prefers: {user_preference}
    
    Please recommend the best charging station, explaining why it's the best based on their preferences.
    """

    # Use Ollama's local LLaMA model to generate a response
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])

    # Extract and return the model's response
    return response['message']['content']


def recommend_charging_station(stations, user_preference):
    """Generate an LLM-based recommendation including ratings, availability, and power levels."""
    
    # Create a formatted string for each station
    station_details = []
    for s in stations:
        google_rating = get_google_rating(s["name"], s["latitude"], s["longitude"])  # Fetch only ratings

        # Append station details in a more user-friendly format
        station_details.append({
            "name": s['name'],
            "distance": round(s['distance'], 1),  # Round distance to 1 decimal place
            "power_levels": ', '.join(map(str, s['power_levels'])),
            "rating": google_rating,
            "cost": s['cost'],
            "availability": s['availability'],
            "num_ports": s['num_ports'],
            "reviews": s['reviews']
        })

    # Calculate scores for each station based on user preference
    for station in station_details:
        station["score"] = calculate_station_score(station, user_preference)

    # Sort stations by weighted score
    ranked_stations = sorted(station_details, key=lambda x: x["score"], reverse=True)

    # Build the formatted output with more space
    recommendation_text = f"""
    ### Recommended Charging Stations

    Here are the top **charging stations** near your location based on **distance**, **rating**, **power**, and **availability**:

    """

    # Add each station with more space for readability
    for i, station in enumerate(ranked_stations, 1):
        recommendation_text += f"""
        **{i}. {station['name']}**:
        - **Distance**: {station['distance']} km
        - **Power**: {station['power_levels']} kW
        - **Rating**: {station['rating']}
        - **Cost**: {station['cost']}
        - **Status**: {station['availability']}

        """

    # Top Recommendation section
    top_station = ranked_stations[0] if ranked_stations else None
    recommendation_text += f"""
    ### **Top Recommendation:**

    **{top_station['name']}**:
    - **Distance**: {top_station['distance']} km
    - **Power**: {top_station['power_levels']} kW
    - **Rating**: {top_station['rating']}
    - **Cost**: {top_station['cost']}
    - **Status**: {top_station['availability']}
    """

    return recommendation_text


def calculate_station_score(station, user_preference):
    """Calculate weighted score for ranking charging stations based on user preferences (allowing multiple factors)."""

    # Normalize user preference and extract multiple factors
    user_preference = user_preference.lower()
    
    # Detecting different aspects of preference
    prefers_distance = any(term in user_preference for term in ["distance", "close", "near", "km"])
    prefers_price = any(term in user_preference for term in ["price", "cost", "free", "cheap"])
    prefers_power = any(term in user_preference for term in ["fast charging", "power", "high power"])
    prefers_rating = any(term in user_preference for term in ["best", "rating", "reviews", "top rated"])

    # Default weight distribution (equal weight if no preference is given)
    base_weight = 1 / 5  # 20% for each factor
    weights = {
        "distance": base_weight,
        "price": base_weight,
        "power": base_weight,
        "availability": base_weight,
        "rating": base_weight
    }

    # Adjust weights dynamically based on user preferences
    selected_factors = {
        "distance": prefers_distance,
        "price": prefers_price,
        "power": prefers_power,
        "rating": prefers_rating
    }

    selected_count = sum(selected_factors.values())  # Count active preferences

    if selected_count > 0:
        user_weight = 0.85  # 85% of weight goes to selected factors
        default_weight = (1 - user_weight) / 2  # Remaining 15% goes to availability & rating if not selected
        
        for key in weights:
            if key in selected_factors and selected_factors[key]:
                weights[key] = user_weight / selected_count  # Distribute weight among selected preferences
            else:
                weights[key] = default_weight  # Give small default weight

    # Normalize cost score
    cost_value = station['cost'] if station['cost'] is not None else ''
    if cost_value.lower() == "free":
        cost_score = 1  # Free stations get the highest score
    else:
        try:
            cost_score = 1 / (1 + float(cost_value.replace('$', '').replace(' ', '')))  # Normalize cost
        except (ValueError, TypeError):  
            cost_score = 0.5  # Default score for unknown costs

    # Normalize distance score (closer is better)
    distance_score = 1 / (1 + station['distance'])

    # Extract and use the highest power level
    power_values = [
        float(power) for power in str(station['power_levels']).replace("kW", "").split(",") if power.strip().replace('.', '', 1).isdigit()
    ]
    max_power = max(power_values) if power_values else 0  # Get highest power
    power_score = max_power / 250  # Normalize with max 250 kW assumption

    # Boost power score if fast charging is preferred
    if prefers_power:
        power_score *= 3  # Prioritize power when "fast charging" is requested

    # Normalize availability score
    availability_score = station['num_ports'] / 10  # Normalize assuming max 10 ports

    # Normalize rating score correctly
    rating_value = 0
    if "⭐" in station['rating']:  # Extracts numeric rating from formatted text
        rating_value = float(station['rating'].split("⭐")[0].strip())

    rating_score = rating_value / 5  # Normalize assuming 5-star max

    # Boost rating score if high rating is preferred
    if prefers_rating:
        rating_score *= 2  # Prioritize rating when "high rating" is requested

    # Weighted Sum
    weighted_score = (
        weights["distance"] * distance_score + 
        weights["price"] * cost_score + 
        weights["power"] * power_score + 
        weights["availability"] * availability_score + 
        weights["rating"] * rating_score
    )
    
    return weighted_score


# Example usage
latitude = 50.640054655177515 # Example: Kamloops, Canada
longitude = -120.37892698241129

stations = get_charging_stations(latitude, longitude, 10)
if stations:
    recommendation = recommend_charging_station(stations, "fast charging high rated")
    print(recommendation)
    recommendation_llama = recommend_charging_station_with_llama(stations, "fast charging high rated")
    print(recommendation_llama)
else:
    print("No available charging stations found.")
