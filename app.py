import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium

# --- City to default coordinates mapping for auto-update ---
city_coords = {
    "Paris": (48.8566, 2.3522),
    "Rome": (41.9028, 12.4964),
    "Amsterdam": (52.3676, 4.9041),
    "Berlin": (52.5200, 13.4050),
    "Prague": (50.0755, 14.4378),
    "Barcelona": (41.3851, 2.1734),
    "Budapest": (47.4979, 19.0402),
    "Vienna": (48.2082, 16.3738),
    "Athens": (37.9838, 23.7275),
    "Istanbul": (41.0082, 28.9784),
    "Dublin": (53.3498, -6.2603),
    "Oslo": (59.9139, 10.7522),
    "Stockholm": (59.3293, 18.0686),
    "Copenhagen": (55.6761, 12.5683),
    "Brussels": (50.8503, 4.3517)
}

# --- Load model and scaler only once (cached) ---
@st.cache_resource
def load_model():
    model = joblib.load("lightgbm_airbnb_model.pkl")
    scaler = joblib.load("feature_scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.title("Airbnb Price Predictor")
st.markdown("Enter listing details below to predict the nightly price.")

# --- Sidebar inputs ---

# City selection - triggers auto-lat/lon update
city = st.sidebar.selectbox(
    "City",
    list(city_coords.keys()),
    key="city_input"
)

# Auto-update latitude/longitude if city changed or not set
if (
    "latitude_input" not in st.session_state
    or "longitude_input" not in st.session_state
    or st.session_state.get("last_city", None) != city
):
    default_lat, default_lon = city_coords[city]
    st.session_state["latitude_input"] = default_lat
    st.session_state["longitude_input"] = default_lon
    st.session_state["last_city"] = city  # store last chosen city

latitude = st.sidebar.number_input(
    "Latitude",
    value=st.session_state["latitude_input"],
    format="%.6f",
    key="latitude_input"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=st.session_state["longitude_input"],
    format="%.6f",
    key="longitude_input"
)

room_type = st.sidebar.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"],
    key="room_type_input"
)

minimum_nights = st.sidebar.slider(
    "Minimum Nights", 1, 30, 1,
    key="min_nights_input"
)

number_of_reviews = st.sidebar.slider(
    "Number of Reviews", 0, 500, 50,
    key="num_reviews_input"
)

reviews_per_month = st.sidebar.number_input(
    "Reviews per Month", value=0.0, step=0.1, format="%.2f",
    key="reviews_per_month_input"
)

calculated_host_listings_count = st.sidebar.slider(
    "Host Listings Count", 1, 10, 1,
    key="host_listings_count_input"
)

availability_365 = st.sidebar.slider(
    "Availability (days/year)", 0, 365, 200,
    key="availability_input"
)

number_of_reviews_ltm = st.sidebar.number_input(
    "Number of Reviews Last 12 Months",
    min_value=0, max_value=500, value=0, step=1,
    key="reviews_ltm_input"
)

# --- Encoding maps ---
room_type_map = {"Entire home/apt": 0, "Private room": 1, "Shared room": 2}
city_map = {c: i for i, c in enumerate(list(city_coords.keys()))}

# --- Prepare input data with exact features and order ---
input_dict = {
    "latitude": latitude,
    "longitude": longitude,
    "room_type": room_type_map[room_type],
    "minimum_nights": minimum_nights,
    "number_of_reviews": number_of_reviews,
    "reviews_per_month": reviews_per_month,
    "calculated_host_listings_count": calculated_host_listings_count,
    "availability_365": availability_365,
    "number_of_reviews_ltm": number_of_reviews_ltm,
    "city": city_map[city]
}

columns_order = [
    'latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
    'number_of_reviews_ltm', 'city'
]

input_df = pd.DataFrame([input_dict], columns=columns_order)

# --- Scale input and predict ---
X_scaled = scaler.transform(input_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=columns_order)

predicted_price = model.predict(X_scaled_df)[0]

st.subheader("Predicted Nightly Price")
st.write(f"**${predicted_price:,.2f}**")

# --- Optional: display top 5 feature importances ---
if st.checkbox("Show top 5 feature importances"):
    importances = model.feature_importances_
    feat_names = input_df.columns
    top_idx = np.argsort(importances)[::-1][:5]
    fi_df = pd.DataFrame({
        "Feature": feat_names[top_idx],
        "Importance": importances[top_idx]
    })
    st.bar_chart(fi_df.set_index("Feature"))

# --- Folium map display ---
st.subheader("Selected Listing on Map")
m = folium.Map(location=[latitude, longitude], zoom_start=13)
folium.Marker([latitude, longitude], tooltip="Your Listing", popup="Selected listing").add_to(m)
st_folium(m, width=700, height=500)
