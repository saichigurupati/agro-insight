import streamlit as st
import google.generativeai as genai
from google.generativeai import GenerativeModel
import requests
from typing import Any, Dict
import os
import json
from datetime import datetime
from PIL import Image
import io


class StreamlitCallbackHandler:
    def __init__(self, container):
        self.container = container
        
    def on_text(self, text: str) -> None:
        self.container.markdown(text)

def validate_api_key(api_key: str, api_type: str) -> bool:
    """Validate API keys before using them"""
    if api_type == "weather":
        try:
            response = requests.get(
                f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}"
            )
            return response.status_code == 200
        except:
            return False
    elif api_type == "google":
        try:
            genai.configure(api_key=api_key)
            model = GenerativeModel('gemini-2.0-flash')
            response = model.generate_content("Test")
            return True
        except:
            return False
    return False

def initialize_genai(api_key: str) -> GenerativeModel:
    genai.configure(api_key=api_key)
    return GenerativeModel('gemini-2.0-flash')

def get_soil_analysis(model: GenerativeModel, query: str) -> str:
    prompt = f"""
    Analyze soil-related queries with expertise in:
    - Soil composition, structure, and texture analysis
    - pH levels measurement and implications for plant growth
    - Macro and micronutrient content evaluation
    - Organic matter content assessment
    - Soil improvement and amendment recommendations
    - Crop-specific soil requirements
    - Soil testing methods and interpretation
    - Soil conservation practices
    
    For non-soil queries, provide relevant expert agricultural or environmental insights.
    
    Question: {query}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating soil analysis: {str(e)}")

def get_weather_data(api_key: str, location: str) -> Dict[str, Any]:
    """
    Get weather data from OpenWeatherMap API
    Handles both city names and coordinate inputs
    """
    base_url = "http://api.openweathermap.org/data/2.5"
    
    try:
        # Check if location is coordinates (contains comma)
        if ',' in location:
            lat, lon = map(float, location.split(','))
            current_url = f"{base_url}/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            forecast_url = f"{base_url}/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        else:
            # Location is a city name
            current_url = f"{base_url}/weather?q={location}&appid={api_key}&units=metric"
            forecast_url = f"{base_url}/forecast?q={location}&appid={api_key}&units=metric"
        
        # Current weather
        current_response = requests.get(
            current_url,
            timeout=10
        )
        current_response.raise_for_status()
        
        # 5-day forecast (OpenWeatherMap free tier provides 5-day forecast with 3-hour steps)
        forecast_response = requests.get(
            forecast_url,
            timeout=10
        )
        forecast_response.raise_for_status()
        
        current_data = current_response.json()
        forecast_data = forecast_response.json()
        
        # Process forecast data to get daily values
        daily_forecast = []
        forecast_by_day = {}
        
        for item in forecast_data['list']:
            date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
            if date not in forecast_by_day:
                forecast_by_day[date] = {
                    'dt': item['dt'],
                    'temp': {
                        'max': item['main']['temp_max'],
                        'min': item['main']['temp_min']
                    },
                    'humidity': item['main']['humidity'],
                    'rain': item.get('rain', {}).get('3h', 0)
                }
            else:
                # Update max/min temperatures
                forecast_by_day[date]['temp']['max'] = max(
                    forecast_by_day[date]['temp']['max'],
                    item['main']['temp_max']
                )
                forecast_by_day[date]['temp']['min'] = min(
                    forecast_by_day[date]['temp']['min'],
                    item['main']['temp_min']
                )
                # Accumulate rain
                forecast_by_day[date]['rain'] += item.get('rain', {}).get('3h', 0)
        
        daily_forecast = list(forecast_by_day.values())
        
        return {
            "current": current_data,
            "forecast": {"list": daily_forecast}
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise Exception("Invalid or expired OpenWeatherMap API key. Please check your API key and try again.")
        else:
            raise Exception(f"HTTP Error: {str(e)}")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except json.JSONDecodeError:
        raise Exception("Invalid response from weather API")
    except Exception as e:
        raise Exception(f"Error fetching weather data: {str(e)}")
def get_crop_recommendations(model: GenerativeModel, soil_type: str, weather_data: Dict[str, Any]) -> str:
    try:
        current_temp = weather_data['current'].get('main', {}).get('temp', 'N/A')
        current_humidity = weather_data['current'].get('main', {}).get('humidity', 'N/A')
        current_precip = weather_data['current'].get('rain', {}).get('1h', 0)
        
        prompt = f"""
        Based on the following conditions, recommend suitable crops:
        Soil Type: {soil_type}
        Current Temperature: {current_temp}°C
        Humidity: {current_humidity}%
        Rainfall: {current_precip}mm
        
        Consider:
        - Seasonal requirements
        - Temperature tolerance
        - Water requirements
        - Soil compatibility
        - Local growing conditions
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating crop recommendations: {str(e)}")

def save_chat_history(messages):
    try:
        if not os.path.exists('chat_history'):
            os.makedirs('chat_history')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chat_history/chat_{timestamp}.json'
            
        with open(filename, 'w') as f:
            json.dump(messages, f, indent=2)
        return True
    except Exception:
        return False


def process_leaf_image(model: GenerativeModel, image) -> str:
    """Analyze leaf image for disease detection"""
    prompt = """
    Analyze this plant leaf image and provide:
    1. Disease identification if present
    2. Severity level
    3. Treatment recommendations
    4. Preventive measures
    Base analysis on visual symptoms, patterns, and discoloration.
    """
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        raise Exception(f"Error analyzing leaf image: {str(e)}")

def get_crop_info_with_images(model: GenerativeModel, crop_name: str) -> Dict[str, Any]:
    """Get detailed crop information"""
    prompt = f"""Provide detailed information about {crop_name} covering:
    1. Basic description (growth cycle, uses, importance)
    2. Growing conditions (soil, climate, water needs)
    3. Major varieties with key characteristics
    4. Common diseases with symptoms and management
    5. Key pests with control measures
    
    Format as descriptive paragraphs for each section."""
    
    try:
        response = model.generate_content(prompt)
        
        # Parse response into structured sections
        sections = {
            "Basics": "Overview and description of the crop",
            "Growing Conditions": "Required growing conditions",
            "Varieties": "Common varieties and characteristics",
            "Diseases": "Common diseases and management",
            "Pests": "Major pests and control measures"
        }
        
        info_parts = {}
        current_section = None
        current_content = []
        
        for line in response.text.split('\n'):
            if any(section in line for section in sections):
                if current_section:
                    info_parts[current_section] = '\n'.join(current_content).strip()
                    current_content = []
                current_section = next(k for k in sections if k in line)
            elif line.strip():
                current_content.append(line)
                
        if current_section and current_content:
            info_parts[current_section] = '\n'.join(current_content).strip()
            
        return info_parts
    except Exception as e:
        raise Exception(f"Error getting crop information: {str(e)}")

    
def get_agricultural_clarification(model: GenerativeModel, query: str) -> str:
    """Get AI-powered clarification for agricultural queries"""
    prompt = f"""
    Provide expert agricultural guidance on: {query}
    Include:
    - Detailed explanation
    - Scientific background
    - Practical recommendations
    - Related best practices
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error getting clarification: {str(e)}")


def main():
    st.set_page_config(
        page_title="Agricultural Assistant",
        page_icon="🌱",
        layout="wide"
    )
    
    # Initialize session state for API validation
    if 'google_api_valid' not in st.session_state:
        st.session_state.google_api_valid = False
    if 'weather_api_valid' not in st.session_state:
        st.session_state.weather_api_valid = False
    
    # Sidebar settings
    st.sidebar.header("Settings")
    google_api_key = st.sidebar.text_input("Enter Google API Key", type="password")
    weather_api_key = st.sidebar.text_input("Enter OpenWeatherMap API Key", type="password")
    
    # API key validation
    if google_api_key and not st.session_state.google_api_valid:
        with st.sidebar:
            with st.spinner("Validating Google API key..."):
                if validate_api_key(google_api_key, "google"):
                    st.session_state.google_api_valid = True
                    st.success("Google API key is valid!")
                else:
                    st.error("Invalid Google API key")
                    
    if weather_api_key and not st.session_state.weather_api_valid:
        with st.sidebar:
            with st.spinner("Validating OpenWeatherMap API key..."):
                if validate_api_key(weather_api_key, "weather"):
                    st.session_state.weather_api_valid = True
                    st.success("OpenWeatherMap API key is valid!")
                else:
                    st.error("Invalid OpenWeatherMap API key")
    
    if st.sidebar.button("Save Chat History"):
        if save_chat_history(st.session_state.get("messages", [])):
            st.sidebar.success("Chat history saved!")
        else:
            st.sidebar.error("Failed to save chat history")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Soil Analysis", "Weather Forecast", "Crop Recommendations", "Disease Detection", "Agricultural Help"])
    
    # Initialize models if API keys are valid
    if not st.session_state.google_api_valid:
        st.info("Please enter a valid Google API key in the sidebar to continue.")
        st.stop()
    
    try:
        model = initialize_genai(google_api_key)
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.stop()
    
    # Tab 1: Soil Analysis
    with tab1:
        st.header("🌱 Soil Analysis Expert")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": "👋 Hello! I'm your Soil Analysis Expert. I can help you with soil composition, pH levels, nutrient content, and improvement recommendations. What would you like to know about your soil?"
            }]
        
        # Display chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        
        # Chat input
        user_query = st.chat_input("Ask about soil analysis...")
        
        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            
            with st.chat_message("assistant"):
                container = st.container()
                try:
                    with st.spinner("Analyzing..."):
                        response = get_soil_analysis(model, user_query)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        container.write(response)
                except Exception as e:
                    container.error(f"Error generating response: {str(e)}")
    
    # Tab 2: Weather Forecast
    with tab2:
        st.header("🌤️ Weather Forecast")
        
        if not st.session_state.weather_api_valid:
            st.info("Please enter a valid OpenWeatherMap API key in the sidebar to view weather data.")
            st.stop()
        
        location_type = st.radio("Search by:", ["City Name", "Coordinates"])
        
        if location_type == "City Name":
            location = st.text_input("Enter city name:")
        else:
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitude:", -90.0, 90.0, 0.0)
            with col2:
                lon = st.number_input("Longitude:", -180.0, 180.0, 0.0)
            location = f"{lat},{lon}"
        
        if location:
            try:
                with st.spinner("Fetching weather data..."):
                    weather_data = get_weather_data(weather_api_key, location)
                
                # Current weather
                st.subheader("Current Weather")
                current = weather_data['current']['main']
                
                cols = st.columns(4)
                cols[0].metric("Temperature", f"{current.get('temp', 'N/A')}°C")
                cols[1].metric("Humidity", f"{current.get('humidity', 'N/A')}%")
                cols[2].metric("Wind", f"{weather_data['current'].get('wind', {}).get('speed', 'N/A')} m/s")
                cols[3].metric("Pressure", f"{current.get('pressure', 'N/A')} hPa")
                
                # Forecast
                st.subheader("7-Day Forecast")
                forecast = weather_data['forecast']['list']
                
                for day in forecast:
                    date = datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d')
                    with st.expander(date):
                        cols = st.columns(4)
                        cols[0].metric("Max Temp", f"{day['temp'].get('max', 'N/A')}°C")
                        cols[1].metric("Min Temp", f"{day['temp'].get('min', 'N/A')}°C")
                        cols[2].metric("Humidity", f"{day.get('humidity', 'N/A')}%")
                        cols[3].metric("Rain", f"{day.get('rain', 0)} mm")
                
            except Exception as e:
                st.error(str(e))
    
    # Tab 3: Crop Recommendations
    with tab3:
        st.header("🌾 Crop Recommendations")
        
        if not st.session_state.weather_api_valid:
            st.info("Please enter a valid OpenWeatherMap API key in the sidebar to get crop recommendations.")
            st.stop()
        
        soil_type = st.selectbox(
            "Select your soil type:",
            ["Sandy", "Clay", "Loamy", "Silt", "Peat", "Chalk", "Loam-Clay"]
        )
        
        location = st.text_input("Enter location for weather data:")
        
        if location and soil_type:
            try:
                with st.spinner("Generating recommendations..."):
                    weather_data = get_weather_data(weather_api_key, location)
                    recommendations = get_crop_recommendations(model, soil_type, weather_data)
                    
                    st.subheader("Recommended Crops")
                    st.write(recommendations)
                    
            except Exception as e:
                st.error(str(e))

    with tab4:
        st.header("🔍 Plant Disease Detection")
        
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            try:
                with st.spinner("Analyzing image..."):
                    analysis = process_leaf_image(model, image)
                    st.subheader("Analysis Results")
                    st.write(analysis)
            except Exception as e:
                st.error(str(e))

    # Tab 5: Agricultural Help
    with tab5:
        st.header("❓ Agricultural Assistance")
        
        # Crop Information Section
        st.subheader("Crop Information")
        crop_name = st.text_input("Enter crop name:")
        
        if crop_name:
            try:
                with st.spinner("Fetching information..."):
                    crop_info = get_crop_info_with_images(model, crop_name)
                    
                    for section, content in crop_info.items():
                        with st.expander(section):
                            st.markdown(content)
            
            except Exception as e:
                st.error(str(e))
        
        # Q&A Section
        st.subheader("Ask Agricultural Questions")
        query = st.text_input("Enter your query:")
        
        if query:
            try:
                with st.spinner("Getting answer..."):
                    answer = get_agricultural_clarification(model, query)
                    st.markdown(answer)
            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()