import requests

def test_weather_list():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.8489,
        "longitude": 106.7720,
        "current": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m"],
    }
    response = requests.get(url, params=params)
    print("List response code:", response.status_code)
    if response.status_code == 200:
        print("List response data:", response.json().get("current"))
    else:
        print("List response error:", response.text)

def test_weather_string():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.8489,
        "longitude": 106.7720,
        "current": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m",
    }
    response = requests.get(url, params=params)
    print("String response code:", response.status_code)
    if response.status_code == 200:
        print("String response data:", response.json().get("current"))

test_weather_list()
test_weather_string()
