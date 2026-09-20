import datetime
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from strands import tool

import secret_store


WEATHER_API_URL = "https://api.weatherapi.com/v1/forecast.json"
MAX_FORECAST_DAYS = 14
REQUEST_TIMEOUT_SECONDS = 10


def _request_forecast(api_key: str, location: str, days: int) -> dict:
    query = urlencode({
        "key": api_key,
        "q": location,
        "days": days,
        "aqi": "no",
        "alerts": "no",
    })
    request = Request(
        f"{WEATHER_API_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "Agromemnon/1.0"},
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except HTTPError as error:
        try:
            error_body = json.load(error)
            message = error_body.get("error", {}).get("message", str(error))
        except (json.JSONDecodeError, OSError):
            message = str(error)
        raise RuntimeError(f"Weather API request failed: {message}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Weather API is unavailable: {error}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError("Weather API returned invalid JSON") from error


def _format_forecast(data: dict) -> str:
    location = data.get("location", {})
    forecast_days = data.get("forecast", {}).get("forecastday", [])
    result = {
        "source": "WeatherAPI.com forecast API",
        "source_url": WEATHER_API_URL,
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "location": {
            "name": location.get("name"),
            "region": location.get("region"),
            "country": location.get("country"),
            "latitude": location.get("lat"),
            "longitude": location.get("lon"),
        },
        "forecast": [],
    }

    for day in forecast_days:
        day_data = day.get("day", {})
        result["forecast"].append({
            "date": day.get("date"),
            "condition": day_data.get("condition", {}).get("text"),
            "temperature_c": {
                "minimum": day_data.get("mintemp_c"),
                "maximum": day_data.get("maxtemp_c"),
                "average": day_data.get("avgtemp_c"),
            },
            "rain_chance_percent": day_data.get("daily_chance_of_rain"),
            "snow_chance_percent": day_data.get("daily_chance_of_snow"),
            "total_precipitation_mm": day_data.get("totalprecip_mm"),
            "max_wind_kph": day_data.get("maxwind_kph"),
            "average_humidity_percent": day_data.get("avghumidity"),
            "uv_index": day_data.get("uv"),
        })

    return json.dumps(result, ensure_ascii=True)


@tool
def weather(location: str, days: int = 7) -> str:
    """Get the weather forecast for a location.

    Args:
        location: Place name or "latitude,longitude".
        days: Number of days to forecast.
    """
    api_key = secret_store.resolve("WEATHERAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "WEATHERAPI_KEY is not configured. Create a WeatherAPI.com account, then "
            "set WEATHERAPI_KEY locally or point WEATHERAPI_KEY_SECRET at a Secrets "
            "Manager secret for a deployed runtime."
        )
    if not isinstance(location, str) or not location.strip():
        raise ValueError("location must be a non-empty place name or latitude,longitude")
    if isinstance(days, bool) or not isinstance(days, int):
        raise ValueError("days must be an integer")
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_FORECAST_DAYS}")

    data = _request_forecast(api_key, location.strip(), days)
    return _format_forecast(data)
