"""Bounded weather lookup using fixed Open-Meteo endpoints."""

from __future__ import annotations

from typing import Any

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_MAX_CITY_CHARS = 100


async def fetch_weather(city: str) -> dict[str, Any]:
    """Return current weather for one bounded city name."""
    city = city.strip()
    if not city or len(city) > _MAX_CITY_CHARS:
        return {"city": city[:_MAX_CITY_CHARS], "error": "invalid_city"}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        try:
            geo = await client.get(GEOCODE_URL, params={"name": city, "count": 1})
            geo.raise_for_status()
            payload = geo.json()
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list) or not results:
                return {"city": city, "error": "not_found"}
            location = results[0]
            if not isinstance(location, dict):
                return {"city": city, "error": "invalid_response"}
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            if not isinstance(latitude, (int, float)) or not isinstance(
                longitude, (int, float)
            ):
                return {"city": city, "error": "invalid_response"}

            response = await client.get(
                FORECAST_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current_weather": True,
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                },
            )
            response.raise_for_status()
            forecast = response.json()
            current = (
                forecast.get("current_weather") if isinstance(forecast, dict) else None
            )
            if not isinstance(current, dict):
                return {"city": city, "error": "invalid_response"}
            return {
                "city": str(location.get("name") or city)[:_MAX_CITY_CHARS],
                "country": str(location.get("country") or "")[:_MAX_CITY_CHARS],
                "weather": current,
            }
        except (httpx.HTTPError, ValueError, TypeError):
            return {"city": city, "error": "weather_service_error"}
