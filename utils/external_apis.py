"""
External API Integrations — RetailIQ
Set OPENWEATHER_API_KEY and CALENDARIFIC_API_KEY in your environment.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, Optional
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# BASE CONNECTOR
# ──────────────────────────────────────────────────────────────────────────────

class BaseAPIConnector:
    """Abstract base for external API connectors."""

    env_key: str = "API_KEY"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv(self.env_key, "")

    def is_connected(self) -> bool:
        return bool(self.api_key)

    def _not_implemented(self) -> dict:
        return {
            "status":  "stub",
            "message": (
                f"{self.__class__.__name__} not configured. "
                f"Set {self.env_key} environment variable."
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 1. OPENWEATHERMAP — Weather API
# ──────────────────────────────────────────────────────────────────────────────

class WeatherAPIConnector(BaseAPIConnector):
    """
    Fetches live temperature and weather condition.
    Free plan: 1,000 calls/day — https://openweathermap.org/api
    """

    env_key  = "OPENWEATHER_API_KEY"
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)

    def get_weather_features(
        self,
        city: str = "Mumbai",
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Return temperature + demand boost factor for the given city."""

        if not self.is_connected():
            return {
                "temperature_c":       28.0,
                "condition":           "clear",
                "is_hot":              True,
                "is_rainy":            False,
                "demand_boost_factor": 1.0,
                "source":              "default (no API key set)",
            }

        try:
            import httpx
            resp = httpx.get(
                self.BASE_URL,
                params={
                    "q":     city,
                    "appid": self.api_key,
                    "units": "metric",      # Celsius
                },
                timeout=5,
            )
            resp.raise_for_status()
            data      = resp.json()
            temp      = data["main"]["temp"]
            condition = data["weather"][0]["main"].lower()
            is_hot    = temp >= 30
            is_rainy  = condition in ("rain", "drizzle", "thunderstorm")
            boost     = self.temperature_to_demand_boost(temp)

            return {
                "temperature_c":       round(temp, 1),
                "condition":           condition,
                "is_hot":              is_hot,
                "is_rainy":            is_rainy,
                "demand_boost_factor": boost,
                "source":              "openweathermap",
            }

        except Exception as e:
            # Return safe defaults if API call fails
            return {
                "temperature_c":       28.0,
                "condition":           "unknown",
                "is_hot":              False,
                "is_rainy":            False,
                "demand_boost_factor": 1.0,
                "source":              f"error: {e}",
            }

    @staticmethod
    def temperature_to_demand_boost(temperature_c: float) -> float:
        """Higher temperature → more beverage demand."""
        if temperature_c >= 35:
            return 1.20   # +20%
        if temperature_c >= 30:
            return 1.12   # +12%
        if temperature_c >= 25:
            return 1.05   # +5%
        return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. CALENDARIFIC — Holiday / Festival API
# ──────────────────────────────────────────────────────────────────────────────

class HolidayAPIConnector(BaseAPIConnector):
    """
    Fetches Indian public holidays and festival dates.
    Free plan: 1,000 calls/month — https://calendarific.com/
    Falls back to hardcoded Indian festival windows when API is unavailable.
    """

    env_key  = "CALENDARIFIC_API_KEY"
    BASE_URL = "https://calendarific.com/api/v2/holidays"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)

    # Hardcoded fallback — used when API is down or key is missing
    FESTIVAL_WINDOWS = [
        (1,  range(1, 4)),    # New Year
        (3,  range(25, 31)),  # Holi
        (8,  range(15, 16)),  # Independence Day
        (10, range(1, 15)),   # Navratri / Dussehra
        (10, range(20, 31)),  # Diwali
        (11, range(1, 5)),    # Diwali aftermath
        (12, range(24, 32)),  # Christmas / New Year Eve
    ]

    def is_festival_day(self, check_date: Optional[date] = None) -> bool:
        """Return True if check_date is a public holiday or festival."""
        d = check_date or date.today()

        if self.is_connected():
            try:
                import httpx
                resp = httpx.get(
                    self.BASE_URL,
                    params={
                        "api_key": self.api_key,
                        "country": "IN",
                        "year":    d.year,
                        "month":   d.month,
                        "day":     d.day,
                        "type":    "national",
                    },
                    timeout=5,
                )
                data = resp.json()
                return len(data.get("response", {}).get("holidays", [])) > 0

            except Exception:
                pass  # fall through to hardcoded fallback

        # Hardcoded fallback
        for month, day_range in self.FESTIVAL_WINDOWS:
            if d.month == month and d.day in day_range:
                return True
        return False

    def get_upcoming_festivals(self, days_ahead: int = 30) -> list:
        """Return list of upcoming festival dates within the next N days."""
        return [
            {"date": str(date.today() + timedelta(days=i)), "is_festival": True}
            for i in range(days_ahead)
            if self.is_festival_day(date.today() + timedelta(days=i))
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 3. MARKETING API — No key needed (stub)
# ──────────────────────────────────────────────────────────────────────────────

class MarketingAPIConnector(BaseAPIConnector):
    """
    Returns campaign intensity for a product.
    No key configured — returns neutral defaults.
    Wire to Meta Ads / Google Ads / internal warehouse when ready.
    """

    env_key = "MARKETING_API_KEY"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = ""   # no key — using defaults

    def get_campaign_intensity(
        self,
        product_name: str,
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Returns neutral marketing defaults."""
        return {
            "campaign_active":     0,
            "total_spend":         0.5,
            "marketing_intensity": 0.5,
            "source":              "default (marketing API not configured)",
        }


# ──────────────────────────────────────────────────────────────────────────────
# ENRICHMENT BUNDLE — combines all three signals
# ──────────────────────────────────────────────────────────────────────────────

class DataEnrichmentService:
    """
    Aggregates weather, holiday, and marketing signals into
    a single dict that augments the prediction input.
    """

    def __init__(self):
        self.weather   = WeatherAPIConnector()
        self.holidays  = HolidayAPIConnector()
        self.marketing = MarketingAPIConnector()

    def enrich(
        self,
        product_name: str,
        city: str = "Mumbai",
        target_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Return enriched feature signals ready for the prediction pipeline."""
        d = target_date or date.today()

        weather_data = self.weather.get_weather_features(city, d)
        mkt_data     = self.marketing.get_campaign_intensity(product_name, d)

        return {
            "festival_flag":       int(self.holidays.is_festival_day(d)),
            "weekend_flag":        int(d.weekday() >= 5),
            "campaign_active":     mkt_data["campaign_active"],
            "total_spend":         mkt_data["marketing_intensity"],
            "weather_temperature": weather_data["temperature_c"],
            "weather_boost":       self.weather.temperature_to_demand_boost(
                                       weather_data["temperature_c"]
                                   ),
        }

    def status(self) -> Dict[str, str]:
        """Check which APIs are active vs using defaults."""
        return {
            "weather":   "✅ connected" if self.weather.is_connected()   else "⚠️  not connected",
            "holidays":  "✅ connected" if self.holidays.is_connected()  else "⚠️  using hardcoded fallback",
            "marketing": "⚠️  not configured (using defaults)",
        }