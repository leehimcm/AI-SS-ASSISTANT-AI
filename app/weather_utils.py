import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

def load_json_dict(path: Path) -> Dict[str, Any]: 
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def load_json_any(path: Path) -> Any: 
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return None
    return None


# Load weather tasks once at module import
data_dir = Path(__file__).resolve().parent.parent / "data"
weather_tasks_path = data_dir / "weather_tasks.json"
weather_tasks_list = load_json_any(weather_tasks_path) or []
TASKS_BY_TITLE: Dict[str, Dict[str, Any]] = {
    item.get("title"): item for item in weather_tasks_list if item.get("title")
}

TH_WIND_MIN_VENT = 0.5
TH_WIND_MAX_VENT = 6.0
TH_CLOUD_CLEAR = 20
AQI_BAD_MIN = 4

RECO_MAP = {
    "vent_ok": [
        "실내 전체 환기"
    ],
    "air_bad": [
        "창틀 먼지 물티슈 청소"
    ],
    "heavy_rain": [
        "현관 신발장 환기 및 탈취",
        "현관 바닥 닦기",
        "창틀 먼지 닦기",
    ],
    "light_rain": [
        "실내 환기"
    ],
    "clear_dry": [
        "침구 표면 먼지 제거 (돌돌이)"
    ],
    "fallback": [
        "침구 표면 먼지 제거 (돌돌이)"
    ],
}


def parse_korean_address(address: str) -> str: 
    parts = [p for p in address.split() if p.endswith(("시", "도", "구"))]
    return ", ".join(parts)


def get_aqi(air: Dict[str, Any]) -> Optional[int]: 
    try:
        return int((air or {}).get("list", [])[0]["main"]["aqi"])
    except Exception:
        return None


def recommend_weather_tasks(
    weather: Dict[str, Any],
    air: Dict[str, Any],
    tasks_by_title: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]: 
    weather = weather or {}
    air = air or {}
    tasks_by_title = tasks_by_title or {}

    # weather id
    try:
        weather_id = int((weather.get("weather") or [{}])[0].get("id", 0))
    except Exception:
        weather_id = 0

    is_heavy_rain = 502 <= weather_id <= 504
    is_light_rain = 500 <= weather_id <= 501
    is_raining = is_heavy_rain or is_light_rain

    wind_speed = float((weather.get("wind") or {}).get("speed", 0.0))
    clouds = int((weather.get("clouds") or {}).get("all", 0) or 0)
    aqi = get_aqi(air)

    is_clear = clouds <= TH_CLOUD_CLEAR
    wind_ok_for_cross_vent = TH_WIND_MIN_VENT <= wind_speed <= TH_WIND_MAX_VENT
    air_bad = (aqi is not None) and (aqi >= AQI_BAD_MIN)

    selected: List[str] = []

    if wind_ok_for_cross_vent and not is_raining and not air_bad:
        selected += RECO_MAP["vent_ok"]

    if air_bad:
        selected += RECO_MAP["air_bad"]

    if is_heavy_rain:
        heavy_candidates = RECO_MAP["heavy_rain"]
        choice = random.choices(
            heavy_candidates,
            weights=[0.4, 0.4, 0.2],
            k=1,
        )[0]
        selected.append(choice)
    elif is_light_rain:
        selected += RECO_MAP["light_rain"]

    if is_clear and not is_raining and not air_bad:
        selected += RECO_MAP["clear_dry"]

    if not selected:
        selected += RECO_MAP["fallback"]

    # De-duplicate while preserving order
    titles = list(dict.fromkeys(selected))

    out: List[Dict[str, str]] = []
    for t in titles:
        obj = tasks_by_title.get(t)
        if not obj:
            continue
        desc = obj.get("description")
        if desc is None:
            continue
        out.append({"title": t, "description": desc})

    return out


def get_lat_lon_from_address(address: str, api_key: str) -> Tuple[float, float]: 
    parsed = parse_korean_address(address)
    geo_url = (
        "http://api.openweathermap.org/geo/1.0/direct"
        f"?q={parsed}&limit=1&appid={api_key}"
    )
    geo_response = requests.get(geo_url).json()
    if not geo_response:
        raise ValueError("Failed to resolve address to coordinates.")
    lat = float(geo_response[0]["lat"])
    lon = float(geo_response[0]["lon"])
    return lat, lon


def fetch_weather_and_air(
    lat: float,
    lon: float,
    api_key: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
    weather_url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&lang=kr&units=metric"
    )
    air_url = (
        "https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )

    weather_response = requests.get(weather_url).json()
    air_response = requests.get(air_url).json()
    return weather_response, air_response


def build_weather_recommendation(
    data_dir: Path,
    api_key: str,
) -> Dict[str, Any]: 
    user_info_path = data_dir / "user_info.json"
    user_info = load_json_dict(user_info_path)

    raw_address = user_info.get("user_address", "")
    if not raw_address:
        raise ValueError("주소를 찾을 수 없습니다.")

    address = parse_korean_address(raw_address)
    lat, lon = get_lat_lon_from_address(address, api_key)
    weather, air = fetch_weather_and_air(lat, lon, api_key)

    todos = recommend_weather_tasks(
        weather=weather,
        air=air,
        tasks_by_title=TASKS_BY_TITLE,
    )

    weather_desc: Optional[str] = None
    try:
        weather_desc = weather.get("weather", [{}])[0].get("description")
    except Exception:
        weather_desc = None

    return {
        "weather": weather_desc,
        "todos": todos,
    }


if __name__ == "__main__":
    pass
