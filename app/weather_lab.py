import json
from pathlib import Path
import requests
import random

def parse_korean_address(address: str) -> str:
    region_parts = []
    parts = address.split()

    for p in parts:
        if p.endswith(("시", "도", "구")):
            region_parts.append(p)

    return ", ".join(region_parts)

def recommend_cleaning_tasks(weather: dict, air: dict, tasks_by_title: dict) -> list:
    
    # 임계값
    TH_WIND_MIN_VENT = 0.5
    TH_WIND_MAX_VENT = 6.0
    TH_CLOUD_CLEAR = 20
    AQI_BAD_MIN = 4

    def aqi_val(a):
        try:
            return int((a or {}).get("list", [])[0]["main"]["aqi"])
        except Exception:
            return None
 
    # weather.id 기반 강수 상태 판별
    try:
        weather_id = int((weather.get("weather") or [{}])[0].get("id", 0))
    except:
        weather_id = 0

    # 기상 코드 기반 비 여부/강도
    is_heavy_rain = 502 <= weather_id <= 504           # heavy ~ extreme rain
    is_light_rain = 500 <= weather_id <= 501           # light / moderate rain
    is_raining = is_heavy_rain or is_light_rain        # 비가 오고 있다면 True
    
    wind = float((weather or {}).get("wind", {}).get("speed", 0.0))
    clouds = int((weather or {}).get("clouds", {}).get("all", 0))
    aqi = aqi_val(air)
      
    is_clear = clouds <= TH_CLOUD_CLEAR
    wind_ok_for_cross_vent = TH_WIND_MIN_VENT <= wind <= TH_WIND_MAX_VENT
    air_bad = (aqi is not None) and (aqi >= AQI_BAD_MIN)

    RECO_MAP = {
        "vent_ok": [
            "실내 전체 환기"
        ],
        "air_bad": [
            "창틀 먼지 물티슈 청소"
        ],
        "heavy_rain": [
            "현관 신발장 환기 및 탈취", # 0.4
            "현관 바닥 닦기", # 0.4
            "창틀 먼지 닦기" # 0.2
        ],
        "light_rain": [
            "실내 환기"
        ],
        "clear_dry": [
            "침구 표면 먼지 제거 (돌돌이)"
        ],
        "fallback": [
            "침구 표면 먼지 제거 (돌돌이)"
        ]
    }

    # 제목 선택
    selected = []
    if wind_ok_for_cross_vent and not is_raining and not air_bad:
        selected += RECO_MAP["vent_ok"]
    if air_bad:
        selected += RECO_MAP["air_bad"]
    
    if is_heavy_rain:
        heavy_candidates = RECO_MAP["heavy_rain"]
        choice = random.choices(
            heavy_candidates,
            weights=[0.4, 0.4, 0.2],
            k=1
        )[0]
        selected.append(choice)    
    elif is_light_rain:
        selected += RECO_MAP["light_rain"]
    if is_clear and not is_raining and not air_bad:
        selected += RECO_MAP["clear_dry"]
    if not selected:
        selected += RECO_MAP["fallback"]

    # 중복 제거 유지
    seen = set()
    titles = []
    for t in selected:
        if t not in seen:
            titles.append(t)
            seen.add(t)

    # 최종 반환(title, description만; 저장된 텍스트 그대로)
    out = []
    for t in titles:
        obj = tasks_by_title.get(t)
        if not obj:
            continue
        desc = obj.get("description")
        if desc is None:
            continue
        out.append({"title": t, "description": desc})
    return out

def load_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    user_info_path = Path(__file__).resolve().parent.parent / "data" / "user_info.json"
    user_info = load_json(user_info_path)

    API_KEY = "6a96849159eb130473a048891638bed8"  # default key
    address = parse_korean_address(user_info.get("user_address", ""))

    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={address}&limit=1&appid={API_KEY}"
    geo_response = requests.get(geo_url).json()

    if not geo_response:
        print("주소를 찾을 수 없습니다. 다시 입력해주세요.")
        raise SystemExit(1)

    lat, lon = geo_response[0]['lat'], geo_response[0]['lon']
    lat, lon = 54.759989, -2.719648 # 영국 온흐림 - 맞바람환기
    lat, lon = 52.238911, 4.616836 # 네덜란드 튼구름 - fallback
    lat, lon = -6.944843, -56.914602 # 브라질 튼구름 - 돌돌이
    lat, lon = 22.419627, 114.293846



    from pprint import pprint
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&lang=kr&units=metric"
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    weather_response = requests.get(weather_url).json()
    air_response = requests.get(air_url).json()

    weather_tasks_path = Path(__file__).resolve().parent.parent / "data" / "weather_tasks.json"
    weather_tasks_list = load_json(weather_tasks_path)
    tasks_by_title = {item.get("title"): item for item in weather_tasks_list if item.get("title")}

    todos = recommend_cleaning_tasks(weather_response, air_response, tasks_by_title)

    # 지정된 포맷으로 출력
    weather_desc = None
    try:
        weather_desc = weather_response.get("weather", [{}])[0].get("description")
    except Exception:
        weather_desc = None

    result = {
        "weather": weather_desc,
        "todos": todos
    }

    pprint(weather_response)
    print(json.dumps(result, ensure_ascii=False, indent=2))
