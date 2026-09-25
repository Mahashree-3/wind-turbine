import sys
import requests

BASE_URL = "http://127.0.0.1:8000"

passed = 0
failed = 0


def check(condition, test_name, extra=""):
    global passed, failed
    if condition:
        print(f"[PASS] {test_name}")
        passed += 1
    else:
        print(f"[FAIL] {test_name} {extra}")
        failed += 1


def test_root():
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        check(r.status_code == 200, "GET / returns 200")
        data = r.json()
        check("status" in data, "GET / returns status field")
    except requests.exceptions.ConnectionError:
        check(False, "GET / reachable", "- Is the backend running? (uvicorn backend.main:app)")
        sys.exit(1)


def test_dashboard():
    r = requests.get(f"{BASE_URL}/api/dashboard", timeout=5)
    check(r.status_code == 200, "GET /api/dashboard returns 200")
    data = r.json()
    check("sensors" in data, "GET /api/dashboard has 'sensors' key")
    check("diagnosis" in data, "GET /api/dashboard has 'diagnosis' key")
    if "sensors" in data:
        sensors = data["sensors"]
        for field in ["acoustic", "vibration", "temperature", "current"]:
            check(field in sensors, f"GET /api/dashboard sensors has '{field}'")


def test_sensors():
    r = requests.get(f"{BASE_URL}/api/sensors", timeout=5)
    check(r.status_code == 200, "GET /api/sensors returns 200")
    data = r.json()
    for field in ["acoustic", "vibration", "temperature", "current"]:
        check(field in data, f"GET /api/sensors has '{field}'")


def test_diagnosis():
    r = requests.get(f"{BASE_URL}/api/diagnosis", timeout=5)
    check(r.status_code == 200, "GET /api/diagnosis returns 200")
    data = r.json()
    check(isinstance(data, dict), "GET /api/diagnosis returns a JSON object")


def test_trend():
    r = requests.get(f"{BASE_URL}/api/trend", timeout=5)
    check(r.status_code == 200, "GET /api/trend returns 200")
    data = r.json()
    check(isinstance(data, list), "GET /api/trend returns a list")
    check(len(data) == 30, "GET /api/trend returns 30 days")
    if data:
        check("day" in data[0] and "score" in data[0], "GET /api/trend items have 'day' and 'score'")


def test_health():
    r = requests.get(f"{BASE_URL}/api/health", timeout=5)
    check(r.status_code == 200, "GET /api/health returns 200")
    data = r.json()
    for field in ["status", "dataset_loaded", "model_loaded"]:
        check(field in data, f"GET /api/health has '{field}'")


def test_not_found():
    r = requests.get(f"{BASE_URL}/api/does-not-exist", timeout=5)
    check(r.status_code == 404, "GET /api/does-not-exist returns 404")


if __name__ == "__main__":
    print(f"Running backend tests against {BASE_URL} ...\n")

    test_root()
    test_dashboard()
    test_sensors()
    test_diagnosis()
    test_trend()
    test_health()
    test_not_found()

    print(f"\n{passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)
