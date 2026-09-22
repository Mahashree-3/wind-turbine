from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mock_data

app = FastAPI(title="Wind Turbine Bearing Fault Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "API is running"}


@app.get("/api/sensors")
def get_sensors():
    return mock_data.SENSORS


@app.get("/api/diagnosis")
def get_diagnosis():
    return mock_data.DIAGNOSIS


@app.get("/api/trend")
def get_trend():
    return mock_data.TREND