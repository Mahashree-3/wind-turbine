\# Backend API — Wind Turbine Bearing Fault Diagnosis



FastAPI backend that serves sensor readings, fault diagnosis, and health status

derived from the trained bearing-fault model and the KAIST dataset.



\## Running the server



From the project root:



```bash

uvicorn backend.main:app --reload

```



The server starts at `http://127.0.0.1:8000` by default.



\## Endpoints



\### `GET /`

Basic liveness check.



\*\*Response 200\*\*

```json

{ "status": "API is running" }

```



\---



\### `GET /api/dashboard`

Returns sensor readings and diagnosis derived from the \*\*same\*\* randomly picked sample, so the values are consistent with each other.



\*\*Response 200\*\*

```json

{

&#x20; "sensors": {

&#x20;   "acoustic": 0.42,

&#x20;   "vibration": 0.031,

&#x20;   "temperature": 61.3,

&#x20;   "current": "coming soon"

&#x20; },

&#x20; "diagnosis": {

&#x20;   "...": "model prediction fields",

&#x20;   "true\_fault\_type\_for\_demo": "Inner Race"

&#x20; }

}

```



\*\*Error responses\*\*

\- `503 Service Unavailable` — dataset file is missing (run `data/preprocess.py` first)

\- `500 Internal Server Error` — unexpected failure while generating the sample



```json

{ "detail": "Dataset not found at <path>. Run data/preprocess.py first." }

```



\---



\### `GET /api/sensors`

Returns only the sensor readings block from a freshly picked sample.



\*\*Response 200\*\*

```json

{

&#x20; "acoustic": 0.42,

&#x20; "vibration": 0.031,

&#x20; "temperature": 61.3,

&#x20; "current": "coming soon"

}

```



\*\*Error responses:\*\* same as `/api/dashboard` (`503`, `500`).



\---



\### `GET /api/diagnosis`

Returns only the model's diagnosis for a freshly picked sample.



\*\*Response 200\*\*

```json

{

&#x20; "...": "model prediction fields",

&#x20; "true\_fault\_type\_for\_demo": "Normal"

}

```



\*\*Error responses:\*\* same as `/api/dashboard` (`503`, `500`).



\---



\### `GET /api/trend`

Returns a 30-day illustrative health-score trend (placeholder — real trend needs

run-to-failure time-series data, which the current dataset does not provide).



\*\*Response 200\*\*

```json

\[

&#x20; { "day": 1, "score": 98.3 },

&#x20; { "day": 2, "score": 96.7 }

]

```



\---



\### `GET /api/health`

Reports whether the backend, dataset, and model file are ready.



\*\*Response 200\*\*

```json

{

&#x20; "status": "ok",

&#x20; "dataset\_loaded": true,

&#x20; "model\_loaded": true

}

```



\## Error handling



\- If `data/processed/features.npz` is missing or fails to load, endpoints that

&#x20; depend on it (`/api/dashboard`, `/api/sensors`, `/api/diagnosis`) return

&#x20; `503` with a clear `detail` message instead of crashing the server.

\- Any other unexpected exception is caught and returned as `500` with a short

&#x20; message, so the API process stays alive during a demo.



\## Testing



Integration tests live in `test\_backend.py` at the project root.



```bash

\# terminal 1

uvicorn backend.main:app --reload



\# terminal 2

pip install requests

python test\_backend.py

```



All tests should print `\[PASS]`; the script exits non-zero if anything fails.

