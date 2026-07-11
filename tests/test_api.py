#/tests/test_api.py
"""
Unit tests for the FastAPI prediction service.

This module tests the `/predict` endpoint by sending a valid request payload
and verifying the response structure.

Note: The test requires that the champion models (regressor@champion and
classifier@champion) are registered in MLflow. If they are not available,
the test may return a 503 error (models not loaded).
"""

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_predict_endpoint():
    """
    Test the `/predict` endpoint with a complete feature payload.

    The payload includes all features required by the model signatures:
    - year, quarter, month, day_of_month, day_of_week, distance_group,
      dep_time_blk, origin_airport_id, dest_airport_id,
      flight_number_marketing_airline, flight_number_operating_airline,
      tail_number, crs_dep_time, crs_arr_time, crs_elapsed_time, distance.

    Optional fields `flight_uid` and `ground_truth` are also provided
    to mimic a realistic request.

    Asserts:
        - HTTP status code is 200 (if models are loaded) or possibly 503
          (if models are not loaded). We check for a successful response
          only if models are available; otherwise, we skip or accept 503.
        - If 200, the response contains both regression and classification
          predictions.
    """
    # Complete payload with all required features
    payload = {
        "year": 2020,
        "quarter": 1,
        "month": 1,
        "day_of_month": 15,
        "day_of_week": 3,
        "distance_group": 5,
        "dep_time_blk": "0900-0959",
        "origin_airport_id": 12478,
        "dest_airport_id": 12892,
        "flight_number_marketing_airline": 1234,
        "flight_number_operating_airline": 1234,
        "tail_number": "N123AA",
        "crs_dep_time": 900,
        "crs_arr_time": 1200,
        "crs_elapsed_time": 180.0,
        "distance": 2475,
        "flight_uid": "test-flight-001",
        "ground_truth": {
            "arr_delay_minutes": 5.0,
            "arr_del15": 0
        }
    }

    response = client.post("/predict", json=payload)

    # If models are loaded, we expect 200; otherwise, 503 is acceptable.
    if response.status_code == 200:
        data = response.json()
        assert "regression_prediction" in data
        assert "classification_prediction" in data
        assert "classification_proba" in data
    elif response.status_code == 503:
        # Models not loaded – this is acceptable in a test environment.
        assert response.json()["detail"] == "Model pipelines not loaded."
    else:
        # Unexpected status code – fail the test.
        assert False, f"Unexpected status code: {response.status_code}"