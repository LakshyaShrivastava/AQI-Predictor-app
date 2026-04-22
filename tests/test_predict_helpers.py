"""Tests for scripts.predict_helpers (AQI conversion and OWM client)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from scripts.predict_helpers import get_historical_pm25, pm25_to_aqi


@pytest.mark.parametrize(
    "pm25,expected",
    [
        (0.0, 0),
        (12.0, 50),
        (12.1, 51),
        (35.4, 100),
        (35.5, 101),
        (55.4, 150),
        (55.5, 151),
        (150.4, 200),
        (150.5, 201),
        (250.4, 300),
        (250.5, 301),
        (500.4, 500),
        (500.5, 501),
    ],
)
def test_pm25_to_aqi_category_boundaries(pm25, expected):
    assert pm25_to_aqi(pm25) == expected


def test_pm25_to_aqi_invalid_returns_none():
    assert pm25_to_aqi(None) is None
    assert pm25_to_aqi(-1.0) is None
    assert pm25_to_aqi("bad") is None


def test_get_historical_pm25_success_averages_list():
    fake_json = {
        "list": [
            {"components": {"pm2_5": 10.0}},
            {"components": {"pm2_5": 20.0}},
        ]
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_json
    mock_resp.raise_for_status = MagicMock()

    with patch("scripts.predict_helpers.requests.get", return_value=mock_resp):
        out = get_historical_pm25(37.0, -122.0, datetime(2024, 6, 15), "fake-key")

    assert out == pytest.approx(15.0)


def test_get_historical_pm25_empty_list_returns_none():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"list": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("scripts.predict_helpers.requests.get", return_value=mock_resp):
        out = get_historical_pm25(37.0, -122.0, datetime(2024, 6, 15), "fake-key")

    assert out is None


def test_get_historical_pm25_request_error_returns_none():
    import requests

    with patch(
        "scripts.predict_helpers.requests.get",
        side_effect=requests.exceptions.RequestException("network"),
    ):
        out = get_historical_pm25(37.0, -122.0, datetime(2024, 6, 15), "fake-key")

    assert out is None
