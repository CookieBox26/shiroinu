#!/bin/bash

# uv run python run.py configs/sample_traffic_mini_0.toml -r

# uv run python run.py configs/sample_traffic_sa.toml -r -f png --dpi 64
uv run python run.py configs/sample_traffic_dlinear_0.toml -f png --dpi 64

# uv run python run.py configs/sample_weather_sa.toml

uv run python register.py
