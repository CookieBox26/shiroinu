#!/bin/bash

# uv run python run.py configs/sample_traffic_mini_0.toml -c
# uv run python run.py configs/sample_traffic_mini_1.toml -c

uv run python run.py configs/sample_traffic_sa.toml -c -f png --dpi 64
uv run python run.py configs/sample_traffic_dlinear_0.toml -c -f png --dpi 64

# uv run python run.py configs/sample_weather_sa.toml -c

uv run python register.py
