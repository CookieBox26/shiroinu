# Shiroinu


- [Data preparation](#data-preparation)
- [Installing dependencies](#installing-dependencies)
- [Running commands](#running-commands)
- [Note](#note)


### Data Preparation

Please download the data from [the Google Drive provided in Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place it somewhere on your machine.
Then, specify the path in `config/xxx.toml`.

> **⚠️ Note**  
> Some column names in `weather.csv` may appear garbled.
> Please fix them in advance using [rename_weather_columns.py](rename_weather_columns.py).


### Installing dependencies

For **Linux with CUDA 12.4** or **Windows with CPU**, you can install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
# Example: install uv on Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Example: install uv on Windows
pip install uv

# Check version (tested with uv 0.8.3)
uv --version  # e.g. uv 0.8.3

# Install dependencies
uv sync
```

If your setup is different, edit `pyproject.toml`, delete `uv.lock`, and run the commands above.  
If you're unable to use uv, install the dependencies from `pyproject.toml` manually.


### Running commands

Please define the settings in `config/xxx.toml` in advance.

```
uv run python run.py configs/sample_traffic_mini_0.toml
```

The outputs will be generated under `outputs/sample_traffic_mini_0/`.  
A report will be saved as `report.html`.


### Note

- If a non-empty comma-separated string or a list is specified as the `data.white_list`, only the time series corresponding to those columns will be used. However, internally, regardless of the original column names, they will be renamed in the specified order as `y0, y1, y2, ...`.
- Some features are tested with unit tests. To run them, install with `uv sync --extra test` and run with `uv run pytest`.
