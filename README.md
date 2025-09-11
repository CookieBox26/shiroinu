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
A report will be saved as `outputs/sample_traffic_mini_0/report.html`.

You can also specify the following options:

- `-s 0,1` : Skip tasks 0 and 1
- `-r` : Generate the report without running any tasks
- `-c` : Clear the log directory if it already exists
- `-q` : Suppress output to stdout
- `-i` : Do not embed graph images in the report (default: embedded)
- `-f png` : Set the format of graph images in the report to PNG (default: SVG)
- `--dpi 64` : Specify the resolution of graph images in the report (effective only when using PNG)
    - For example, `-f png --dpi 64` reduces the report file size, though the graphs will be lower quality.
- `--max_n_graph 20` : Limit the number of graph images in the report to 20 (default: 1000)
    - For tasks with `task_type = valid`, the number of graphs is `ceil(n_channel / 5)`, so reducing this upper limit can help decrease report generation time and file size.

### Note

- If a non-empty comma-separated string or a list is specified as the `data.white_list`, only the time series corresponding to those columns will be used. However, internally, regardless of the original column names, they will be renamed in the specified order as `y0, y1, y2, ...`.
- Some features are tested with unit tests. To run them, install with `uv sync --extra test` and run with `uv run pytest`.
