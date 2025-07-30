# Shiroinu

## Prerequisites

Please download the data from [the Google Drive provided in Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place it somewhere on your machine.
Then, specify the path in `config/xxx.toml`.

## Execution Commands

Please define the settings in `config/xxx.toml` in advance.

```
python run.py configs/sample_mini_0.toml
```

## Note

- If a non-empty comma-separated string or a list is specified as the `data.white_list`, only the time series corresponding to those columns will be used. However, internally, regardless of the original column names, they will be renamed in the specified order as `y0, y1, y2, ...`.
