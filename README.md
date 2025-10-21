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

- Some features are tested with unit tests. To run them, install with `uv sync --extra test` and run with `uv run pytest`.
- If a non-empty comma-separated string or a list is specified as the `data.white_list`, only the time series corresponding to those columns will be used. However, internally, regardless of the original column names, they will be renamed in the specified order as `y0, y1, y2, ...`.

### 各クラスの説明

- **全データ管理クラス `TSDataManager`** : CSV ファイルから全期間のデータを読み込みデータフレームとして保持します。また、データ切り出し用の入出力長も保持します。
  - ホワイトリストが渡された場合は列を絞り込み、`timestamp, timestep, y0, y1, y2, ...` とします。
    - `timestep` は CSV ファイル内の全期間のデータに通して割り振られるステップ番号です (ステップ番号を扱うモデル向けです)。
  - 入力長や出力長は、指定範囲の切り出し時に考慮します。
    - 時系列予測では `N` 行のデータから取れるサンプル数は `N - 入力長 - 出力長 + 1` になります。例えば「前から 80% を学習用に、後ろ 20% を評価用にしたい」というとき、単に「データフレームの行の前から 80%、後ろ 20%」と切り出すのではなく、「前から 80% のサンプルに必要な行、後ろ 20% のサンプルに必要な行」を切り出すようにしています (∴ 学習用データフレームと評価用データフレームは一部重複します)。
    - 実際の推論用の入出力長はモデルに指定します。`TSDataManager` に指定する入出力長は、実際の推論用の入出力長さと同じかより長くなければなりません。
      - 入力長を変化させて比較したい場合、データ切り出し用の入出力長も変化させるとサンプルがずれてしまうので、データ切り出し用の入力長は、実験するであろう最も長い入力長にするのがおすすめです。
- **バッチ供給用データ保持クラス `TSDataSet`** : `TSDataManager` からデータフレームの指定範囲への参照を受け取り保持します。これを `torch.utils.data.DataLoader` に渡すことでバッチ `TSBatch` を供給するイテレータが得られます。また、スケーラー用に各列の平均, 標準偏差, 第1四分位, 第2四分位, 第3四分位も算出し保持します。
  - **バッチ `TSBatch`** は下記フィールドをもち、バッチ供給時に `torch.tensor` を対象デバイスに送ります。
    - `TSBatch.tsta` (`numpy.array`)  batch_size, seq_len
    - `TSBatch.tste` (`torch.tensor`)  batch_size, seq_len
    - `TSBatch.data` (`torch.tensor`)  batch_size, seq_len, n_channel
    - `TSBatch.tsta_future` (`numpy.array`)  batch_size, pred_len
    - `TSBatch.tste_future` (`torch.tensor`)  batch_size, pred_len
    - `TSBatch.data_future` (`torch.tensor`)  batch_size, pred_len, n_channel
      - 推論時に `TSBatch.data_future` を参照してはならないことはいうまでもありません。
- **基底モデルクラス `BaseModel`** : 全ての予測モデルはこれを継承してください。
  - 全ての予測モデルは `BaseModel.create(**kwargs, state_path=None)` によりインスタンス化してください。
    - あれば学習済みの重みをセットした上で、対象デバイスに送ります。
  - 全ての予測モデルは以下の 4 つの抽象メソッドを実装してください。
    - `extract_input(batch)` : バッチからそのモデルが要するフィールドを要する長さだけ抽出します。
      - **スケールされた入力を期待するモデルはここでスケールしてください。**
    - `forward(input_)` : `extract_input(batch)` の結果を受け取り、モデルの生の出力を返してください。
      - 必要に応じデバッグ情報を返すようにしても構いません。
    - `extract_target(batch)` : `forward(input_)` が目指すべきターゲットを抽出します。
      - **生の出力がスケールされた状態になるモデルは、ここでターゲット側もスケールしてください。**
    - `predict(batch)` : バッチから `batch.data_future[:, :pred_len]` を目指すように出力してください。多くの場合は以下のようになるはずです。
      - `self.forward(self.extract_input(batch))` : スケールしないモデルの場合。
      - `self.scaler.rescale(self.forward(self.extract_input(batch)))` : スケールするモデルの場合。
      - `self.scaler.rescale(self.forward(self.extract_input(batch))[0])` : さらにデバッグ情報も出力するモデルの場合。
  - 現在、以下のモデルが実装されています。
    - `SimpleAverage` : 重みを学習しません。1 周期のステップ数を与え、単純に過去 n 周期の平均で予測します。過去の周期ほど 1.0倍 → 0.9倍 → 0.81倍 と取り込む比率を減衰させることもできます (減衰係数は決め打ちで全チャネル共通です)。
    - `SimpleAverageTrainable` : `SimpleAverage` でチャネルごとに減衰係数を学習できる版です。
    - `DLinear` : 標準化スケーラで入力をスケールする DLinear です。
    - `DLinearIqr` : IQR スケーラで入力をスケールする DLinear です。
    - `PatchTSTIqr` : IQR スケーラで入力をスケールする PatchTST です。
- **基底損失クラス `BaseLoss`** : 全ての損失はこれを継承してください。
  - インスタンス化時に損失計算用の重みを対象デバイスに送ります。
  - 全ての損失は `forward(pred, true)` を実装してください。
  - 現在、以下の損失が実装されています。
    - `MSELoss`
    - `MAELoss`

### TODO
- チャネル間のスケールが異なる多変量時系列向けに、各系列をスケールしてから誤差を計算するような損失が必要です。
- PatchTST のマスク予測による事前学習は未実装です。
- バッチ供給の実装が非効率的な可能性があります。
