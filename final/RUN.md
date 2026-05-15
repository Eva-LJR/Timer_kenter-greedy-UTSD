# How To Run

## 1. Build extension

In your conda env (example: `sl`):

```bash
python setup.py build_ext --inplace
```

If CUDA toolchain is available (`nvcc` + `cl.exe`), it builds CUDA path automatically.
If CUDA toolchain is missing, it falls back to CPU-only build.

## 2. Run filtering

```bash
python process_utsd.py
```

## 3. Data source priority

`process_utsd.py` uses local Arrow first:

- `UTSD-1G-0_1/utsd-train.arrow`

If this file exists, no HF download is needed.
If not, it downloads via:

- `load_dataset("thuml/UTSD", "UTSD-1G", split="train")`

## 4. Label-parallel CUDA (new)

The script can process multiple labels in parallel on one GPU.

Default workers: `4`.

Override workers with env var:

```bash
set CUDA_LABEL_WORKERS=2
python process_utsd.py
```

Notes:

- Parallel mode is enabled when `USE_CUDA=True` and `CUDA_LABEL_WORKERS > 1`.
- If VRAM usage is too high, reduce `CUDA_LABEL_WORKERS` and/or `PAIR_BATCH_SIZE`.

## 5. Progress display

You will see:

- Dataset grouping progress (`Grouping`)
- Label-level selection progress (`Selecting labels`)
- Label-level saving progress (`Saving`)
- CUDA pair progress inside each label (percentage + ETA)

Related knobs in `process_utsd.py`:

- `PAIR_BATCH_SIZE` (default `4096`)
- `CUDA_PROGRESS_EVERY_BATCHES` (default `50`)
- `CUDA_LABEL_WORKERS` (default `4`)
- `MIN_FILTER_SIZE` (default `1000`; labels below this keep all samples)

## 6. Ctrl+C behavior

`Ctrl+C` is supported.

Because heavy work is inside C++/CUDA loops, interruption is checked at loop/batch boundaries.
So interruption is not always immediate in the middle of one GPU batch.

If you want faster response to `Ctrl+C`, reduce:

- `PAIR_BATCH_SIZE` (for example `2048` or `1024`)

Smaller batch means more interrupt points, but slightly lower throughput.

## 7. Output location

Filtered files are written to:

- `UTSD-1G-0_1/<label>/*.npy`

Each file is shaped `(length, 1)`.
