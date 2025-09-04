# Datasets Directory

This directory contains scripts and data for processing research datasets into training format. The approach is **fail-fast** - never modify original repository data, always maintain a path back to the source.

## Structure

```
datasets/
├── repos/                     # Downloaded author repositories (DO NOT MODIFY)
│   ├── discoverybench/       # DiscoveryBench original repo  
│   └── QRData/               # QRData original repo
├── data/                     # Processed data files
│   ├── discoverybench/       # DiscoveryBench data copies
│   └── qrdata/              # QRData data copies
├── data_parquet/            # Final training datasets
│   ├── combined_train.parquet
│   └── combined_validation.parquet
├── move_discoverybench.py   # DiscoveryBench processing script
├── move_qr.py              # QRData processing script
├── create_combined_dataset.py  # Main training data creation
├── discoverybench.json     # Processed DiscoveryBench data
└── qrdata.json            # Processed QRData data
```

## Usage Workflow

### 1. Download Original Repositories
Place the original author repositories in `repos/`:
- Download DiscoveryBench repo → `repos/discoverybench/`
- Download QRData repo → `repos/QRData/`

### 2. Process Individual Datasets

**DiscoveryBench:**
```bash
python move_discoverybench.py
```
- Copies `real/train` and `synth/train` data from repo to `data/discoverybench/`
- Processes metadata files and creates standardized `discoverybench.json`
- Transforms data paths to absolute paths

**QRData:**
```bash
python move_qr.py
```
- Copies benchmark data from repo to `data/qrdata/data/`
- Transforms `QRData.json` to standardized format → `qrdata.json`
- Updates file paths to absolute paths

### 3. Create Training Dataset
```bash
python create_combined_dataset.py
```
- Loads processed JSON files (`discoverybench.json`, `qrdata.json`)
- Creates dataset-specific prompts and formats
- Combines datasets with 80/20 train/validation split
- Outputs final parquet files to `data_parquet/`

## Data Formats

### Standardized JSON Structure
Both datasets are transformed into this common format:
```json
{
  "context": "Dataset description and domain info",
  "question": "The analytical question to answer",
  "answer": "Ground truth answer/hypothesis",
  "data": ["/absolute/path/to/data/file.csv"],
  "metadata": {
    "domain": "research domain",
    "columns_info": "detailed column descriptions",
    "dataset_descriptions": ["dataset descriptions"]
  }
}
```

### Final Training Format
Training data includes:
- `data_source`: "qrdata" or "discoverybench"
- `prompt`: System and user messages with full analysis instructions
- `env_class`: "allocated_code"
- `reward_spec`: Rule-based evaluation with ground truth
- `extra_info`: Original metadata and indexing

## Key Features

- **Fail-fast approach**: Never use `.get()` on dictionaries, let exceptions propagate
- **Preserve originals**: Never modify files in `repos/` directory
- **Reproducible**: Fixed random seed (41) for consistent splits
- **Dataset-specific prompting**: Different prompt templates for QR vs DiscoveryBench tasks
- **Combined training**: Mixed dataset training with proper source tracking

## Output
- Combined datasets with 80/20 train/validation split
- Format: Parquet files for efficient loading

## Docker Integration

The `data/` folder can be mounted into Docker executor environments using the compose generation script:

```bash
cd ../executors/
python generate_compose_100.py \
  -n 120 \
  --types "executor-prebuilt:120" \
  -m /data/fede/SkyRL/datasets/data:/data:ro
```

This enables:
- DiscoveryBench data accessible at `/data/discoverybench/`
- QRData accessible at `/data/qrdata/`
- Analysis code can directly reference the absolute paths specified in the JSON files
