# SearchR1 Replication Setup Instructions 

Reference: [SearchR1](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md) and [SGLang Instructions](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md). 

## Retriever environments 
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

## Download the Index
```bash
conda activate retriever

local_dir=~/data
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

## Start the Local Flat e5 Retrieval Server 

GPU version 
```bash
conda activate retriever

bash examples/search/retriever/retrieval_launch.sh 
```

## Prepare Datasets 
```bash
python examples/search/searchr1_dataset.py --local_dir $local_dir
```