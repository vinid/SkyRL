import json
import shutil
import os
import glob

from pathlib import Path
from typing import Dict, List, Any



def load_synthetic_data(data_dir: Path) -> List[Dict[str, Any]]:
    """Load synthetic data - copied from create_discovery_dataset.py"""
    dataset = []
    pattern = os.path.join(str(data_dir), "*/metadata_*.json")
    files = glob.glob(pattern)
    
    for metadata_path in files:
        # Try UTF-8 first, fallback to latin-1 for encoding issues
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except UnicodeDecodeError:
            with open(metadata_path, 'r', encoding='latin-1') as f:
                metadata = json.load(f)
        
        # Get dataset directory
        dataset_dir = os.path.dirname(metadata_path)
        dataset_name = os.path.basename(dataset_dir)
        
        # Extract information
        domain = metadata['domain']
        datasets_info = metadata['datasets']
        
        # Combine information from all datasets
        dataset_descriptions = []
        columns_info = ""
        dataset_paths = []
        
        for i, dataset_info in enumerate(datasets_info):
            dataset_descriptions.append(f"Dataset {i+1}: {dataset_info['description']}")
            columns = dataset_info['columns']
            columns_info += f"\n=== Dataset {i+1}: {dataset_info.get('name', f'dataset_{i+1}')} ===\n"
            for col in columns:
                columns_info += f"- {col['name']}: {col['description']}\n"
            
            # For synthetic data, typically use 'data.csv' but check if name is specified
            csv_name = dataset_info.get('name', 'data.csv')
            full_path = f"/data/discoverybench/synth/train/{dataset_name}/{csv_name}"
            dataset_paths.append(full_path)
        
        dataset_desc = "\n".join(dataset_descriptions)
        # Use all available dataset paths, separated by newlines
        combined_dataset_path = "\n".join(dataset_paths)
        
        # Get queries
        queries = metadata['queries']
        if isinstance(queries, list) and queries:
            for query_data in queries:
                query = query_data['question']
                true_hypothesis = query_data['true_hypothesis']
                
                entry = {
                    "context": f"Domain: {domain}\nDataset: {dataset_desc}",
                    "question": query,
                    "answer": true_hypothesis,
                    "data": dataset_paths,
                    "metadata": {
                        "domain": domain,
                        "columns_info": columns_info.strip(),
                        "domain_knowledge": '',
                        "workflow_tags": '',
                        "dataset_descriptions": dataset_descriptions
                    }
                }
                dataset.append(entry)
    
    return dataset


def load_real_data(data_dir: Path) -> List[Dict[str, Any]]:
    """Load real data - copied from create_discovery_dataset.py"""
    dataset = []
    pattern = os.path.join(str(data_dir), "*/metadata_*.json")
    files = glob.glob(pattern)
    
    for metadata_path in files:
        # Try UTF-8 first, fallback to latin-1 for encoding issues
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except UnicodeDecodeError:
            with open(metadata_path, 'r', encoding='latin-1') as f:
                metadata = json.load(f)
        
        # Get dataset directory
        dataset_dir = os.path.dirname(metadata_path)
        dataset_name = os.path.basename(dataset_dir)
        
        domain = metadata['domain']
        domain_knowledge = metadata['domain_knowledge']
        workflow_tags = metadata['workflow_tags']
        datasets_info = metadata['datasets']
        
        # Some test files don't have workflow field
        try:
            workflow = metadata['workflow']
        except KeyError:
            workflow = ''
        
        # Combine information from all datasets
        dataset_descriptions = []
        columns_info = ""
        dataset_paths = []
        
        for i, dataset_info in enumerate(datasets_info):
            dataset_descriptions.append(f"Dataset {i+1}: {dataset_info['description']}")
            csv_name = dataset_info['name']
            full_path = f"/data/discoverybench/real/train/{dataset_name}/{csv_name}"
            dataset_paths.append(full_path)
                
            # Format columns info for this dataset
            columns_info += f"\n=== Dataset {i+1}: {csv_name} ===\n"
            raw_columns = dataset_info['columns']['raw']
            for col in raw_columns:
                columns_info += f"- {col['name']}: {col['description']}\n"
            
        dataset_desc = "\n".join(dataset_descriptions)
        # Use all available dataset paths, separated by newlines
        combined_dataset_path = "\n".join(dataset_paths)

        if len(dataset_paths) > 1:
            print(f"Warning: {metadata_path} has multiple dataset paths: {dataset_paths}")
        
        # Get queries
        queries = metadata['queries']
        if isinstance(queries, list) and queries:
            for query_list in queries:
                if isinstance(query_list, list):
                    for query_data in query_list:
                        query = query_data['question']
                        true_hypothesis = query_data['true_hypothesis']
                        
                        entry = {
                            "context": f"Domain: {domain}\nDataset: {dataset_desc}",
                            "question": query,
                            "answer": true_hypothesis,
                            "data": dataset_paths,
                            "metadata": {
                                "domain": domain,
                                "domain_knowledge": domain_knowledge,
                                "workflow_tags": workflow_tags,
                                "columns_info": columns_info.strip(),
                                "dataset_descriptions": dataset_descriptions,
                                "workflow": workflow
                            }
                        }
                        dataset.append(entry)
    
    return dataset

def move_discoverybench_data():
    base_dir = Path(__file__).parent
    discoverybench_repo_dir = base_dir / "repos" / "discoverybench"
    
    # Source directories
    real_train_dir = discoverybench_repo_dir / "discoverybench" / "real" / "train"
    synth_train_dir = discoverybench_repo_dir / "discoverybench" / "synth" / "train"
    
    # Destination directory
    discoverybench_data_dir = base_dir / "data" / "discoverybench"
    
    # Destination paths
    real_destination = discoverybench_data_dir / "real" / "train"
    synth_destination = discoverybench_data_dir / "synth" / "train" 
    discoverybench_json_destination = base_dir / "discoverybench.json"
    
    # Check if destinations already exist
    if real_destination.exists():
        raise FileExistsError(f"Directory {real_destination} already exists")
    
    if synth_destination.exists():
        raise FileExistsError(f"Directory {synth_destination} already exists")
        
    if discoverybench_json_destination.exists():
        raise FileExistsError(f"File {discoverybench_json_destination} already exists")
    
    # Create consolidated JSON
    print("Processing metadata files...")
    all_entries = []
    
    # Process real training data
    if real_train_dir.exists():
        print(f"Processing real training data from {real_train_dir}")
        real_entries = load_real_data(real_train_dir)
        all_entries.extend(real_entries)
        print(f"Found {len(real_entries)} real training entries")
        
        # Copy real training data
        print(f"Copying {real_train_dir} to {real_destination}")
        shutil.copytree(str(real_train_dir), str(real_destination))
    else:
        print(f"Directory {real_train_dir} does not exist")
    
    # Process synthetic training data  
    if synth_train_dir.exists():
        print(f"Processing synthetic training data from {synth_train_dir}")
        synth_entries = load_synthetic_data(synth_train_dir)
        all_entries.extend(synth_entries)
        print(f"Found {len(synth_entries)} synthetic training entries")
        
        # Copy synthetic training data
        print(f"Copying {synth_train_dir} to {synth_destination}")
        shutil.copytree(str(synth_train_dir), str(synth_destination))
    else:
        print(f"Directory {synth_train_dir} does not exist")
    
    # Save consolidated JSON
    if all_entries:
        print(f"Saving {len(all_entries)} entries to {discoverybench_json_destination}")
        with open(discoverybench_json_destination, 'w') as f:
            json.dump(all_entries, f, indent=2)
        print(f"Consolidated JSON saved to {discoverybench_json_destination}")
    else:
        print("No entries found to save")
    
    # Count files moved
    real_count = len(list(real_destination.rglob("*"))) if real_destination.exists() else 0
    synth_count = len(list(synth_destination.rglob("*"))) if synth_destination.exists() else 0
    
    print(f"Real training files moved: {real_count}")
    print(f"Synthetic training files moved: {synth_count}")
    print(f"Total files moved: {real_count + synth_count}")
    print(f"Total JSON entries created: {len(all_entries)}")

if __name__ == "__main__":
    move_discoverybench_data()
