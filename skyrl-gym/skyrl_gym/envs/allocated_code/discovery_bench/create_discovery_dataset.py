import pandas as pd
import os
import json
import glob
import random
from typing import Dict, List, Any


def make_discovery_prefix(dp: Dict[str, Any]) -> str:
    """Create the prompt prefix for DiscoveryBench tasks"""
    query = dp['query']
    dataset_description = dp['dataset_description']
    columns_info = dp['columns_info']
    domain_knowledge = dp['domain_knowledge']
    workflow_tags = dp['workflow_tags']
    
    prefix = f"""
You are a discovery agent who can execute Python code to analyze datasets and generate scientific hypotheses.

TASK: Analyze the provided dataset to answer a research query and generate a scientific hypothesis.

DATASET INFORMATION:
{dataset_description}

COLUMNS:
{columns_info}

{f"DOMAIN KNOWLEDGE: {domain_knowledge}" if domain_knowledge else ""}
{f"WORKFLOW TAGS: {workflow_tags}" if workflow_tags else ""}

QUERY: {query}

DATASET LOCATION:
{dp['dataset_path']}

INSTRUCTIONS:
1. Load and explore the dataset using Python (use the dataset location above)
2. Perform statistical analysis to find relationships between variables
3. Analyze the data to answer the research query
4. Provide insights and conclusions based on your analysis
"""
    return prefix


def load_synthetic_data(split: str) -> List[Dict[str, Any]]:
    """Load synthetic data for given split (train/dev/test)"""
    dataset = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(base_dir, f"data_folder/synth/{split}/*/metadata_*.json")
    files = glob.glob(pattern)
    
    for metadata_path in files:
        try:
            # Try UTF-8 first, fallback to latin-1 for encoding issues
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except UnicodeDecodeError:
                with open(metadata_path, 'r', encoding='latin-1') as f:
                    metadata = json.load(f)
            
            # Get dataset directory
            dataset_dir = os.path.dirname(metadata_path)
            csv_path = os.path.join(dataset_dir, 'data.csv')
            
            if not os.path.exists(csv_path):
                continue
                
            # Extract information
            domain = metadata['domain']
            datasets_info = metadata['datasets']
            
            dataset_desc = datasets_info[0]['description']
            columns = datasets_info[0]['columns']
            
            # Format columns info
            columns_info = ""
            for col in columns:
                columns_info += f"- {col['name']}: {col['description']}\n"
            
            # Get queries
            queries = metadata['queries']
            if isinstance(queries, list) and queries:
                query_data = queries[0]
                query = query_data['question']
                true_hypothesis = query_data['true_hypothesis']
                
                example = {
                    'query': query,
                    'dataset_description': f"Domain: {domain}\nDataset: {dataset_desc}",
                    'columns_info': columns_info.strip(),
                    'domain_knowledge': '',
                    'workflow_tags': '',
                    'dataset_path': csv_path,
                    'true_hypothesis': true_hypothesis,
                    'true_workflow': '',
                    'metadata_type': 'synth',
                    'source': f'synth_{split}'
                }
                dataset.append(example)
                
        except Exception as e:
            print(f"Error processing {metadata_path}: {e}")
            continue
    
    return dataset


def load_real_data(split: str) -> List[Dict[str, Any]]:
    """Load real data for given split (train/test)"""
    dataset = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(base_dir, f"data_folder/real/{split}/*/metadata_*.json")
    files = glob.glob(pattern)
    
    for metadata_path in files:
        try:
            # Try UTF-8 first, fallback to latin-1 for encoding issues
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except UnicodeDecodeError:
                with open(metadata_path, 'r', encoding='latin-1') as f:
                    metadata = json.load(f)
            
            # Get dataset directory
            dataset_dir = os.path.dirname(metadata_path)
            
            domain = metadata['domain']
            domain_knowledge = metadata['domain_knowledge']
            workflow_tags = metadata['workflow_tags']
            datasets_info = metadata['datasets']
            
            # Some test files don't have workflow field
            try:
                workflow = metadata['workflow']
            except KeyError:
                workflow = ''
            
            dataset_desc = datasets_info[0]['description']
            csv_name = datasets_info[0]['name']
            csv_path = os.path.join(dataset_dir, csv_name)
            
            if not os.path.exists(csv_path):
                continue
            
            # Format columns info
            columns_info = ""
            raw_columns = datasets_info[0]['columns']['raw']
            for col in raw_columns:
                columns_info += f"- {col['name']}: {col['description']}\n"
            
            # Get queries
            queries = metadata['queries']
            if isinstance(queries, list) and queries:
                for query_list in queries:
                    if isinstance(query_list, list):
                        for query_data in query_list:
                            query = query_data['question']
                            # Test data doesn't have true_hypothesis
                            try:
                                true_hypothesis = query_data['true_hypothesis']
                            except KeyError:
                                true_hypothesis = ""
                            
                            example = {
                                'query': query,
                                'dataset_description': f"Domain: {domain}\nDataset: {dataset_desc}",
                                'columns_info': columns_info.strip(),
                                'domain_knowledge': domain_knowledge,
                                'workflow_tags': workflow_tags,
                                'dataset_path': csv_path,
                                'true_hypothesis': true_hypothesis,
                                'true_workflow': workflow,
                                'metadata_type': 'real',
                                'source': f'real_{split}'
                            }
                            dataset.append(example)
                            
        except Exception as e:
            print(f"Error processing {metadata_path}: {e}")
            continue
    
    return dataset


def load_discovery_data(split: str) -> List[Dict[str, Any]]:
    """Load all DiscoveryBench data for a given split"""
    dataset = []
    
    # Load synthetic data
    synth_data = load_synthetic_data(split)
    print(f"Loaded {len(synth_data)} synthetic {split} examples")
    dataset.extend(synth_data)
    
    # Load real data (only train exists for real data)
    if split == 'train':
        real_data = load_real_data(split)
        dataset.extend(real_data)
    
    return dataset


def create_discovery_dataset():
    """Create the full DiscoveryBench dataset"""
    print("Loading DiscoveryBench training data...")
    train_data = load_discovery_data('train')
    
    print("Loading DiscoveryBench validation data...")
    val_data = load_discovery_data('dev')
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")
    
    def process_example(example: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
        """Process a single example into the required format"""
        
        example_with_query = {
            'query': example['query'],
            'dataset_description': example['dataset_description'],
            'columns_info': example['columns_info'],
            'domain_knowledge': example['domain_knowledge'],
            'workflow_tags': example['workflow_tags'],
            'dataset_path': example['dataset_path']
        }
        
        user_content = make_discovery_prefix(example_with_query)
        
        system_prompt = {
            "role": "system",
            "content": "You are a discovery agent who can analyze datasets and generate scientific hypotheses."
        }
        
        # Expected output is just the true hypothesis (as the ground truth answer)
        expected_output = example['true_hypothesis']
        
        data = {
            "data_source": "discovery_bench",
            "prompt": [
                system_prompt,
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "env_class": "allocated_code",
            "reward_spec": {
                "method": "rule",
                "ground_truth": expected_output
            },
            "extra_info": {
                "query": example['query'],
                "true_hypothesis": example['true_hypothesis'],
                "true_workflow": example['true_workflow'],
                "dataset_path": example['dataset_path'],
                "metadata_type": example['metadata_type'],
                "source": example['source'],
                "split": split,
                "index": idx
            }
        }
        return data
    
    dataset = []
    
    # Process training data
    for idx, example in enumerate(train_data):
        processed = process_example(example, idx, 'train')
        dataset.append(processed)
    
    # Process validation data
    for idx, example in enumerate(val_data):
        processed = process_example(example, idx, 'validation')
        dataset.append(processed)
    
    return dataset


def save_discovery_dataset(output_dir=None):
    """Save the DiscoveryBench dataset"""
    if output_dir is None:
        # Save locally to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = create_discovery_dataset()
    
    train_data = [d for d in dataset if d['extra_info']['split'] == 'train']
    val_data = [d for d in dataset if d['extra_info']['split'] == 'validation']
    
    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_path = os.path.join(output_dir, "discovery_train.parquet")
    val_path = os.path.join(output_dir, "discovery_validation.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    
    # Print a random example
    if train_data:
        import json
        random_example = random.choice(train_data)
        print("\n" + "="*80)
        print("RANDOM TRAINING EXAMPLE (FULL):")
        print("="*80)
        print(json.dumps(random_example, indent=2))
        print("="*80)
    
    return train_path, val_path


if __name__ == "__main__":
    save_discovery_dataset()