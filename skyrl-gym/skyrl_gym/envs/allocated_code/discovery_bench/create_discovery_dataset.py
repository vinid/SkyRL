import pandas as pd
import os
import json
import glob
import random
from typing import Dict, List, Any


def convert_to_docker_path(local_path: str) -> str:
    """Convert local file path to Docker mount path"""
    # Replace the local data_folder path with /data
    local_data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_folder")
    if local_path.startswith(local_data_folder):
        return local_path.replace(local_data_folder, "/data")
    return local_path


def make_discovery_prefix(dp: Dict[str, Any]) -> str:
    """Create the prompt prefix for DiscoveryBench tasks"""
    query = dp['query']
    dataset_description = dp['dataset_description']
    columns_info = dp['columns_info']
    domain_knowledge = dp['domain_knowledge']
    workflow_tags = dp['workflow_tags']
    
    prefix = f"""
You are an expert data scientist and data analyst who tackles analytical challenges through systematic thinking and thorough investigation. 
For each task, you will receive a question along with file paths to relevant data and background information. Your analysis should make full use of these data sources.

TASK: Analyze the provided dataset to generate a scientific hypothesis that answers the query.

DATASET INFORMATION:
{dataset_description}

COLUMNS:
{columns_info}

{f"DOMAIN KNOWLEDGE: {domain_knowledge}" if domain_knowledge else ""}
{f"WORKFLOW TAGS: {workflow_tags}" if workflow_tags else ""}

QUERY: {query}

DATASET LOCATIONS (use full paths):
{dp['dataset_path']}

INSTRUCTIONS:
1. Load and explore the dataset(s) using Python (use the dataset locations above)
2. If multiple datasets are provided, analyze each one to find out what is relevant and consider how they relate to each other
3. Perform data preprocessing, statistical analysis and, when appropriate, apply additional methods such as regression modeling, hypothesis testing, time-series, or spatial analysis, or feature engineering to identify relationships between variables.
4. Where simple statistics are insufficient or the data does not contain enough information itself, attempt more advanced or alternative approaches (e.g., using regression, dimensionality reduction, robustness checks, or combining datasets).
5. Provide insights and conclusions based on your analysis and come up with a scientific hypothesis that answers the query in your final answer section.
6. Do one step at a time. Explore the data and then answer the query.
7. Do not use plotting libraries. You cannot see the plots.
8. When workflow tags are provided, you should use them to guide your analysis.
9. Only form your final answer when you have enough evidence.


You MUST use the following format for your response. Each step must follow this exact structure:

<reasoning>
Write clear reasoning about what you plan to do next and why. Be specific about your analytical approach.
</reasoning>
<python>
Write executable Python code here. Each code block should do ONE specific task.
Code must be complete and runnable. Include all necessary imports.
</python>
<information>
The output/results from your Python code will appear here.
This section is read-only - you cannot write here.
</information>

Repeat these blocks for each analysis step. When you reach your conclusion:

<answer>
Write your final scientific hypothesis here. Requirements:
1. Direct and concise answer to the query
2. Derived from the provided dataset
3. Clearly stating the context of hypothesis (if any), variables chosen (if any) and relationship between those variables (if any) including any statistical significance.
</answer>

Here is a concrete example:

Question: Which city has the most people in the dataset?

<reasoning>
First, I will load the dataset and examine its structure to understand what data we have available for analysis.
</reasoning>
<python>
import pandas as pd
df = pd.read_csv('/data/data.csv')
print(df.columns)
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
</python>
<information>
Index(['id', 'name', 'age', 'gender', 'city', 'country'], dtype='object')

First few rows:
   id    name  age gender       city country
0   1   Alice   25      F  New York     USA
1   2     Bob   30      M  Chicago     USA
2   3  Carol   35      F   Boston     USA

Basic statistics:
              id         age
count  100.0000  100.000000
mean    50.5000   32.456000
std     29.0115    8.234567
</information>
<reasoning>
Now that I understand the data structure, I will analyze the city distribution to identify population patterns.
</reasoning>
<python>
city_counts = df.groupby('city').size().sort_values(ascending=False)
print("City distribution:")
print(city_counts)
print("\nPercentage by city:")
print((city_counts / len(df) * 100).round(2))
</python>
<information>
City distribution:
New York      100
Los Angeles    50
Chicago        30
Houston        20
Miami          10
Name: count, dtype: int64

Percentage by city:
New York      47.62
Los Angeles   23.81
Chicago       14.29
Houston        9.52
Miami          4.76
Name: count, dtype: float64
</information>
<reasoning>
I can see that New York has the most people in the dataset.
</reasoning>
<answer>
New York is the most represented city, accounting for nearly half (47.62%) of all records with 100 people.
</answer>
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
                csv_name = dataset_info[]
                csv_path = os.path.join(dataset_dir, csv_name)
                
                if os.path.exists(csv_path):
                    dataset_paths.append(convert_to_docker_path(csv_path))
            
            if not dataset_paths:
                continue
                
            dataset_desc = "\n".join(dataset_descriptions)
            # Use all available dataset paths, separated by newlines
            combined_dataset_path = "\n".join(dataset_paths)
            
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
                    'dataset_path': combined_dataset_path,
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
            
            # Combine information from all datasets
            dataset_descriptions = []
            columns_info = ""
            dataset_paths = []
            
            for i, dataset_info in enumerate(datasets_info):
                dataset_descriptions.append(f"Dataset {i+1}: {dataset_info['description']}")
                csv_name = dataset_info['name']
                csv_path = os.path.join(dataset_dir, csv_name)
                
                if os.path.exists(csv_path):
                    dataset_paths.append(convert_to_docker_path(csv_path))
                    
                    # Format columns info for this dataset
                    columns_info += f"\n=== Dataset {i+1}: {csv_name} ===\n"
                    raw_columns = dataset_info['columns']['raw']
                    for col in raw_columns:
                        columns_info += f"- {col['name']}: {col['description']}\n"
                else:
                    print(f"Warning: CSV file not found: {csv_path}")
            
            if not dataset_paths:
                continue
                
            dataset_desc = "\n".join(dataset_descriptions)
            # Use all available dataset paths, separated by newlines
            combined_dataset_path = "\n".join(dataset_paths)
            
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
                                'dataset_path': combined_dataset_path,
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
        
        # Convert true hypothesis to golden_answers format
        golden_answers = [example['true_hypothesis']] if example['true_hypothesis'] else []
        
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
                "ground_truth": str(example['true_hypothesis'])
            },
            "extra_info": {
                "question": example['query'],
                "original_answers": golden_answers,
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