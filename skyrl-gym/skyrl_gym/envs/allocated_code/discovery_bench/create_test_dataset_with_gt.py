#!/usr/bin/env python3
"""
Create test dataset for evaluating trained models on DiscoveryBench with proper ground truth matching.
"""

import os
import json
import glob
import random
import pandas as pd
from typing import List, Dict, Any, Tuple

from create_discovery_dataset import make_discovery_prefix


def load_ground_truth_mapping(csv_path: str) -> Dict[Tuple[str, int, int], str]:
    """Load ground truth from DiscoveryBench answer key CSV."""
    df = pd.read_csv(csv_path)
    ground_truth_map = {}
    
    for _, row in df.iterrows():
        key = (row['dataset'], row['metadataid'], row['query_id'])
        ground_truth_map[key] = row['gold_hypo']
    
    print(f"Loaded {len(ground_truth_map)} ground truth entries")
    return ground_truth_map


def load_real_test_data_with_gt(ground_truth_csv: str) -> List[Dict[str, Any]]:
    """Load real test data from data_folder/real/test/ with ground truth matching."""
    dataset = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load ground truth mapping
    ground_truth_map = load_ground_truth_mapping(ground_truth_csv)
    
    # Find all test directories
    test_dir = os.path.join(base_dir, "data_folder/real/test")
    test_datasets = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    print(f"Found {len(test_datasets)} test datasets: {test_datasets}")
    
    matched_count = 0
    total_queries = 0
    
    for dataset_name in test_datasets:
        dataset_path = os.path.join(test_dir, dataset_name)
        
        # Find all metadata files
        metadata_files = glob.glob(os.path.join(dataset_path, "metadata_*.json"))
        
        for metadata_path in metadata_files:
            # Extract metadata_id from filename
            filename = os.path.basename(metadata_path)
            metadata_id = int(filename.split('_')[1].split('.')[0])
            
            try:
                # Try UTF-8 first, fallback to latin-1 for encoding issues
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except UnicodeDecodeError:
                    with open(metadata_path, 'r', encoding='latin-1') as f:
                        metadata = json.load(f)
                
                domain = metadata.get('domain', '')
                domain_knowledge = metadata.get('domain_knowledge', '')
                workflow_tags = metadata.get('workflow_tags', '')
                datasets_info = metadata.get('datasets', [])
                
                if not datasets_info:
                    print(f"Warning: No datasets info in {metadata_path}")
                    continue
                
                dataset_desc = datasets_info[0].get('description', '')
                csv_name = datasets_info[0].get('name', '')
                csv_path = os.path.join(dataset_path, csv_name)
                
                if not os.path.exists(csv_path):
                    print(f"Warning: CSV file not found: {csv_path}")
                    continue
                
                # Format columns info
                columns_info = ""
                raw_columns = datasets_info[0].get('columns', {}).get('raw', [])
                for col in raw_columns:
                    columns_info += f"- {col.get('name', '')}: {col.get('description', '')}\n"
                
                # Get queries
                queries = metadata.get('queries', [])
                if isinstance(queries, list) and queries:
                    for query_group in queries:
                        if isinstance(query_group, list):
                            for query_idx, query in enumerate(query_group):
                                if isinstance(query, dict) and 'question' in query:
                                    total_queries += 1
                                    question = query['question']
                                    
                                    # Look up ground truth
                                    gt_key = (dataset_name, metadata_id, query_idx)
                                    ground_truth = ground_truth_map.get(gt_key, "")
                                    
                                    if ground_truth:
                                        matched_count += 1
                                        print(f"✓ Matched: {dataset_name}, metadata_{metadata_id}, query_{query_idx}")
                                    else:
                                        print(f"✗ No ground truth for: {dataset_name}, metadata_{metadata_id}, query_{query_idx}")
                                    
                                    # Create example
                                    example = {
                                        'query': question,
                                        'dataset_description': dataset_desc,
                                        'columns_info': columns_info.strip(),
                                        'domain_knowledge': domain_knowledge,
                                        'workflow_tags': workflow_tags,
                                        'dataset_path': csv_path,
                                        'true_hypothesis': ground_truth,
                                        'metadata_type': 'real',
                                        'source': dataset_name,
                                        'metadata_id': metadata_id,
                                        'query_id': query_idx
                                    }
                                    dataset.append(example)
                else:
                    print(f"Warning: No valid queries found in {metadata_path}")
                    
            except Exception as e:
                print(f"Error processing {metadata_path}: {e}")
                continue
    
    print(f"Total queries found: {total_queries}")
    print(f"Matched with ground truth: {matched_count}")
    print(f"Match rate: {matched_count/total_queries*100:.1f}%" if total_queries > 0 else "No queries found")
    
    return dataset


def create_test_dataset_with_gt(ground_truth_csv: str):
    """Create test dataset for evaluation with ground truth."""
    print("Loading real test data with ground truth...")
    test_data = load_real_test_data_with_gt(ground_truth_csv)
    print(f"Loaded {len(test_data)} test examples")
    
    def process_example(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """Process a single example into the required format."""
        
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
            "content": "You are a discovery agent who can analyze datasets and generate scientific hypotheses. /nothink"
        }
        
        # Use ground truth if available
        golden_answers = [example['true_hypothesis']] if example['true_hypothesis'] else []
        
        data = {
            "data_source": "discovery_bench_test",
            "prompt": [
                system_prompt,
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "env_class": "allocated_code_test",
            "reward_spec": {
                "method": "rule",
                "ground_truth": example['true_hypothesis']
            },
            "extra_info": {
                "question": example['query'],
                "original_answers": golden_answers,
                "dataset_path": example['dataset_path'],
                "metadata_type": example['metadata_type'],
                "source": example['source'],
                "metadata_id": example['metadata_id'],
                "query_id": example['query_id'],
                "split": "test",
                "index": idx
            }
        }
        return data
    
    dataset = []
    
    # Process test data
    for idx, example in enumerate(test_data):
        processed = process_example(example, idx)
        dataset.append(processed)
    
    return dataset


def save_test_dataset_with_gt(ground_truth_csv: str, output_dir=None):
    """Save the test dataset with ground truth."""
    if output_dir is None:
        # Save to the data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = create_test_dataset_with_gt(ground_truth_csv)
    
    if not dataset:
        print("No test data found!")
        return None
    
    # Filter out examples without ground truth for training evaluation
    dataset_with_gt = [d for d in dataset if d['reward_spec']['ground_truth']]
    dataset_without_gt = [d for d in dataset if not d['reward_spec']['ground_truth']]
    
    print(f"Examples with ground truth: {len(dataset_with_gt)}")
    print(f"Examples without ground truth: {len(dataset_without_gt)}")
    
    if not dataset_with_gt:
        print("No examples with ground truth found!")
        return None
    
    # Shuffle the data
    random.shuffle(dataset_with_gt)
    
    test_df = pd.DataFrame(dataset_with_gt)
    test_path = os.path.join(output_dir, "discovery_test_with_gt.parquet")
    
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved {len(dataset_with_gt)} test samples with ground truth to {test_path}")
    
    # Also save the full dataset (including those without ground truth) for reference
    if dataset_without_gt:
        full_df = pd.DataFrame(dataset)
        full_path = os.path.join(output_dir, "discovery_test_full.parquet")
        full_df.to_parquet(full_path, index=False)
        print(f"Saved {len(dataset)} full test samples to {full_path}")
    
    # Print a random example
    if dataset_with_gt:
        random_example = random.choice(dataset_with_gt)
        print("\n" + "="*80)
        print("RANDOM TEST EXAMPLE WITH GROUND TRUTH:")
        print("="*80)
        print(f"Query: {random_example['extra_info']['question']}")
        print(f"Ground Truth: {random_example['reward_spec']['ground_truth']}")
        print(f"Dataset: {random_example['extra_info']['source']}")
        print(f"Metadata ID: {random_example['extra_info']['metadata_id']}")
        print(f"Query ID: {random_example['extra_info']['query_id']}")
        print("="*80)
    
    return test_path


if __name__ == "__main__":
    ground_truth_csv = "/data/fan/discoverybench_answer_key.csv"
    test_path = save_test_dataset_with_gt(ground_truth_csv)
    if test_path:
        print(f"\nTest dataset with ground truth created successfully: {test_path}")
    else:
        print("Failed to create test dataset")
