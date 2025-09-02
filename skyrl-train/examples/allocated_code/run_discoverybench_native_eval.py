#!/usr/bin/env python3
"""
Run DiscoveryBench native evaluation using saved predictions from SkyRL test environment.
This script processes the saved prediction files and runs the original DiscoveryBench evaluation.
"""

import os
import sys
import json
import glob
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_predictions(predictions_dir: str) -> List[Dict[str, Any]]:
    """Load all prediction files from the directory."""
    predictions = []
    
    pattern = os.path.join(predictions_dir, "prediction_*.json")
    prediction_files = glob.glob(pattern)
    
    print(f"Found {len(prediction_files)} prediction files")
    
    for filepath in prediction_files:
        try:
            with open(filepath, 'r') as f:
                prediction = json.load(f)
                predictions.append(prediction)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    # Sort by example index for consistent ordering
    predictions.sort(key=lambda x: x.get('example_index', 0))
    
    return predictions


def find_metadata_file(dataset_name: str, metadata_id: int, base_dir: str) -> str:
    """Find the metadata file for a given dataset and metadata_id."""
    metadata_path = os.path.join(
        base_dir, 
        "skyrl-gym/skyrl_gym/envs/allocated_code/discovery_bench/data_folder/real/test",
        dataset_name,
        f"metadata_{metadata_id}.json"
    )
    
    if os.path.exists(metadata_path):
        return metadata_path
    
    # Try alternative path
    alt_path = os.path.join(
        base_dir,
        "SkyRL/skyrl-gym/skyrl_gym/envs/allocated_code/discovery_bench/data_folder/real/test", 
        dataset_name,
        f"metadata_{metadata_id}.json"
    )
    
    if os.path.exists(alt_path):
        return alt_path
    
    raise FileNotFoundError(f"Metadata file not found for {dataset_name}, metadata_{metadata_id}")


def run_discoverybench_evaluation(
    predictions: List[Dict[str, Any]], 
    discovery_eval_script: str,
    base_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """Run DiscoveryBench evaluation on all predictions."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    skyrl_scores = []
    discoverybench_scores = []
    
    for i, prediction in enumerate(predictions):
        try:
            print(f"\nEvaluating prediction {i+1}/{len(predictions)}")
            
            # Extract data
            dataset_name = prediction['dataset_name']
            metadata_id = prediction['metadata_id']
            query_id = prediction['query_id']
            query = prediction['query']
            predicted_hypothesis = prediction['predicted_hypothesis']
            ground_truth = prediction['ground_truth']
            skyrl_reward = prediction['skyrl_reward']
            
            print(f"Dataset: {dataset_name}, Metadata: {metadata_id}, Query: {query_id}")
            print(f"Query: {query[:100]}...")
            print(f"Predicted: {predicted_hypothesis[:100]}...")
            print(f"Ground Truth: {ground_truth[:100]}...")
            print(f"SkyRL Score: {skyrl_reward}")
            
            # Check if predicted_hypothesis is empty
            if not predicted_hypothesis or predicted_hypothesis.strip() == "":
                print("Predicted hypothesis is empty, assigning score 0")
                
                # Store results with score 0 for empty predictions
                result_data = {
                    'example_index': prediction['example_index'],
                    'dataset_name': dataset_name,
                    'metadata_id': metadata_id,
                    'query_id': query_id,
                    'query': query,
                    'predicted_hypothesis': predicted_hypothesis,
                    'ground_truth': ground_truth,
                    'skyrl_reward': skyrl_reward,
                    'discoverybench_score': 0.0,
                    'eval_result': {"final_score": 0.0, "note": "Empty prediction hypothesis"}
                }
                
                results.append(result_data)
                skyrl_scores.append(skyrl_reward)
                discoverybench_scores.append(0.0)
                print(f"DiscoveryBench Score: 0.0")
                continue
            
            # Find metadata file
            try:
                metadata_path = find_metadata_file(dataset_name, metadata_id, base_dir)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                tmp_output_path = tmp_file.name
            
            try:
                # Run discovery_eval.py
                cmd = [
                    'python', discovery_eval_script,
                    '--gold_hypo', ground_truth,
                    '--pred_hypo', predicted_hypothesis,
                    '--metadata_path', metadata_path,
                    '--metadata_type', 'real',
                    '--eval_output_path', tmp_output_path,
                    query
                ]
                
                result_proc = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
                
                if result_proc.returncode == 0:
                    # Parse the evaluation result
                    try:
                        eval_result = None
                        
                        # First try to read from the output file (more reliable)
                        if os.path.exists(tmp_output_path):
                            with open(tmp_output_path, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    eval_result = json.loads(content)
                        
                        # If file reading failed, try parsing stdout
                        if eval_result is None:
                            stdout_lines = result_proc.stdout.strip().split('\n')
                            
                            # Look for the last complete JSON object in stdout
                            # The discovery_eval.py prints the final JSON at the end
                            json_lines = []
                            in_json = False
                            brace_count = 0
                            
                            for line in stdout_lines:
                                line = line.strip()
                                if line.startswith('{'):
                                    in_json = True
                                    json_lines = [line]
                                    brace_count = line.count('{') - line.count('}')
                                elif in_json:
                                    json_lines.append(line)
                                    brace_count += line.count('{') - line.count('}')
                                    if brace_count == 0:
                                        # Complete JSON object found
                                        json_str = '\n'.join(json_lines)
                                        try:
                                            eval_result = json.loads(json_str)
                                            break  # Use the last complete JSON
                                        except json.JSONDecodeError:
                                            continue
                        
                        if eval_result is None:
                            eval_result = {"error": "No valid JSON found"}
                        
                        # Extract DiscoveryBench score
                        db_score = eval_result.get('final_score', 0.0)
                        
                        print(f"DiscoveryBench Score: {db_score}")
                        
                        # Store results
                        result_data = {
                            'example_index': prediction['example_index'],
                            'dataset_name': dataset_name,
                            'metadata_id': metadata_id,
                            'query_id': query_id,
                            'query': query,
                            'predicted_hypothesis': predicted_hypothesis,
                            'ground_truth': ground_truth,
                            'skyrl_reward': skyrl_reward,
                            'discoverybench_score': db_score,
                            'eval_result': eval_result
                        }
                        
                        results.append(result_data)
                        skyrl_scores.append(skyrl_reward)
                        discoverybench_scores.append(db_score)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing evaluation result: {e}")
                        print(f"Raw output: {result_proc.stdout}")
                        # Continue with error result
                        result_data = {
                            'example_index': prediction['example_index'],
                            'dataset_name': dataset_name,
                            'metadata_id': metadata_id,
                            'query_id': query_id,
                            'query': query,
                            'predicted_hypothesis': predicted_hypothesis,
                            'ground_truth': ground_truth,
                            'skyrl_reward': skyrl_reward,
                            'discoverybench_score': 0.0,
                            'eval_result': {"error": f"JSON parsing failed: {str(e)}"}
                        }
                        results.append(result_data)
                        skyrl_scores.append(skyrl_reward)
                        discoverybench_scores.append(0.0)
                        
                else:
                    print(f"Error running evaluation: {result_proc.stderr}")
                    
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_output_path):
                    os.unlink(tmp_output_path)
                    
        except Exception as e:
            print(f"Error processing prediction {i}: {e}")
            continue
    
    # Calculate correlation metrics
    correlation_results = {}
    if len(skyrl_scores) > 1 and len(discoverybench_scores) > 1:
        try:
            pearson_corr, pearson_p = pearsonr(skyrl_scores, discoverybench_scores)
            spearman_corr, spearman_p = spearmanr(skyrl_scores, discoverybench_scores)
            
            correlation_results = {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'num_samples': len(skyrl_scores)
            }
            
            print(f"\n=== Correlation Analysis ===")
            print(f"Number of samples: {len(skyrl_scores)}")
            print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
            print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
            
        except Exception as e:
            print(f"Error calculating correlations: {e}")
    
    # Calculate summary statistics
    summary_stats = {
        'skyrl_mean': np.mean(skyrl_scores) if skyrl_scores else 0,
        'skyrl_std': np.std(skyrl_scores) if skyrl_scores else 0,
        'discoverybench_mean': np.mean(discoverybench_scores) if discoverybench_scores else 0,
        'discoverybench_std': np.std(discoverybench_scores) if discoverybench_scores else 0,
        'total_evaluated': len(results),
        'total_predictions': len(predictions)
    }
    
    # Save detailed results
    final_results = {
        'summary_stats': summary_stats,
        'correlation_results': correlation_results,
        'detailed_results': results
    }
    
    results_path = os.path.join(output_dir, 'discoverybench_native_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save CSV for easy analysis
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'evaluation_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to:")
        print(f"- JSON: {results_path}")
        print(f"- CSV: {csv_path}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='Run DiscoveryBench native evaluation')
    parser.add_argument('--predictions_dir', required=True, 
                       help='Directory containing prediction JSON files')
    parser.add_argument('--discovery_eval_script', required=True, 
                       help='Path to discovery_eval.py')
    parser.add_argument('--base_dir', default='/data/fan', 
                       help='Base directory for finding metadata files')
    parser.add_argument('--output_dir', required=True, 
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Check for required API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable must be set")
        print("DiscoveryBench evaluation uses GPT-4o for hypothesis comparison")
        return None
    
    # Load predictions
    print(f"Loading predictions from: {args.predictions_dir}")
    predictions = load_predictions(args.predictions_dir)
    
    if not predictions:
        print("No predictions found!")
        return
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Run evaluation
    results = run_discoverybench_evaluation(
        predictions, 
        args.discovery_eval_script, 
        args.base_dir,
        args.output_dir
    )
    
    return results


if __name__ == "__main__":
    main()
