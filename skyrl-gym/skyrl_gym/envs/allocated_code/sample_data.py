import pandas as pd
import os


def create_sample_coding_problems():
    problems = [
        {
            "question": "Calculate the sum of numbers from 1 to 10. The answer should be a number.",
            "ground_truth": "55",
            "solution_code": "result = sum(range(1, 11))\nprint(result)"
        },
        {
            "question": "Find the factorial of 5. The answer should be a number.",
            "ground_truth": "120", 
            "solution_code": "import math\nresult = math.factorial(5)\nprint(result)"
        },
        {
            "question": "Calculate 7 to the power of 3. The answer should be a number.",
            "ground_truth": "343",
            "solution_code": "result = 7 ** 3\nprint(result)"
        },
        {
            "question": "Generate the first 5 fibonacci numbers. The answer should be a list.",
            "ground_truth": "[0, 1, 1, 2, 3]",
            "solution_code": "fib = [0, 1]\nfor i in range(3):\n    fib.append(fib[-1] + fib[-2])\nprint(fib)"
        },
        {
            "question": "Count how many even numbers are between 1 and 20 (inclusive). The answer should be a number.",
            "ground_truth": "10",
            "solution_code": "count = sum(1 for i in range(1, 21) if i % 2 == 0)\nprint(count)"
        },
        {
            "question": "Find the maximum number in the list [3, 7, 2, 9, 1, 5]. The answer should be a number.",
            "ground_truth": "9",
            "solution_code": "numbers = [3, 7, 2, 9, 1, 5]\nresult = max(numbers)\nprint(result)"
        },
        {
            "question": "Check if the number 17 is prime. The answer should be True or False.",
            "ground_truth": "True",
            "solution_code": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nresult = is_prime(17)\nprint(result)"
        },
        {
            "question": "Reverse the string 'hello'. The answer should be a string.",
            "ground_truth": "olleh",
            "solution_code": "text = 'hello'\nresult = text[::-1]\nprint(result)"
        },
        {
            "question": "Find all divisors of 12. The answer should be a list of numbers.",
            "ground_truth": "[1, 2, 3, 4, 6, 12]",
            "solution_code": "n = 12\ndivisors = []\nfor i in range(1, n + 1):\n    if n % i == 0:\n        divisors.append(i)\nprint(divisors)"
        },
        {
            "question": "Calculate the sum of squares of numbers from 1 to 4 (1² + 2² + 3² + 4²). The answer should be a number.",
            "ground_truth": "30",
            "solution_code": "result = sum(i**2 for i in range(1, 5))\nprint(result)"
        }
    ]
    
    system_prompt = {
        "role": "system", 
        "content": "You are a helpful coding assistant. You must conduct reasoning inside <think> and </think> first. Then write Python code inside <python> and </python> tags to solve problems. After executing code, provide your final answer inside <answer> and </answer> tags, without detailed explanations. For example, User: What is 2+3? <think>I need to calculate 2 + 3</think> <python>result = 2 + 3\nprint(result)</python> <information> 5 </information> <think> after the calcualtion I have identified that the answer is 5 </think> <answer>5</answer>."
    }
    
    dataset = []
    for i, problem in enumerate(problems):
        user_content = f"Solve this problem step by step: {problem['question']}"
        
        data = {
            "data_source": "sample_coding",
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
                "ground_truth": problem["ground_truth"]
            },
            "extra_info": {
                "question": problem["question"],
                "solution_code": problem["solution_code"],
                "index": i
            }
        }
        dataset.append(data)
    
    dataset = dataset * 5
    
    return dataset


def save_sample_dataset(output_dir="./data/allocated_code"):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = create_sample_coding_problems()
    
    # Split into train/validation
    train_data = dataset[:40]  
    val_data = dataset[40:]    
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    
    return train_path, val_path


if __name__ == "__main__":
    save_sample_dataset() 