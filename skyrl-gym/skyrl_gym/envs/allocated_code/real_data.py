import pandas as pd
import os
import datasets


def make_prefix(dp, template_type='base'):
    question = dp['question']

    if template_type == 'base':
        prefix = f"""
            You are answering a question that could require multiple steps to solve, combining search, coding and reasoning.
            
            IMPORTANT: You MUST follow this EXACT format for EVERY response:
            1. Start with <reasoning> to show your reasoning process </reasoning>
            2. Use <python> for searches or calculations </python>. You can use the search functions to search the web.
            3. When code returns results, they will appear between <information> and </information> tags
            4. Use as many <reasoning> / <python> blocks as needed to solve the question
            5. ALWAYS end with <answer> your final answer </answer>
            6. Feel free to test code implementation before returning the final answer
            
            RULES:
            - NEVER skip the <reasoning> step or provide a final answer without <answer> tags
            - ALL code must be inside <python> tags
            - Code sessions are not isolated - you can use variables from one code block in another
            - Break down complex questions into smaller steps in the <reasoning> section
            - Answers are typically very short (1-3 words)
            - If you fail at something, try again but think first
            - Look at how we interleave the <reasoning> and <python> tags to see how to format your response
            
            You can:
            1. Search for information:
            <python>
            result = search('YOUR_SEARCH_QUERY')
            print(result)
            </python>
            
            2. Perform calculations using pandas, numpy, or math (no file I/O):
            <python>
            # Your calculation code here
            print(result)  # ALWAYS print the result
            </python>

            # START EXAMPLES

            Example 1 (Search):
            Question: ``Where do heart's electrical impulses originate?```
            <reasoning>
            Ok this answer can be found by searching for the origin of heart's electrical impulses.
            </reasoning>
            <python>
            result = search('origin of heart's electrical impulses')
            print(result)
            </python>
            <information>
            [Search results appear here]
            </information>
            <reasoning>
            No other information is needed, the final answer is in the search results.
            </reasoning>
            <answer>
            [Concise answer here]
            </answer>

            Example 2 (Search + Math): 
            
            Question: ```
            What is the square root of VAR_XTWI?

            Where: VAR_XTWI is obtained by getting the answer to this question/implementing the following coding problem: "who did the seahawks play in super bowl 2014?"

            Here is the guideline to help you format the answer: the answer consists of EXACTLY 2 words representing a proper team name Once you have found the answer to the problem, use the following formula to get the final value of VAR_XTWI: len(str(answer)) - 10. 

            The len has to be computed on the entire answer, that's why we cast it to a string.```

            <reasoning> to solve this i first need to find the value of VAR_XTWI, which is the length of the answer to the question "who did the seahawks play in super bowl 2014?" </reasoning>
            <python> search("who did the seahawks play in super bowl 2014?") </python>
            <information> 

            [1] the seahawks are playing against the denver broncos in super bowl 2014 

            </information>
            <reasoning> ok so since i know that the answer is composed of words, the current answer is "denver broncos", let me compute the length of the answer and subtract 10 to get the final value of VAR_XTWI. </reasoning>
            <python> 
            VAR_XTWI = len("denver broncos") - 10 

            print(VAR_XTWI)
            </python>
            <information>
            4
            </information>
            <reasoning> now there is additional information i need to find. in this case the square root of VAR_XTWI </reasoning>
            <python>
            k = sqrt(VAR_XTWI)
            print(k)
            </python>
            <information>
            2
            </information>
            <reasoning> ok I have all the information I need, let me return the answer </reasoning>
            <answer>
            2
            </answer>

            # END EXAMPLES

            Now answer the following question using the EXACT format shown above:
            Question: {question}
            """
    else:
        raise NotImplementedError
    return prefix


def create_real_dataset():
    print("Loading datasets from HuggingFace...")
    
    flashrag_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
    math_dataset = datasets.load_dataset('federicotogether/math-search-o1-v1', download_mode='force_redownload')
    frames_dataset = datasets.load_dataset('federicotogether/frames-train')

    sampled_flashrag_train = flashrag_dataset['train'].shuffle(seed=42).select(range(200))
    sampled_flashrag_test = flashrag_dataset['test'].shuffle(seed=42).select(range(50))
    
    # Sample 350 examples from frames dataset (300 for train, 50 for test)
    sampled_frames_train = frames_dataset['test'].shuffle(seed=42).select(range(300))
    sampled_frames_test = frames_dataset['test'].shuffle(seed=42).select(range(300, 350))
    
    normalized_frames_train = sampled_frames_train.map(lambda x: {
        'question': x['Prompt'], 
        'golden_answers': [x['Answer']]
    })
    
    normalized_frames_test = sampled_frames_test.map(lambda x: {
        'question': x['Prompt'], 
        'golden_answers': [x['Answer']]
    })
    
    train_dataset = datasets.concatenate_datasets([
        sampled_flashrag_train,
        math_dataset['train'],
        normalized_frames_train
    ]).shuffle(seed=42)

    test_dataset = datasets.concatenate_datasets([
        sampled_flashrag_test,
        math_dataset['test'],
        normalized_frames_test
    ]).shuffle(seed=42)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    def process_example(example, idx, split):
        question = example['question'].strip()
        if question[-1] != '?':
            question += '?'
        
        example_with_question = {'question': question}
        user_content = make_prefix(example_with_question, template_type='base')
        
        system_prompt = {
            "role": "system", 
            "content": "You are a helpful assistant that can search for information and perform calculations to answer questions. /nothink"
        }
        
        ground_truth = example['golden_answers']
        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            ground_truth = ground_truth[0]
        
        data = {
            "data_source": "real_nq_math",
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
                "ground_truth": str(ground_truth)
            },
            "extra_info": {
                "question": question,
                "original_answers": example['golden_answers'],
                "split": split,
                "index": idx
            }
        }
        return data

    dataset = []
    
    for idx, example in enumerate(train_dataset):
        processed = process_example(example, idx, 'train')
        dataset.append(processed)
    
    for idx, example in enumerate(test_dataset):
        processed = process_example(example, idx, 'test')
        dataset.append(processed)
    
    return dataset


def save_real_dataset(output_dir="./data/allocated_code"):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = create_real_dataset()
    
    train_data = [d for d in dataset if d['extra_info']['split'] == 'train']
    val_data = [d for d in dataset if d['extra_info']['split'] == 'test']
    
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
    save_real_dataset() 