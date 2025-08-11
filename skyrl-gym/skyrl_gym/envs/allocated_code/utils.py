import re
from litellm import completion
import json


def extract_answer(solution_str):
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    answer = extract_answer(solution_str)
    if answer is None:
        return 0
    else:
        if answer == str(ground_truth).strip():
            return score
        else:
            return format_score


def compute_llm_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    solution_str =  str(solution_str)
    ground_truth = str(ground_truth)

    prompt = f"""You are a judge evaluating scientific hypotheses. You need to score how well the predicted hypothesis matches the ground truth hypothesis.

Predicted Hypothesis:
{solution_str}

Ground Truth Hypothesis:
{ground_truth}

Evaluate the hypothesis and provide a score between 0 and 1, where:
- 1.0 means the hypotheses make the same scientific claim
- 0.0 means completely different or contradictory claims
- Values between 0 and 1 indicate partial alignment in variables, relationships, or context

Return your response in the following format:
<answer>SCORE</answer>

Only return the numeric score between 0 and 1 within the answer tags."""

    try:
        response = completion(
            model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
        )
        score_str = extract_answer(response.choices[0].message.content)
        if score_str:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    solution = """def add(a, b):
    return a + b"""
    ground_truth = """def add(a, b):
    return a + b"""
    
    score = compute_llm_score(solution, ground_truth)
    print(f"Score: {score}") 