import re


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