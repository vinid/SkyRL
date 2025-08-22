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


def is_code_execution_successful(tool_output: str) -> bool:
    """
    Check if code execution was successful (no errors)
    Args:
        tool_output: The output result from code execution
    Returns:
        bool: True if execution was successful, False if there were errors
    """
    if not tool_output:
        return False
    
    # Check for common error indicators
    error_indicators = [
        "Error", "Exception", "Traceback", "SyntaxError", 
        "NameError", "TypeError", "ValueError", "IndexError",
        "KeyError", "AttributeError", "ImportError", "ModuleNotFoundError"
    ]
    
    tool_output_lower = tool_output.lower()
    for indicator in error_indicators:
        if indicator.lower() in tool_output_lower:
            return False
    
    return True


def compute_llm_score(solution_str, history_str, ground_truth, query, method="strict", format_score=0.0, score=1.0):
    '''
Entire Trajectory:
{history_str}
    '''
    if solution_str is None:
        return 0.0
    solution_str =  str(solution_str)
    ground_truth = str(ground_truth)
    history_str = str(history_str)

    prompt = f"""You are a judge evaluating scientific hypotheses. You need to score how well the predicted hypothesis matches the ground truth hypothesis.
Both the hypotheses answer the natural language query "Query" over the dataset(s).
To evaluate the hypothesis, you need to consider three dimensions that define the hypothesis: Context, Variables, and Relations. 
Here are the definitions for these dimensions:
- Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., “for men over \
the age of 30”, “in Asia and Europe”. 
- Variables: Known concepts that interact in a meaningful way under a given context to \
produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
- Relations: Interactions between a given set of variables under a given context to produce \
the hypothesis. E.g., “quadratic relationship”, “inversely proportional”, piecewise conditionals, \
or "None" if there is no interacting relationship.
Compare the predicted hypothesis with the ground truth hypothesis in terms of these three dimensions.

Query:
{query}

Predicted Hypothesis:
{solution_str}

Ground Truth Hypothesis:
{ground_truth}

Evaluate the hypothesis and provide a score between 0 and 1, where:
- 1.0 means the hypotheses make the same scientific claim
- 0.0 means completely different or contradictory claims
- Values between 0 and 1 indicate partial alignment in variables, relationships, or context
- If the predicted hypothesis is None, return 0.0

Return your response in the following format:
<reasoning>REASONING</reasoning>
<answer>SCORE</answer>

Only return the numeric score between 0 and 1 within the answer tags."""

    try:
        response = completion(
            model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            # model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        score_str = extract_answer(response.choices[0].message.content)
        if score_str:
            score = float(score_str)
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0

def compute_llm_score_discrete(solution_str, history_str, ground_truth, query, method="strict", format_score=0.0, score=1.0):
    if solution_str is None:
        return 0.0
    solution_str =  str(solution_str)
    ground_truth = str(ground_truth)
    history_str = str(history_str)

    prompt = f"""You are a judge evaluating scientific hypotheses. You need to score how well the predicted hypothesis matches the ground truth hypothesis.
Both the hypotheses answer the natural language query "Query" over the dataset(s).
To evaluate the hypothesis, you need to consider three dimensions that define the hypothesis: Context, Variables, and Relations. 
Here are the definitions for these dimensions:
- Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., “for men over \
the age of 30”, “in Asia and Europe”. 
- Variables: Known concepts that interact in a meaningful way under a given context to \
produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
- Relations: Interactions between a given set of variables under a given context to produce \
the hypothesis. E.g., “quadratic relationship”, “inversely proportional”, piecewise conditionals, \
or "None" if there is no interacting relationship.

Compare the predicted hypothesis with the ground truth hypothesis in terms of these three dimensions.

Query:
{query}

Predicted Hypothesis:
{solution_str}

Ground Truth Hypothesis:
{ground_truth}

Evaluate the hypothesis and provide your score from the following options, where:
- 0: Two hypotheses are completely different or contradict each other; Major context mismatch, or variables/relations do not align at all.
- 0.2: The two hypotheses are weakly aligned. Context unclear OR minor context mismatch; some overlap/compatibility in variables/relations.
- 0.5: The two hypotheses are partially aligned. Context equivalent; variables/relations are partially aligned; some variables/relations are incompatible but not contradictory.
- 0.8: There are no contradictions and the two hypotheses are highly aligned. Context equivalent and variables and relation mostly match with only minor omissions or weaker phrasing.
- 1.0: Two hypotheses make the same scientific claim. Context equivalent AND variables match/superset AND relation matches.

Return your response in the following format:
<reasoning>REASONING</reasoning>
<answer>SCORE</answer>

Only return the numeric score chosen from the options (0, 0.2, 0.5, 0.8, 1.0) within the answer tags."""

    try:
        response = completion(
            model="together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            # model="o4-mini",
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