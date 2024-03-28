from datasets import load_dataset
import numpy as np 
from matplotlib import pyplot as plt 
from scipy import stats
import pandas as pd 
from tqdm import tqdm 

from openai import OpenAI

# Setup: 
eval_client = OpenAI() # prompt evaluation 
prompt_client = OpenAI() # prompt generator 

dataset_all = load_dataset("ehovy/race", "all")

# Utils: 
def get_response_msg(response):
    response_content = response.choices[0].message.content
    return response_content

def _get_query(prompt, example): 
    '''
    Generate 
    '''
    query = f'''
{prompt}

article={example['article']} 

question={example['question']}

options={example['options']}
'''
    return query 

# main functions: 
def eval(prompt, test_size=10):
    '''
    Returns: 
        Per-sample results: 
            prompt
            article
            question 
            options 
            correct_answer 
            
            model_answer: str
            correctness: True/False
            CE
        
        Aggregated results:
            avg_CE: average cross entropy for the tested data 
            accuracy
            accuracy_CI: 95% confidence interval for accuracy
    ''' 

    results = []

    for example in tqdm(dataset_all['train'].select(range(test_size))):
        query = _get_query(prompt, example)

        response = eval_client.chat.completions.create(
            model="gpt-4",
            messages=[
                    {"role": "user", "content": query}
                ],
                logprobs=True, 
                top_logprobs=10, 
                seed=1234,
        )

        corresp = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

        # CROSS ENTROPY
        logprobs = np.ones(4) * np.log(10e-5) # init probabilities to 0 
        # classes = [A, B, C, D, OTHERS]. Don't care about OTHERS since it's never a correct answer
        # probs doesn't sum up to 1
        for t in response.choices[0].logprobs.content[0].top_logprobs: 
            if t.token in ('A', 'B', 'C', 'D'): 
                logprobs[corresp[t.token]] = t.logprob 
        print(logprobs)
        ce = -logprobs[corresp[example['answer']]]

        model_answer = response.choices[0].message.content
        
        results.append({
            "prompt": prompt, 
            "article": example['article'],
            "question": example['question'],
            "options": example['options'],
            "correct_answer": example['answer'],

            "model_answer": model_answer,
            "correctness": example['answer'] == model_answer, 
            "ce": ce,
        })

    results_df = pd.DataFrame(results)
    avg_ce = results_df.ce.mean()
    accuracy = results_df.correctness.mean() 

    THRESHOLD = 0.1
    t_stat = stats.ttest_1samp(results_df.correctness.to_numpy() * 1, THRESHOLD)
    ci = t_stat.confidence_interval(confidence_level=0.95)

    return {
        "results_df": results_df, 
        "avg_ce": avg_ce, 
        "accuracy": accuracy, 
        "ci": ci,
    } 


def select_eval_examples(results_df, n=10): 
    indices = np.random.choice(np.arange(len(results_df)), size=n, replace=False)
    s = []
    for i in indices: 
        s.append(
            f"""
Prompt: {results_df.loc[i, 'prompt']}
Article: {results_df.loc[i, 'article']}
Question: {results_df.loc[i, 'question']}
Options: {results_df.loc[i, 'options']}
Correct Answer: {results_df.loc[i, 'correct_answer']}
Model Answer: {results_df.loc[i, 'model_answer']}
Entropy: {results_df.loc[i, 'ce']}
"""
)
    return s 

prompt = "Answer the question based on the article. Your only choices of answers are A, B, C, D"
eval_result = eval(prompt, test_size=10)
breakpoint()

# meta_prompt = f"""
# You are a grad student working for an NLP lab. Your objective is to generate a prompt that minimize entropy. 
# Do not include article, questions, or answer into your prompt. 

# The model is given your prompt, an article, a question, and a list of options. The model returns the correct answer based on the prompt. 
# {" ".join(s)}
# """
# print(meta_prompt)






