import os
from datasets import load_dataset
import numpy as np 
from matplotlib import pyplot as plt 
from scipy import stats
import pandas as pd 
from tqdm import tqdm 
import math 

from openai import OpenAI

# Setup: 
eval_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # prompt evaluation 
prompt_gen_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # prompt generator 

dataset_all = load_dataset("ehovy/race", "all")

# Utils: 
def get_response_msg(response):
    response_content = response.choices[0].message.content
    return response_content

def _get_query(prompt, example): 
    '''
    Generate query for eval_client
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
        # print(logprobs)
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
        "prompt": prompt, 
        "results_df": results_df, 
        "avg_ce": avg_ce, 
        "accuracy": accuracy, 
        "ci": ci,
    } 


def select_eval_examples(results_df, n=10, strategy='random'):
    '''
    select eval examples to feed into promp_gen_client that optimizes for a better prompt
    
    Inputs: 
        results_df: eval result of past prompts as df 
        n: number of examples 
        strategy: 
            highly recommended that you get a diverse set of prompts and outcomes 
            - random: randomly select n prompts 
            - best_worse: select the bottom n/2 and top n/2 examples ranked by entropy 
    ''' 
    assert n <= len(results_df)

    if strategy == 'random': 
        indices = np.random.choice(np.arange(len(results_df)), size=n, replace=False)
    elif strategy == 'best_worst': 
        sorted_idx = results_df.ce.sort_values(ascending=False).index
        indices = list(sorted_idx[:math.ceil(n/2)]) + list(sorted_idx[-math.floor(n/2):])
    else:
        raise NotImplementedError(f"strategy {strategy} is not implemented")
    s = []
    for i in indices: 
        s.append(
            f"""
Prompt: {results_df.loc[i, 'prompt']}
Correct Answer: {results_df.loc[i, 'correct_answer']}
Model Answer: {results_df.loc[i, 'model_answer']}
Entropy: {results_df.loc[i, 'ce']}
"""
)
    # Article: {results_df.loc[i, 'article']}
    # Question: {results_df.loc[i, 'question']}
    # Options: {results_df.loc[i, 'options']}
    return s 

def get_meta_prompt(eval_result, n=10, strategy='best_worst'): 
    if isinstance(eval_result, dict): 
        results_df = eval_result['results_df']
    elif isinstance(eval_result, list): 
        results_df = pd.concat([e['results_df'] for e in eval_result], ignore_index=True)
    else:
        raise TypeError(f"eval_result only accept type dict or list, got {type(eval_result)}")
    
    s = select_eval_examples(results_df, n=n, strategy='best_worst')
    meta_prompt = f"""
You are responsible for prompting a large language model on a question-answering task. 
Your objective is to generate a MODEL_PROMPT that encourages the model to produce the correct answer and minimizes entropy. 
Do not include article, questions, or answer into your prompt. 

Here are some sample prompts and the model's responses to them, with the format: 
Prompt: <MODEL_PROMPT>
Correct Answer: <MODEL'S TARGET OUTPUT>
Model Answer: <MODEL'S ACTUAL OUTPUT>
Entropy: <SOMETHING YOU WANT TO MINIMIZE>
{" ".join(s)}
"""
    return meta_prompt
