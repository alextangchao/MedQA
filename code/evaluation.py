from evaluate import load
import pandas as pd 
import os
from together import Together
import ast
from openai import OpenAI

# load results from medpalm
medpalm = pd.read_csv("med-palm-paragraph.csv")
questions = list(medpalm['question'])
predictions = list(medpalm['predict'])
references = list(medpalm['answer'])


# load results from llama3
llama3 = pd.read_csv("llama3.csv")
# questions = list(llama3['question'])
# predictions = list(llama3['qlora-1844'])
# references = list(llama3['answer'])



# calculate BERTscore

bertscore = load("bertscore")

bertscore = bertscore.compute(predictions=predictions, references=references, lang="en") # precision, recall, f1score


## prompt for llm evaluation

prompt = \
"""
This is an evaluation task for a medical Question-Answering system, your task is to evaluate the predicted answer of medical question and compare it with the gold answer provided as a reference. 

The evaluation of the predicted answer would be based on four metrics: coherence, naturalness, correctness and completeness, you are asked to provide four ratings based on these metrics separately.

Here are some instructions on rating the four metrics:
1. Correctness: correctness addresses the faithfulness of an explanation with respect to the predictive model, this reflects whether the model understands the given question. You need to check that whether the prediction contains correct information about the question and if it is related to the question;
2. Coherence: coherence describes how accordant the explanation is with prior knowledge and beliefs, you need to check the logical flow, thematic consistency, and stylistic uniformity in the predicted answer. This step is to evaluate the consistency of the output answer;
3. Naturalness: naturalness evaluates how closely the predicted answer resembles a native human speaker. A natural answer should be fluent and indistinguishable from human written text, you need to check the similarity between the prediction and a human written text;
4. Completeness: completeness measures how complete the prediction is. A complete answer should be comprehensive and thorough, covers all relevant aspects in the given question. You should check whether the prediction answers all aspects mentioned in the question compared to the gold answer and the question itself;

Based on the instructions, you are required to output the rating based on a 10-point scale. 1 (incompatible to the instructions and incorrect), 5 (relates to the instructions but have some flaws) and 10 (perfectly aligns with the question and gold answer as well as the rating instructions ).

The gold Q-A pairs and model prediction are:

question: {%s}
gold answer: {%s}
prediction:{%s}

The output should be provided in JSON format as follows with no extra words and only the JSON dictionary:
{"question": the input question, "prediction": given predicted answer, "gold answer": given gold answer, "correctness": the score for correctness, "coherence": score for coherence, "naturalness": score for naturalness, "completeness": score for completeness, "reason": reason for giving the ratings for the four scores}
    

"""


#### llama3 70b

client = Together(api_key = "together ai api key")

#### GPT-4

client = OpenAI(
    api_key = "api key",
    base_url = "api base"
)

#### get evaluation

Report = pd.DataFrame()

for i in range(len(questions)):
    content = prompt% tuple([questions[i], references[i], predictions[i]])
    
    response = client.chat.completions.create(
    model="model", # llama3 70b: "META-LLAMA/LLAMA-3-70B-CHAT-HF"; GPT4: "gpt-4"
    messages=[{"role": "user", "content": content}],
    temperature = 0.2,
)
    report = response.choices[0].message.content
    report = ast.literal_eval(report)
    Report = pd.concat([Report, pd.DataFrame([report])], ignore_index=True)

    if (i+1)%10 == 0:
        print("-----------------------eval complete for ", i+1, "predictions---------------------")