import vertexai
from vertexai.language_models import TextGenerationModel
from dotenv import load_dotenv
import pandas

load_dotenv()

vertexai.init(project="cs685-421521", location="us-central1")


def run_palm(question):
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "temperature": 0.9,
        "top_p": 1,
    }
    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(f"{question}\n\nAnswer in one paragraph:", **parameters)
    # print(response)
    print(f"Response from Model:\n{response.text}\n\n")
    return response.text


def load_data(n):
    print("Loading data...")

    df = pandas.read_csv("data/test_set.csv")
    print(list(df))
    if n != -1:
        df = df.head(n)
    questions = df.get("question").tolist()
    answers = df.get("answer").tolist()

    return questions, answers


def save_data(questions, results, answers, id):
    print("Saving data...")

    df = pandas.DataFrame(
        {"question": questions, "predict": results, "answer": answers}
    )
    df.to_csv(f"output/med-palm-{id}.csv", encoding="utf-8", index=False)


if __name__ == "__main__":
    num = -1
    question_list, answer_list = load_data(num)

    result_list = []
    for i, question in enumerate(question_list):
        print(f"Question {i}:")
        result = run_palm(question)
        result_list.append(result)

        if i % 100 == 0:
            save_data(question_list[: i + 1], result_list, answer_list[: i + 1], i)

    save_data(question_list, result_list, answer_list, num)
