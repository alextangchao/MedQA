import pandas
import re


def save_data(questions, results, answers):
    print("Saving data...")

    df = pandas.DataFrame(
        {"question": questions, "predict": results, "answer": answers}
    )
    df.to_csv(f"output/med-palm-{num}.csv", encoding="utf-8", index=False)


def print_arr(arr):
    print("-" * 20)
    for ele in arr:
        print(ele)


if __name__ == "__main__":
    df = pandas.read_csv("data/test_set.csv")
    print(list(df))
    # df = df.head()
    print(df)
    # questions = df.get("question").tolist()
    # answers = df.get("answer").tolist()

    re_pattern = re.compile(r"\s*Question:\s*(.*)\s*URL:\s*(.*)\s*Answer:\s*(.*)\s*",flags=re.DOTALL)

    questions = []
    url = []
    answers = []

    for values in df["Answer"]:
        m = re_pattern.fullmatch(values)
        if m:
            questions.append(m.group(1))
            url.append(m.group(2))
            answers.append(m.group(3))
        else:
            print("Error:")
            print(m)
            print(values)

    # print_arr(questions)
    # print_arr(url)
    # print_arr(answers)

    result_df = pandas.DataFrame(
        {
            "question": questions,
            "answer": answers,
            "AnswerID": df["AnswerID"].tolist(),
            "url": url,
        }
    )
    print(result_df)
    result_df.to_csv(f"data/new_test_set.csv", encoding="utf-8", index=False)
