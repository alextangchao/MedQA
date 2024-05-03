import pandas
import re


def print_arr(arr):
    print("-" * 20)
    for ele in arr:
        print(ele)


if __name__ == "__main__":
    df = pandas.read_csv("data/raw_test_set.csv")
    print(list(df))
    # df = df.head()
    print(df)

    question_pattern = re.compile(
        r"\s*Question:\s*(.*)\s*URL:\s*(.*)\s*Answer:\s*(.*)\s*"
    )
    answer_pattern = re.compile(
        r"\s*Question:\s*(.*)\s*URL:\s*(.*)\s*Answer:\s*(.*)\s*", flags=re.DOTALL
    )

    questions = []
    url = []
    answers = []

    for values in df["Answer"]:
        question_match = question_pattern.match(values)
        answer_match = answer_pattern.match(values)
        if question_match and answer_match:
            questions.append(question_match.group(1))
            url.append(question_match.group(2))
            answer_text = answer_match.group(3)
            if answer_text[-2:] == " \n":
                answers.append(answer_text[:-2])
            else:
                print(answer_text)
        else:
            print("Error:")
            print(question_match)
            print(answer_match)
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
