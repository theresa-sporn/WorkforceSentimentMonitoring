import pandas as pd


def get_test_data():
    path = 'http://s3.amazonaws.com/assets.datacamp.com/production/course_935/datasets/500_amzn.csv'
    df = pd.read_csv(path)
    df = df.drop(columns=['pg_num', 'url'])
    df = df.rename(
        columns={
            "pros": "positives",
            "cons": "negatives",
        }
    )

    text_columns = ["positives", "negatives"]
    df["review"] = df[text_columns].astype("U").agg(" ".join, axis=1)
    return df

if __name__ == "__main__":
    df = get_test_data()
    print(df)
