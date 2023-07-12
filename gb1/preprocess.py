import pandas as pd

df = pd.read_csv("raw_data/sampled.csv")

df = df.rename(columns={"target": "label"})

train_index = []
validate_index = []
test_index = []

for i in range(len(df)):
    row = df.iloc[i]
    if row.set == "test":
        test_index.append(i)
    elif row.validation == True:
        validate_index.append(i)
    else:
        train_index.append(i)

train_df = df.loc[train_index, ["sequence", "label"]]
validate_df = df.loc[validate_index, ["sequence", "label"]]
test_df = df.loc[test_index, ["sequence", "label"]]
assert len(train_df) + len(validate_df) + len(test_df) == len(df)

train_df.to_csv("splits/train.csv", index=False)
validate_df.to_csv("splits/validate.csv", index=False)
test_df.to_csv("splits/test.csv", index=False)