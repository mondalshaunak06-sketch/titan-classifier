import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
print("Shape:", df.shape)
print(df.dtypes)
print(df.isna().sum())

df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["embarked"], drop_first=True)
df["family_size"] = df["sibsp"] + df["parch"] + 1