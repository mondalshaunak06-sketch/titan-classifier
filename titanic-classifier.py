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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

y = df["survived"]
basic_features = ["pclass", "sex", "age", "fare"]

train_df, test_df, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(train_df[basic_features], y_train)
preds = model.predict(test_df[basic_features])
print("Baseline accuracy:", round(accuracy_score(y_test, preds), 3))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

extended_features = ["pclass", "sex", "age", "fare", "family_size"] + [
    c for c in df.columns if c.startswith("embarked_")
]

feature_sets = {"basic": basic_features, "extended": extended_features}
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
for fs_name, cols in feature_sets.items():
    for model_name, m in models.items():
        m.fit(train_df[cols], y_train)
        p = m.predict(test_df[cols])
        acc = accuracy_score(y_test, p)
        results.append((fs_name, model_name, round(acc, 3)))
        print(f"\n{model_name} | {fs_name} features")
        print("Accuracy:", round(acc, 3))
        print("Confusion matrix:\n", confusion_matrix(y_test, p))

print("\nSummary")
print(pd.DataFrame(results, columns=["features", "model", "accuracy"]))