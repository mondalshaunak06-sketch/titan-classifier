# titan-classifier
A machine learning project that predicts whether a passenger survived the Titanic

## Dataset
The Titanic dataset (loaded with seaborn). It has 891 passengers with details like class, sex, age, fare and family members on board. The target is `survived` (0 = no, 1 = yes).

## Approach
1. **Load and inspect** the data with Pandas (shape, dtypes, missing values).
2. **Clean and prepare:** filled missing ages with the median, filled missing embarkation ports with the most common value, encoded `sex` as 0/1 and one-hot encoded `embarked`. Added a `family_size` feature.
3. **Split** the data 80/20 into train and test sets.
4. **Train and compare** two models (Logistic Regression and Random Forest) on two feature sets:
   - Basic: pclass, sex, age, fare
   - Extended: basic + family_size + embarked

## Results

| Features | Model | Accuracy |
|----------|-------|----------|
| basic | LogisticRegression | ___ |
| basic | RandomForest | ___ |
| extended | LogisticRegression | ___ |
| extended | RandomForest | ___ |

**Best result:** ___ with ___ features, accuracy ___.

**What I learned:** -basic concepts of datasets, directories, cleaning data which gives the overview crux of ml using sckit and python 

## How to run
```bash
pip install pandas scikit-learn seaborn
python titanic-classifier.py
```