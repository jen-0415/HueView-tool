import pandas as pd
print(pd.read_csv("data/processed/landmark_failures.csv")["reason"].value_counts())