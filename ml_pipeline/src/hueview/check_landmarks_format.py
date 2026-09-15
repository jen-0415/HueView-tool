import pandas as pd
df = pd.read_csv("data/processed/landmarks_index.csv")
print(df.columns.tolist())
print(df.head(3).to_string())