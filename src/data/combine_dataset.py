import pandas as pd
import glob

files = glob.glob("dataset/raw/*.csv")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df.to_csv("dataset/raw/Combined Data.csv", index=False)