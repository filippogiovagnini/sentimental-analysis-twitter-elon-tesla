import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from utilities import *
import time

# Load the XLSX file
file_path = 'twitter-elon-tesla-data.xlsx'
df = pd.read_excel(file_path)

df['Cleaned_Tweets'] = df['post'].apply(cleantwt)

print("Starting...")
start_time = time.time()

df['sentiment_score_bert'] = df['Cleaned_Tweets'].apply(lambda x: sentiment_analysis_bert(x[:512]))

end_time = time.time()
print("Done!")
print("Time taken:", end_time - start_time, "seconds")

print(df.head(20))

df.to_csv('twitter-elon-tesla-bert.csv', index=False)