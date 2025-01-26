import numpy as np
import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from code_gpus.utilities import *

# Load the XLSX file
file_path = 'twitter-elon-tesla-data.xlsx'
df = pd.read_excel(file_path)

df['Cleaned_Tweets'] = df['post'].apply(cleantwt)

df['sentiment_score_bert'] = df['Cleaned_Tweets'].apply(lambda x: sentiment_analysis_bert(x[:512]))

df.to_csv('twitter-elon-tesla-bert.csv', index=False)