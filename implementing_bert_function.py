from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

tokens = tokenizer.encode("It was good but could have been better. Great", return_tensors="pt")
tokenizer.decode(tokens[0])

result = model(tokens)

int(torch.argmax(result.logits)) + 1