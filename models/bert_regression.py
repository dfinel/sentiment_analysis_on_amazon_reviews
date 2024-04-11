from insights_part import remove_links
import pandas as pd
from transformers import BertConfig, AutoTokenizer,AutoModelForSequenceClassification
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
df = pd.read_csv('/Users/danfinel/Downloads/Reviews.csv')
df = df.loc[:,['Text']].iloc[:1000]
df['Text'] = df['Text'].str.replace(r'<[^>]*>', '', regex=True)
df['Text'] = df['Text'].apply(remove_links)

config = BertConfig.from_pretrained('/config.json')
model = AutoModelForSequenceClassification.from_pretrained('/model.safetensors', config = config)
tokenizer = AutoTokenizer.from_pretrained('/model.safetensors', config = config)

def preprocess_function_regr(examples):
    return tokenizer(examples["Text"], truncation=True, max_length=64, padding = 'max_length')


ds_test_regr = Dataset.from_pandas(df)
ds_test_regr_tok = ds_test_regr.map(preprocess_function_regr,remove_columns = ['Text'])

with torch.no_grad():
    outputs = model(**ds_test_regr_tok)

predictions = outputs.logits.squeeze().tolist()