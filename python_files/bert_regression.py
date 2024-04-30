import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
from itertools import chain
import re
def remove_links(review):
    pattern = r'\bhttps?://\S+'
    return re.sub(pattern, '', review)


# df = pd.read_csv('/Users/danfinel/Downloads/Reviews.csv')
# df = df.loc[:,['Text']].iloc[:1000]
# df['Text'] = df['Text'].str.replace(r'<[^>]*>', '', regex=True)
# df['Text'] = df['Text'].apply(remove_links)

model = AutoModelForSequenceClassification.from_pretrained(
  '../topic_magnet/bert_regr_other_pretrained', num_labels = 1)
tokenizer = AutoTokenizer.from_pretrained(
  '../topic_magnet/bert_regr_other_pretrained')

def preprocess_function_regr(examples):
    return tokenizer(examples["Text"], truncation=True, max_length=64, padding = 'max_length')

def get_predictions(reviews):
  #new_test = pd.DataFrame(reviews)
  new_ds_regr = Dataset.from_pandas(reviews)
  new_ds_regr_tok = new_ds_regr.map(preprocess_function_regr, remove_columns = ['Text'])
  input_ids = torch.tensor(new_ds_regr_tok['input_ids'])
  token_type_ids = torch.tensor(new_ds_regr_tok['token_type_ids'])
  attention_mask = torch.tensor(new_ds_regr_tok['attention_mask'])
  with torch.no_grad():
    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    predictions = outputs.logits
  return predictions

def get_ratings_perc(reviews):
  preds = get_predictions(reviews)
  predictions_list = list(chain.from_iterable(preds.tolist()))
  predictions_array = np.clip(predictions_list,1,5)
  predictions_array = [round(x) for x in predictions_array]
  sum = np.unique(predictions_array, return_counts = True)[1].sum()
  ratings_perc = np.unique(predictions_array, return_counts = True)[1]/sum *100
  return ratings_perc

def get_ratings_dic(reviews):
  ratings_perc = get_ratings_perc(reviews)
  dic = {}
  for i in range(1,6):
    dic[i] = ratings_perc[i-1].round(2)
  return dic

#print(get_ratings_dic(df))




# new_test = pd.DataFrame(df.loc[:,'Text'].iloc[:1000])
# new_ds_regr = Dataset.from_pandas(new_test)
# new_ds_regr_tok = new_ds_regr.map(preprocess_function_regr, remove_columns = ['Text'])
#
# input_ids = torch.tensor(new_ds_regr_tok['input_ids'])
# token_type_ids = torch.tensor(new_ds_regr_tok['token_type_ids'])
# attention_mask = torch.tensor(new_ds_regr_tok['attention_mask'])
# with torch.no_grad():
#   outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#   predictions = outputs.logits
#
# predictions_list = list(chain.from_iterable(predictions.tolist()))
# predictions_array = np.clip(predictions_list,1,5)
# predictions_array = [round(x) for x in predictions_array]
# print(np.unique(predictions_array, return_counts = True))