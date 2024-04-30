import pandas as pd
import numpy as np
import re
from sklearn.model_selection import GroupShuffleSplit
def remove_links(review):
    pattern = r'\bhttps?://\S+'
    return re.sub(pattern, '', review)

df = pd.read_csv('/Users/danfinel/Downloads/Reviews.csv')
df['Text'] = df['Text'].str.replace(r'<[^>]*>', '', regex=True)
df['Text'] = df['Text'].apply(remove_links)


splitter_temp = GroupShuffleSplit(test_size=.40, n_splits=1, random_state = 42)
split_temp = splitter_temp.split(df[:100000], groups=df[:100000]['ProductId'])
train_inds, temp_inds = next(split_temp)

train = df.iloc[train_inds]
temp = df.iloc[temp_inds]

splitter_val = GroupShuffleSplit(test_size=.50, n_splits=1, random_state = 42)
split_val = splitter_val.split(temp, groups=temp['ProductId'])
val_inds, test_inds = next(split_val)

val = temp.iloc[val_inds]
test = temp.iloc[test_inds]

X_train = train.drop(columns = 'Score')
y_train = train.Score

X_val = val.drop(columns = 'Score')
y_val = val.Score

X_test = test.drop(columns = 'Score')
y_test = test.Score

from transformers import AutoTokenizer,AutoModelForSequenceClassification
base_model = 'bert-base-cased'
learning_rate = 2e-5
max_length = 64
batch_size = 32
epochs = 5
nbr_samples = 10000
tokenizer_regr = AutoTokenizer.from_pretrained(base_model)
model_regr = AutoModelForSequenceClassification.from_pretrained(base_model,num_labels = 1)

X_train_bert = X_train.iloc[:nbr_samples]
del X_train_bert['ProductId']
X_train_bert['label'] = y_train.iloc[:nbr_samples].astype(float)

X_val_bert = X_val.iloc[:nbr_samples]
del X_val_bert['ProductId']
X_val_bert['label'] = y_val.iloc[:nbr_samples].astype(float)

from datasets import Dataset
ds_train_regr = Dataset.from_pandas(X_train_bert)
ds_val_regr = Dataset.from_pandas(X_val_bert)

def preprocess_function_regr(examples):
    return tokenizer_regr(examples["Text"], truncation=True, max_length=64, padding = 'max_length')

ds_train_regr_tok = ds_train_regr.map(preprocess_function_regr, remove_columns = ['Text'])
ds_val_regr_tok = ds_val_regr.map(preprocess_function_regr, remove_columns = ['Text'])

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    accuracy = sum([1 for e in single_squared_errors if e < 0.25]) / len(single_squared_errors)

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}

from transformers import TrainingArguments

output_dir = ".."

training_args = TrainingArguments(
    output_dir = output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    weight_decay=0.01,
)
from transformers import Trainer
import torch
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = Trainer(
    model=model_regr,
    args=training_args,
    train_dataset=ds_train_regr_tok,
    eval_dataset=ds_val_regr_tok,
    compute_metrics=compute_metrics_for_regression
)

trainer.train()

tokenizer_regr.save_pretrained('.')
model_regr.save_pretrained('.', from_pt = True)