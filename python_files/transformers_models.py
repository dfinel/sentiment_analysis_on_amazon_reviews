from transformers import pipeline
import spacy

nlp = spacy.load('en_core_web_sm')
sentiment_pipeline = pipeline("sentiment-analysis", model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')
classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

nlp.to_disk('spacy_model')

sentiment_pipeline.save_pretrained('my_sentiment_model')

classifier.save_pretrained('my_zero_shot')