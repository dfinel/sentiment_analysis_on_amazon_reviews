import grequests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm
import spacy
from collections import Counter
from transformers import pipeline
from flask import Flask
from bert_regression import get_ratings_dic
import os
from langchain.llms import OpenAI
import gradio as gr
import asyncio



os.environ["OPENAI_API_KEY"] = ""


nlp = spacy.load('../topic_magnet/spacy_model')
sentiment_pipeline = pipeline("sentiment-analysis", model='../topic_magnet/my_sentiment_model')
classifier = pipeline(task="zero-shot-classification", model="../topic_magnet/my_zero_shot")


custom_headers = {
    # Eliminating non-english reviews
    "Accept-language": "en;q=1.0",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "User-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
}


def get_soup(response):
    if response.status_code != 200:
        print("Error in getting webpage")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def get_soup_reviews(soup):
    review_elements = soup.select("div.review")

    scraped_reviews = []

    for review in review_elements:
        r_content_element = review.select_one("span.review-text")
        r_content = r_content_element.text if r_content_element else None
        preprocessed_review = r_content.replace('\n', '')

        scraped_reviews.append(preprocessed_review)

    return scraped_reviews


def scrape_reviews(base_url):
    all_reviews = []
    star_ratings = ['one', 'two', 'three', 'four', 'five']

    for star in tqdm(star_ratings):
        page_number = 1

        while True:
            url = f"{base_url}&filterByStar={star}_star&&pageNumber={page_number}"
            response = grequests.get(url, headers=custom_headers).send().response
            soup = get_soup(response)

            if not soup:
                continue  # Skip to next star rating if unable to parse page

            reviews = get_soup_reviews(soup)
            all_reviews.extend(reviews)

            # Note: there's a valid page for any pageNumber,
            # so we need to stop scraping based on the button of next page
            # Check for the presence of the "Next page" element
            next_page_element = soup.find("li", class_="a-disabled a-last")
            if next_page_element:
                break  # Exit loop if "Next page" element is found

            page_number += 1

    return all_reviews
def remove_links(review):
    pattern = r'\bhttps?://\S+'
    return re.sub(pattern, '', review)


def preprocess_data(df):
    df.rename(columns={'content': 'Text'}, inplace = True)
    df.Text = df.Text.astype(str)
    df['Text'] = df['Text'].str.replace(r'<[^>]*>', '', regex=True)
    df['Text'] = df['Text'].apply(remove_links)
    return df


def get_noun_ver_adj(reviews):
    noun_ver_adj = []
    for i in tqdm(range(reviews.shape[0])):
        sente = nlp(reviews.iloc[i])
        for token in sente:
            noun = adj = adverb = adv_verb = neg = ''
            if token.dep_ == 'ROOT':
                for child in token.children:
                    if child.pos_ == 'NOUN':
                        noun = child.text
                    elif child.pos_ == 'ADJ':
                        adj = child.text
                        for other_child in child.children:
                            if other_child.pos_ == 'ADV':
                                adverb = other_child.text
                    elif child.pos_ == 'ADV':
                        adv_verb = child.text
                    elif child.pos_ == 'PART':
                        neg = child.text
                if noun and adj:
                    if adverb:
                        noun_ver_adj.append((noun, token.text, adverb, adj))
                    elif adv_verb and neg:
                        noun_ver_adj.append((noun, token.text, adv_verb, neg, adj))
                    elif neg:
                        noun_ver_adj.append((noun, token.text, neg, adj))
                    else:
                        noun_ver_adj.append((noun, token.text, adj))
    return noun_ver_adj


def get_most_common_noun(noun_ver_adj):
    element_counts_lemma_noun = Counter(nlp(item[0].lower())[0].lemma_ for item in noun_ver_adj)
    most_common_noun = list(map(lambda x: x[0], element_counts_lemma_noun.most_common(10)))
    return most_common_noun[:5]


def get_insights(topic, noun_ver_adj):
    list_tuples = [' '.join(x) for x in noun_ver_adj if nlp(x[0].lower())[0].lemma_ == topic]
    results = sentiment_pipeline(list_tuples)
    pos = 0
    neg = 0
    pos_adj = []
    neg_adj = []
    for sentence, result in zip(list_tuples, results):
        if result['label'] == 'POSITIVE':
            pos += 1
            pos_adj.append(sentence.rsplit(None, 1)[-1].lower())
        else:
            neg += 1
            neg_adj.append(sentence.rsplit(None, 1)[-1].lower())
    most_common_pos_adj = list(map(lambda x: x[0], Counter(pos_adj).most_common(5)))
    most_common_neg_adj = list(map(lambda x: x[0], Counter(neg_adj).most_common(5)))
    return most_common_pos_adj, most_common_neg_adj


def get_df_all_topics_sent(reviews, sentiment, most_common_noun, threshold=0.6):
    # Get the dataframe of all topics with the corresponding sentiment (positive or negative)
    reviews_list = reviews.to_list()
    hypothesis = f'This product review reflect a {sentiment} sentiment of the {{}}'
    df_sent = classifier(reviews_list, most_common_noun, hypothesis_template=hypothesis, multi_label=True)
    df_sent = pd.DataFrame(df_sent)
    df_sent = df_sent.set_index('sequence').apply(pd.Series.explode).reset_index()
    df_sent = df_sent[df_sent['scores'] >= threshold]
    return df_sent


def get_both_df(reviews,most_common_noun):
    # get both df and remove indexes from the positive and negative dataframes where the score is higher in one or the other df
    df_pos = get_df_all_topics_sent(reviews, 'positive', most_common_noun)
    print('done')
    df_neg = get_df_all_topics_sent(reviews, 'negative', most_common_noun)
    merged_df = pd.merge(df_pos, df_neg, on=['sequence', 'labels'], suffixes=('_pos', '_neg'))
    to_remove_pos = merged_df[merged_df.scores_pos < merged_df.scores_neg][['sequence', 'labels']]
    indexes_pos_to_remove = df_pos.reset_index().merge(to_remove_pos, on=['sequence', 'labels'], how='inner').set_index(
        'index').index
    to_remove_neg = merged_df[merged_df.scores_pos > merged_df.scores_neg][['sequence', 'labels']]
    indexes_neg_to_remove = df_neg.reset_index().merge(to_remove_pos, on=['sequence', 'labels'], how='inner').set_index(
        'index').index
    df_pos.drop(index=indexes_pos_to_remove, inplace=True)
    df_neg.drop(index=indexes_neg_to_remove, inplace=True)
    return df_pos, df_neg


def get_df_sent_topic(topic, df_all_topic_sentim):
    # get the reviews of a specific topic corresponding to the given sentiment
    df_topic = df_all_topic_sentim[df_all_topic_sentim.labels == topic].copy()
    df_topic.drop(columns=['labels', 'scores'], inplace=True)
    return df_topic


def get_percentages_topic(topic, df_all_topic_pos, df_all_topic_neg):
    # get percentages of positive and negative reviews for the given topic
    df_pos = get_df_sent_topic(topic, df_all_topic_pos)
    df_neg = get_df_sent_topic(topic, df_all_topic_neg)
    pos_perc = round(df_pos.shape[0] / (df_pos.shape[0] + df_neg.shape[0]) * 100, 2)
    neg_perc = round(df_neg.shape[0] / (df_pos.shape[0] + df_neg.shape[0]) * 100, 2)
    return pos_perc, neg_perc


def get_df_adjectives(sentiment, reviews, topic,df_all_topic_sent, noun_ver_adj, threshold=0.6):
    reviews_list = reviews.to_list()
    if sentiment == 'positive':
        adj = get_insights(topic, noun_ver_adj)[0]
    else:
        adj = get_insights(topic, noun_ver_adj)[1]
    hypothesis = f'The {sentiment} sentiment representing the product {topic} is {{}}'
    df_topic = get_df_sent_topic(topic, df_all_topic_sent)
    df_adj = classifier(df_topic.sequence.tolist(), adj, hypothesis_template=hypothesis, multi_label=True)
    df_adj = pd.DataFrame(df_adj)
    df_adj = df_adj.set_index('sequence').apply(pd.Series.explode).reset_index()
    df_adj = df_adj[df_adj['scores'] >= threshold]
    return (df_adj.labels.value_counts(normalize=True).values.round(2) * 100).astype(int), df_adj.labels.value_counts(
        normalize=True).index.values.astype(str)

def get_topics_adjectives(most_common_noun, noun_ver_adj):
    dic = {}
    for i in range(5):
        dic[most_common_noun[i]] = get_insights(most_common_noun[i], noun_ver_adj)
    return dic

def generate_feedback(dic, temperature = 0.9):
  text = f"""Create a summary adressed to a business owner of a product about its reviews.
We provide the main topics of the reviews with their main attributes. 
For each topic which are the keys of the dictionary, the first list is positive adjectives and the second is negative.
Start the text by : 'Dear business owner,'
You have to create subpart for each topic and explain on the first part of each topic the positive attributes by writing :
topic :
positive feedbacks : sentences explaining the positive feedbacks
negative feedbacks : sentences explaining the negative feedbacks
Finish the text by signing with this company name : 'Topic Magnet'.
 
: {dic}
   """
  llm = OpenAI(temperature = temperature, max_tokens = 1000)
  generated_text = llm(text)
  return generated_text.strip()



def get_reviews(url = 'https://www.amazon.co.uk/product-reviews/B0B21DW5DL/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_review'):
    df = pd.DataFrame({'Text': scrape_reviews(url)})
    df = preprocess_data(df)
    reviews = df.Text
    noun_ver_adj = get_noun_ver_adj(reviews)
    most_common_noun = get_most_common_noun(noun_ver_adj)
    dic1 = get_topics_adjectives(most_common_noun, noun_ver_adj)
    dic2 = get_ratings_dic(df)
    dic3 = pd.DataFrame({"Ratings": list(dic2.keys()), "Values in %": list(dic2.values())}).to_markdown(index = False)
    generated_text = generate_feedback(dic1)
    return dic3,generated_text

# gr.Interface(fn = get_reviews, inputs = gr.Textbox(), outputs = gr.Textbox(), title = 'The Topic Magnet',
#             description = 'Enter the url of your Amazon reviews to get real ratings and valuable insights').launch(share = True)

#print(get_reviews())

if __name__ == '__main__':
    interface = gr.Interface(fn=get_reviews, inputs=gr.Textbox(label = 'Enter the URL of your product reviews'), outputs=[gr.Markdown(label = 'True ratings in %'),gr.Textbox(label = 'Actionable insights')], title='Topic Magnet',
                             description='Enter the url of your reviews to get real ratings and valuable insights !',
                             examples = ['https://www.amazon.co.uk/product-reviews/B0B21DW5DL/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&reviewerType=all_review',
                                         'https://www.amazon.co.uk/Amazon-Brand-Solimo-Complete-Adult/product-reviews/B07GFLKX76/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_review',
                                         'https://www.amazon.co.uk/Twinings-English-Breakfast-Multipack-Biodegradable/product-reviews/B0BT53VW1N/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_review',
                                         'https://www.amazon.co.uk/PG-Tips-Pyramid-Bags-Total/product-reviews/B07CJGT17P/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_review',
                                         ])
    interface.launch(share = True)
    #app.run(host = '0.0.0.0', debug = True, port = 5000)


#print(most_common_noun)
#print(get_insights(most_common_noun[0],noun_ver_adj))

#dfs_topics = get_both_df(reviews,most_common_noun)
#df_all_topic_pos = dfs_topics[0]
#df_all_topic_neg = dfs_topics[1]
#print(get_percentages_topic(most_common_noun[0],df_all_topic_pos,df_all_topic_neg))
#print(get_df_adjectives('positive',reviews,most_common_noun[0],noun_ver_adj))

