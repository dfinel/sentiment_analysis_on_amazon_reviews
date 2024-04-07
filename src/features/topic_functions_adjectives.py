from collections import Counter
import spacy
from tqdm import tqdm
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# Define subject-pronouns to remove
personal_pronouns = ['i', 'me', 'you', 'he', 'she', 'it', 'we', 'us']  # removed 'they', 'them'


def create_sub_phrases(data):
    """Returns lists of tuple phrases"""
    wave_1 = []
    wave_2 = []

    for i in tqdm(range(len(data))):
        sente = nlp(data.iloc[i])
        for token in sente:
            noun = ''
            adj = ''
            adverb = ''
            neg = ''
            adv_verb = ''
            subject = ''
            dobj_text = ''
            adj_text = ''
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

                    if child.dep_ in ['nsubj',
                                      'nsubjpass'] and child.text.lower() not in personal_pronouns:  # nsubjpass for passive voice
                        subject = child.text

                    elif child.dep_ == 'dobj' and child.pos_ == 'NOUN':
                        dobj_text = child.text
                        for grandchild in child.children:
                            if grandchild.dep_ == 'amod' and grandchild.pos_ == 'ADJ':
                                adj_text = grandchild.text

                if noun and adj:
                    if adverb:
                        wave_1.append((noun, token.text, adverb, adj))
                    elif adv_verb and neg:
                        wave_1.append((noun, token.text, adv_verb, neg, adj))
                    elif neg:
                        wave_1.append((noun, token.text, neg, adj))
                    else:
                        wave_1.append((noun, token.text, adj))

                if subject and dobj_text and adj_text:
                    wave_2.append((subject, token.text, adj_text, dobj_text))
    return wave_1 + wave_2


def get_topics(phrases):
    """Returns list of top 5 topics"""
    topic_grams = []
    topics = []
    for phrase in phrases:
        if len(phrase) == 3:
            uni_gram = nlp(phrase[2])[0].lemma_
            topic_grams.append(uni_gram)
        if len(phrase) == 4:
            bi_gram = phrase[2] + " " + phrase[3]
            topic_grams.append(bi_gram)
    counter = Counter(topic_grams).most_common(5)
    for tuple in counter:
        topics.append(tuple[0])
    return topics


def load_data(file_path):
    """Load and return the content of the file at the given path"""
    try:
        dataframe = pd.read_csv("file_path")
        return dataframe
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None


if __name__ == "__main__":
    path = "../data/amazon_reviews.csv"
    df = pd.read_csv(path)
    reviews = df.content
    phrases = create_sub_phrases(reviews)
    topics = get_topics(phrases)
    print(topics)

