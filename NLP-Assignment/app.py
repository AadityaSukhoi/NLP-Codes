"""
Streamlit NLP Text Analysis System
Author: Aaditya Ranjan Moitra

This application performs NLP tasks locally without using external APIs.
"""

import streamlit as st
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# ------------------------------
# Streamlit UI
# ------------------------------

st.title("NLP Text Analysis System")

st.sidebar.title("Options")
option = st.sidebar.selectbox(
    "Choose NLP Operation",
    ["All", "Preprocessing", "Keywords", "NER", "Sentiment", "N-Grams", "BoW"]
)

# Input
text_input = st.text_area("Enter Text")

uploaded_file = st.file_uploader("Upload .txt file", type=["txt"])

if uploaded_file:
    text_input = uploaded_file.read().decode("utf-8")

# ------------------------------
# NLP FUNCTIONS
# ------------------------------

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    lower = [word.lower() for word in tokens]
    no_punct = [w for w in lower if w not in string.punctuation]
    no_stop = [w for w in no_punct if w not in stop_words]
    stemmed = [stemmer.stem(w) for w in no_stop]
    lemmatized = [lemmatizer.lemmatize(w) for w in no_stop]

    return tokens, lower, no_stop, stemmed, lemmatized

def keyword_extraction(text):
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

def ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities, doc

def sentiment(text):
    score = sia.polarity_scores(text)
    compound = score['compound']

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, score

def ngrams(text):
    tokens = nltk.word_tokenize(text.lower())

    unigrams = Counter(tokens)
    bigrams = Counter(nltk.bigrams(tokens))

    return unigrams.most_common(10), bigrams.most_common(10)

def bow(text):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out(), X.toarray()

# ------------------------------
# EXECUTION
# ------------------------------

if text_input:

    st.subheader("Input Text")
    st.write(text_input)

    # PREPROCESSING
    if option in ["All", "Preprocessing"]:
        st.subheader("Preprocessing")

        tokens, lower, no_stop, stemmed, lemmatized = preprocess(text_input)

        st.write("Tokens:", tokens)
        st.write("Lowercase:", lower)
        st.write("Stopword Removed:", no_stop)
        st.write("Stemmed:", stemmed)
        st.write("Lemmatized:", lemmatized)

    # KEYWORDS
    if option in ["All", "Keywords"]:
        st.subheader("Keywords (TF-IDF)")
        keywords = keyword_extraction(text_input)
        st.write(keywords)

    # NER
    if option in ["All", "NER"]:
        st.subheader("Named Entities")
        entities, doc = ner(text_input)

        for ent in entities:
            st.write(f"{ent[0]} → {ent[1]}")

        # Highlight
        st.write("Highlighted Text:")
        st.write(spacy.displacy.render(doc, style="ent"), unsafe_allow_html=True)

    # SENTIMENT
    if option in ["All", "Sentiment"]:
        st.subheader("Sentiment Analysis")

        label, score = sentiment(text_input)
        st.write("Sentiment:", label)
        st.write("Scores:", score)

    # N-GRAMS
    if option in ["All", "N-Grams"]:
        st.subheader("N-Grams")

        uni, bi = ngrams(text_input)

        st.write("Top Unigrams:", uni)
        st.write("Top Bigrams:", bi)

    # BoW
    if option in ["All", "BoW"]:
        st.subheader("Bag of Words")

        vocab, vector = bow(text_input)
        st.write("Vocabulary Size:", len(vocab))
        st.write("Sample Vector:", vector)

    # VISUALIZATION
    st.subheader("Visualization")

    words = nltk.word_tokenize(text_input.lower())
    freq = Counter(words)

    common = freq.most_common(10)

    words = [w[0] for w in common]
    counts = [w[1] for w in common]

    fig, ax = plt.subplots()
    ax.bar(words, counts)
    plt.xticks(rotation=45)

    st.pyplot(fig)