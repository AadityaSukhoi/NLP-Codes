"""
Simple Next Word Predictor (Assignment Ready)

- Unigram, Bigram, Trigram
- Laplace Smoothing
- Top-K Predictions
- Perplexity
- Streamlit UI (ChatGPT-style)
"""

import streamlit as st
import re
import math
from collections import defaultdict, Counter

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    with open("dataset.txt", "r", encoding="utf-8") as f:
        return f.readlines()

# -------------------------------
# PREPROCESSING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def tokenize(text):
    return text.split()

def preprocess(sentences):
    corpus = []
    for s in sentences:
        tokens = tokenize(clean_text(s))
        if tokens:
            corpus.append(tokens)
    return corpus

# -------------------------------
# BUILD MODELS
# -------------------------------
def build_models(corpus):
    unigram = Counter()
    bigram = defaultdict(Counter)
    trigram = defaultdict(Counter)

    for sent in corpus:
        for i in range(len(sent)):
            unigram[sent[i]] += 1

            if i > 0:
                bigram[sent[i-1]][sent[i]] += 1

            if i > 1:
                trigram[(sent[i-2], sent[i-1])][sent[i]] += 1

    return unigram, bigram, trigram

# -------------------------------
# LAPLACE SMOOTHING
# -------------------------------
def laplace(count, total, V):
    return (count + 1) / (total + V)

# -------------------------------
# PREDICTION
# -------------------------------
def predict(text, unigram, bigram, trigram, V, k=3):
    words = tokenize(clean_text(text))

    # TRIGRAM
    if len(words) >= 2:
        context = (words[-2], words[-1])
        if context in trigram:
            candidates = trigram[context]
            total = sum(candidates.values())
            probs = {w: laplace(c, total, V) for w, c in candidates.items()}
            return sorted(probs, key=probs.get, reverse=True)[:k]

    # BIGRAM
    if len(words) >= 1:
        context = words[-1]
        if context in bigram:
            candidates = bigram[context]
            total = sum(candidates.values())
            probs = {w: laplace(c, total, V) for w, c in candidates.items()}
            return sorted(probs, key=probs.get, reverse=True)[:k]

    # UNIGRAM fallback
    total = sum(unigram.values())
    probs = {w: laplace(c, total, V) for w, c in unigram.items()}
    return sorted(probs, key=probs.get, reverse=True)[:k]

# -------------------------------
# PERPLEXITY
# -------------------------------
def perplexity(corpus, unigram, bigram, trigram, V):
    log_prob = 0
    N = 0

    for sent in corpus:
        for i in range(len(sent)):
            word = sent[i]

            if i >= 2:
                context = (sent[i-2], sent[i-1])
                count = trigram[context][word]
                total = sum(trigram[context].values())

            elif i == 1:
                context = sent[i-1]
                count = bigram[context][word]
                total = sum(bigram[context].values())

            else:
                count = unigram[word]
                total = sum(unigram.values())

            prob = laplace(count, total, V)
            log_prob += math.log(prob)
            N += 1

    return math.exp(-log_prob / N)

# -------------------------------
# LOAD + BUILD
# -------------------------------
data = load_data()
corpus = preprocess(data)
unigram, bigram, trigram = build_models(corpus)
V = len(unigram)

# -------------------------------
# UI
# -------------------------------
st.title("Next Word Predictor")

st.markdown("Type a sentence and get predictions using **N-gram models**.")

user_input = st.text_input("Enter your text:")

k = st.slider("Top-K Predictions", 1, 5, 3)

if user_input:
    preds = predict(user_input, unigram, bigram, trigram, V, k)

    st.markdown("### Prediction")
    st.markdown(
        f"""
**Input:** `{user_input}`  

**Top {k} Predictions:**  
**{", ".join(preds)}**
"""
    )

st.markdown("---")

if st.button("Show Model Evaluation"):
    ppl = perplexity(corpus, unigram, bigram, trigram, V)

    st.markdown("### Model Evaluation")
    st.markdown(f"**Perplexity:** `{round(ppl, 2)}`")