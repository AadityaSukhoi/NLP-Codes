# NLP-Based Text Analysis System (Streamlit)

## Overview
This project is a web-based NLP application built using Streamlit that performs comprehensive text analysis using classical Natural Language Processing techniques.  

The system processes raw text input and transforms it into structured insights such as keywords, named entities, sentiment, and vector representations — all **without using any external APIs**.

---

## Objective
To design and implement an interactive NLP pipeline that:
- Processes unstructured text
- Extracts meaningful insights
- Demonstrates core NLP concepts using local models and libraries

---

## Features

### 1. Text Preprocessing
- Tokenization
- Lowercasing
- Stop-word removal
- Stemming
- Lemmatization  
✔ Each step is displayed clearly in the UI

---

### 2. Keyword Extraction
- Implemented using **TF-IDF**
- Displays top important words from the input text

---

### 3. Named Entity Recognition (NER)
- Implemented using **spaCy (local model)**
- Extracts:
  - Person
  - Organization
  - Location
  - Date  
✔ Entities are also highlighted in the text

---

### 4. Sentiment Analysis
- Implemented using **VADER (NLTK)**
- Outputs:
  - Sentiment (Positive / Negative / Neutral)
  - Polarity scores

---

### 5. N-Gram Analysis
- Unigrams and Bigrams generated
- Displays most frequent combinations

---

### 6. Bag-of-Words (BoW)
- Converts text into vector representation
- Displays:
  - Vocabulary size
  - Sample vector

---

### 7. Visualization
- Word frequency bar chart
- Helps in understanding text distribution visually

---

### 8. User Interface
- Built using **Streamlit**
- Features:
  - Text input box
  - File upload (.txt)
  - Sidebar for selecting NLP operations

---

## Tech Stack

- Python
- Streamlit
- NLTK
- spaCy
- scikit-learn
- Matplotlib

---

## Steps to Run the Project

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd nlp-streamlit-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 4. Run the Application
``` bash
streamlit run app.py
```
## Input Options
- Direct text input via UI
- Upload `.txt` files

## Sample Input
```
Elon Musk visited India in 2024 and discussed AI innovation.
```

## Sample Output
- Tokens: elon, musk, visited, india, 2024...
- Keywords: musk, india, ai, innovation
- Entities:
    - Elon Musk → Person
    - India → Location
    - 2024 → Date
- Sentiment: Neutral
- Bigrams: ai innovation, visited india

## Project Structure
```
NLP-Assignment/
│── app.py
│── requirements.txt
│── README.md
```
## Constraints Followed
- No external APIs used
- All NLP processing done locally
- Used only NLTK, spaCy, and scikit-learn

## Leanring Outcome
- Built a complete NLP pipeline
- Applied preprocessing and feature extraction
- Implemented real-world NLP techniques
- Developed an interactive web app using Streamlit
