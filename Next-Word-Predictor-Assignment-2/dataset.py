import nltk
nltk.download('brown')
from nltk.corpus import brown

with open("dataset.txt", "w") as f:
    for sent in brown.sents():
        f.write(" ".join(sent) + "\n")