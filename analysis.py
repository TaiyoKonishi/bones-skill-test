from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import json

# set model name
model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# create sentiment pipeline
pipe = pipeline("sentiment-analysis", model = model_name, tokenizer = tokenizer)

# load transcript from file
def load_transcript(file_path = 'transcript.txt'):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        return json.load(f)

# run sentiment analysis
def compute_sentiment(text):
    result = pipe(text)[0]

    if result['label'] == "LABEL_0":
        result['label'] = "Negative"
    elif result['label'] == "LABEL_1":
        result['label'] = "Neutral"
    else:
        result['label'] = "Positive"

    label = result['label']
    score = result['score']
    return label, score

# calculate filler word ratio
def compute_filler_ratio(text):
    filler_words = [
        r"\bum\b",
        r"\blike\b",
        r"\byou know\b",
        r"\bi mean\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\bactually\b",
        r"\bjust\b"
    ]

    # normalize text
    text = text.lower()
    text = text.replace("it's", "it is").replace("n't", " not").replace("'re", " are").replace("he's", "he is")

    # count total and filler words
    total_words = len(re.findall(r"\b\w+\b", text))
    filler_count = sum(len(re.findall(pattern, text)) for pattern in filler_words)

    return filler_count / total_words if total_words > 0 else 0
    filler_words = [
        r"\bum\b",
        r"\blike\b",
        r"\byou know\b",
        r"\bi mean\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\bactually\b",
        r"\bjust\b"
    ]

    # normalize text
    text = text.lower()
    text = text.replace("it's", "it is").replace("n't", " not").replace("'re", " are").replace("he's", "he is")

    # count total and filler words
    total_words = len(re.findall(r"\b\w+\b", text))
    filler_count = sum(len(re.findall(pattern, text)) for pattern in filler_words)

    return filler_count / total_words if total_words > 0 else 0