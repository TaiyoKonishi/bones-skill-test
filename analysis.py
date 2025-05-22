import json
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

# loading transcript file
def load_transcript(file_path = 'transcript.txt'):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        return json.load(f)

def compute_sentiment(text):
    result = pipe(text)[0]

    # Convert model output label (LABEL_0, LABEL_1, LABEL_2) to sentiment labels (Negative, Neutral, Positive)
    if result['label'] == "LABEL_0":
        result['label'] = "Negative"
    elif result['label'] == "LABEL_1":
        result['label'] = "Neutral"
    else:
        result['label'] = "Positive"

    label = result['label']
    score = result['score']
    return label, score

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

# Call function and test outcome
text = load_transcript()

# Test sentiment analyse result outcome
'''
for sentence in text:
    label, score = compute_sentiment(sentence)
    print(f"{sentence}\n→ Sentiment: {label} (Score: {score:.2f})\n")
'''

# Test filler word ratio outcome by first sentence
if isinstance(text, list) and text:
    sample_text = text[0]
    ratio = compute_filler_ratio(sample_text)
    print(f"Text: {sample_text}")
    print(f"Filler Ratio: {ratio:.2f}")
else:
    print("Transcript format is not a list or is empty.")
