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

# Call function and test outcome
text = load_transcript()

# Test sentiment analyse result outcome
for sentence in text:
    label, score = compute_sentiment(sentence)
    print(f"{sentence}\n→ Sentiment: {label} (Score: {score:.2f})\n")
