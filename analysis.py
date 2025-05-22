import json

# loading transcript file
def load_transcript(file_path = 'transcript.txt'):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        return json.load(f)

# Call function and test outcome
text = load_transcript()
print(text)