# 📝 Sentiment & Filler Word Analyzer

A lightweight and interactive **Streamlit app** for performing **sentiment analysis** and computing **filler word ratios** on transcripts.  
Uses a transformer-based model from HuggingFace (`cardiffnlp/twitter-roberta-base-sentiment`) to classify emotions and visualize metrics clearly.

---

## 🚀 Quick Start

### 1. Clone and Set Up

```bash
git clone https://github.com/your_username/sentiment-filler-analyzer.git
cd sentiment-filler-analyzer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

### 2. Run the App

```bash
streamlit run app.py
```

Then open the Streamlit local URL in your browser (e.g., `http://localhost:8501`).

---

## 📊 Metrics Description

### 🔹 Sentiment Score

* Computed via `cardiffnlp/twitter-roberta-base-sentiment` (via HuggingFace Transformers)
* Output labels:

  * `LABEL_0`: Negative
  * `LABEL_1`: Neutral
  * `LABEL_2`: Positive
* Output score range: `0.0 – 1.0` (confidence score for label)

### 🔹 Filler Ratio

* Detects filler words like:

  * `um`, `like`, `you know`, `i mean`, `just`, etc.
* Computed as:

  ```
  Filler Ratio = (# of filler words) / (total number of words)
  ```
* Helps gauge clarity and fluency in spoken communication

---

## 📂 Input Format

**`transcript.txt`** must be a valid JSON array of sentence strings, e.g.:

```json
[
  "Speaker A: Baseball these days is, like, totally different from what it used to be.",
  "Speaker B: Yeah, I know. Um, the way they use data analytics now is kind of insane."
]
```

---

## 💡 In One Extra Hour, I Would Add…

* **A file upload feature** using `st.file_uploader()` to allow users to upload `.txt` or `.json` transcript files (including multiple files for batch analysis).

* **Word-level sentiment highlights** to show which words make the sentence feel positive or negative.
→ This helps users see why a sentence gets its score, useful for learning or communication training.
---