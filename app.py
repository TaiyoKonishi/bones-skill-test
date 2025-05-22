import streamlit as st
from analysis import load_transcript, compute_sentiment, compute_filler_ratio
import matplotlib.pyplot as plt

# color map for sentiment labels
label_colors = {
    "Positive": "green",
    "Neutral": "gray",
    "Negative": "red"
}

# set wide layout
st.set_page_config(layout = "wide")

# app title
st.title("📝 Sentiment Analysis ")

try:
    # load transcript data
    transcript = load_transcript()
    st.success("✅ Transcript loaded successfully!")

    # let user choose sentences
    st.subheader("✅ Select sentences to analyze")

    selected_sentences = []
    for i, sentence in enumerate(transcript, 1):
        if st.checkbox(f"{i:02d}. {sentence}", key = f"cb_{i}"):
            selected_sentences.append((i, sentence))

    # start analysis
    if selected_sentences and st.button("🎯 Analyze Sentiment & Filler Rate"):
        col1, spacer, col2 = st.columns([1.3, 0.2, 1])
        results = []

        # left column: show each result
        with col1:
            st.subheader("📊 Analysis Results")

            for i, sentence in selected_sentences:
                label, score = compute_sentiment(sentence)
                filler_ratio = compute_filler_ratio(sentence)

                # show sentence and results
                st.markdown(f"**{i:02d}. {sentence}**")
                st.markdown(
                    f"<span style='color:{label_colors[label]}; font-weight:bold;'>→ Sentiment: {label} (score: {score:.2f})</span>",
                    unsafe_allow_html = True
                )
                st.markdown(f"🗣️ Filler Rate: `{filler_ratio:.2%}`")

                # draw simple bar chart
                fig, ax = plt.subplots(figsize = (4, 0.4))
                ax.barh(["Sentiment Score"], [score], color = label_colors[label])
                ax.barh(["Filler Rate"], [filler_ratio], color = 'blue')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Score (0–1)")
                st.pyplot(fig)

                st.markdown("---")

                # store result
                results.append({
                    "index": i,
                    "text": sentence,
                    "label": label,
                    "score": score,
                    "filler_ratio": filler_ratio
                })

        # right column: show top scores
        with col2:
            st.subheader("🏆 Top 3 Sentiment Scores")

            # top 3 positive
            pos_top = sorted([r for r in results if r['label'] == 'Positive'], key = lambda x: x['score'], reverse = True)[:3]
            # top 3 negative
            neg_top = sorted([r for r in results if r['label'] == 'Negative'], key = lambda x: x['score'], reverse = True)[:3]

            if pos_top:
                st.markdown("### 🟢 Positive")
                for r in pos_top:
                    st.markdown(f"**{r['index']:02d}.** {r['text']}")
                    st.write(f"Score: `{r['score']:.2f}`")
            else:
                st.info("No sentences labeled as Positive.")

            if neg_top:
                st.markdown("### 🔴 Negative")
                for r in neg_top:
                    st.markdown(f"**{r['index']:02d}.** {r['text']}")
                    st.write(f"Score: `{r['score']:.2f}`")
            else:
                st.info("No sentences labeled as Negative.")

            st.markdown("---")
            st.subheader("🗣️ Top 3 Filler Rates")

            # top 3 filler ratio
            filler_top = sorted(results, key = lambda x: x['filler_ratio'], reverse = True)[:3]

            for r in filler_top:
                st.markdown(f"**{r['index']:02d}.** {r['text']}")
                st.write(f"Filler Rate: `{r['filler_ratio']:.2%}`")

    elif not selected_sentences:
        # no sentence selected
        st.info("⚠️ Please select sentences to analyze.")

except FileNotFoundError:
    # file not found
    st.error("❌ transcript.txt not found.")
except Exception as e:
    # other error
    st.error(f"⚠️ An error occurred: {e}")