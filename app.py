from transformers import pipeline
import streamlit as st

st.set_page_config(page_title="🧠 AI Mental Health Chatbot", page_icon="🧠")
st.title("🧠 AI Mental Health Chatbot")
st.write("Built by **Sampath**")

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=False
    )

sentiment_model = load_model()

responses = {
    "joy":      "😊 That's wonderful! Keep that positive energy!",
    "sadness":  "💙 I'm sorry you're feeling sad. It's okay to feel this way.",
    "anger":    "😤 Take a deep breath. Things will get better.",
    "fear":     "🤗 You're stronger than you think!",
    "disgust":  "😟 That sounds unpleasant. I'm here to listen.",
    "surprise": "😮 Sounds unexpected! How are you feeling about it?",
    "neutral":  "🙂 Tell me more about how your day is going?"
}

user_input = st.text_area("How are you feeling today?", placeholder="Type here...")

if st.button("Analyze & Respond"):
    if user_input.strip():
        result = sentiment_model(user_input)[0]
        emotion = result["label"].lower()
        confidence = round(result["score"] * 100, 2)
        bot_response = responses.get(emotion, "I'm here for you.")

        st.success(f"🧠 Detected Emotion: **{emotion.upper()}** ({confidence}% confidence)")
        st.info(f"🤖 Chatbot: {bot_response}")
    else:
        st.warning("Please type something!")
