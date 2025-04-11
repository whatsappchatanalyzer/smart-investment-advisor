import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from model_utils import load_model, predict_trend
import openai
import torch

openai.api_key = st.secrets.get("OPENAI_API_KEY", "your-api-key")  # Secure API key
model = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="Smart Investment Advisor", layout="centered")
st.title("📈 Smart Investment Advisor")
st.caption("Combining RNN Predictions + GPT Insights + Real Data")

symbol = st.text_input("🔍 Enter stock symbol (e.g. AAPL):", value="AAPL")

if st.button("Fetch & Predict"):
    try:
        data = yf.download(symbol, period="1mo", interval="1d")
        if data.empty:
            st.error("No data found for symbol.")
        else:
            prices = data['Close'].tolist()[-30:]
            prediction = predict_trend(model, prices)

            gpt_prompt = (
                f"The RNN model predicts a value of {prediction:.2f} for {symbol}, "
                f"based on recent prices: {prices[-5:]}. "
                "Should the user consider buying, holding, or selling the stock?"
            )

            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": gpt_prompt}]
            )

            advice = gpt_response.choices[0].message.content

            st.subheader("📊 Stock Price Chart")
            st.line_chart(prices)

            st.subheader("🤖 RNN Prediction")
            st.info(f"Predicted next-day price: **${prediction:.2f}**")

            st.subheader("💬 GPT Investment Advice")
            st.success(advice)

            st.session_state.history.append({
                "symbol": symbol,
                "prediction": prediction,
                "advice": advice,
                "recent": prices[-5:]
            })

    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.session_state.history:
    st.subheader("🕓 Prediction History")
    for entry in reversed(st.session_state.history):
        st.markdown(f"**{entry['symbol']}** → ${entry['prediction']:.2f} | GPT: {entry['advice']}")
