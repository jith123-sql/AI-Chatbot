from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq

load_dotenv() # takes .env file 

st.set_page_config(page_title="Chatbot", page_icon="", layout = "centered")

st.title("Generative AI Chatbot")

chat_hist = []

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

#show the chat history

for message in st.session_state.chat_hist:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm =  ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature= 0.0
)
user_prompt = st.chat_input("Ask me anything!")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_hist.append({"role": "user", "content": user_prompt})

    response = llm.invoke(
        input = [{"role": "system", "content": "You are a helpful assistant."},  *st.session_state.chat_hist]
    )
    assistant_response = response.content
    st.session_state.chat_hist.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)