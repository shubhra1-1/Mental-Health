import os
import streamlit as st
import requests
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# 🌟 Page Config
st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="🧠", layout="wide")

# 🎨 Custom Styling
st.markdown("""
    <style>
        .user-message {
            background-color: #D1D5DB; 
            padding: 12px; 
            border-radius: 10px; 
            max-width: 80%;
            margin-left: 0;
            margin-right: auto;
            font-weight: 500;
            color: black;
            margin-bottom: 10px;
        }
        .ai-message {
            background-color: #D1D5DB; 
            padding: 12px; 
            border-radius: 10px; 
            max-width: 80%;
            margin-left: 0;
            margin-right: auto;
            font-weight: 500;
            color: black;
            margin-bottom: 20px;
        }
        div[data-testid="stTextInput"] > div {
            border-radius: 8px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)


def is_ollama_running():
    try:
        res = requests.get("http://127.0.0.1:11434")
        return res.status_code == 200
    except:
        return False


def main():
    st.title("🧠 AI Mental Health Chatbot 💬")

    if not is_ollama_running():
        st.error("🚫 Ollama is not running. Please start the server using: `ollama run gemma:2b`.")
        return

    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    if "latest_input" not in st.session_state:
        st.session_state["latest_input"] = ""

    system_prompt = """
    You are a supportive and empathetic AI specializing in mental health support.
    Your goal is to provide comfort, active listening, and helpful coping strategies in a warm, understanding tone.
    """

    template = f"""
    {system_prompt}

    Here is the conversation history:
    {{context}}

    User: {{question}}
    AI:
    """

    model = OllamaLLM(model="gemma:2b", num_predict=1000)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    st.subheader("💬 Chat History")
    for message in st.session_state["conversation_history"]:
        role = "user-message" if message["role"] == "user" else "ai-message"
        label = "User" if message["role"] == "user" else "AI"
        st.markdown(f'<div class="{role}"><b>{label}:</b> {message["content"]}</div>', unsafe_allow_html=True)

    user_message = st.text_input("How are you feeling today?",
                                 key=f"input_{len(st.session_state['conversation_history'])}",
                                 placeholder="Type your message...")

    def generate_response():
        user_input = user_message.strip()
        if not user_input or user_input == st.session_state["latest_input"]:
            return

        st.session_state["latest_input"] = user_input
        st.session_state["conversation_history"].append({"role": "user", "content": user_input})

        if user_input.lower() in ["bye", "goodbye", "exit", "quit"]:
            response = "Goodbye! Take care 😊. If you ever need to talk again, I'm here for you."
            st.session_state["conversation_history"].append({"role": "ai", "content": response})
            st.success("Chatbot session ended. Thank you for chatting! 🙏")
            st.rerun()

        else:
            response = ""
            with st.spinner("Thinking..."):
                for chunk in chain.stream({
                    "context": "\n".join([msg["content"] for msg in st.session_state["conversation_history"]]),
                    "question": user_input
                }):
                    response += chunk

            st.session_state["conversation_history"].append({"role": "ai", "content": response})
            st.rerun()

    if st.button("Send"):
        generate_response()


# ✅ This is needed to run the app only when called directly
if __name__ == "__main__":
    main()
