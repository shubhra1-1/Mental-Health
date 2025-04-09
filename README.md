🧠 Mental Health AI
An AI-powered system designed to predict mental health conditions and provide supportive responses through a smart chatbot interface. The project combines predictive modeling, conversational AI, and user privacy features to assist individuals in understanding their mental well-being.

🔍 Project Overview
Mental Health AI aims to:
Predict the likelihood of mental health issues (e.g., anxiety, depression) using demographic and workplace data.
Provide real-time, conversational support through an AI chatbot.
Ensure user privacy with secure local data handling.

⚙️ Features
🧩 Predictive Models: Logistic Regression, Random Forest, Neural Networks trained on open-source mental health datasets.
🤖 AI Chatbot: Built using LangChain, Ollama, and Gemma 2B for natural, context-aware responses.
🌐 Frontend Interface: User-friendly Streamlit + Flask UI for smooth interaction.
🔐 Data Privacy: Local .txt file storage ensures no sensitive data is sent to cloud servers.

♻️ Adaptive Learning (Upcoming):

Incremental Learning: Models update over time with new input data.
Adversarial Training: Simulated threats enhance model robustness.

🛠️ Tech Stack
Frontend: Streamlit, HTML/CSS
Backend: Flask, Python
AI & ML: Scikit-learn, TensorFlow/Keras, LangChain, Ollama, Gemma 2B
Storage: Local .txt and .csv files

🧪 How It Works
Users input demographic/workplace details.
The system predicts the likelihood of needing mental health support.
A smart chatbot engages with the user based on their needs.
Data is securely stored for future improvements without violating privacy.

📚 Dataset
Sourced from open repositories (e.g., Kaggle Mental Health in Tech)
Features: Age, Gender, self_employed, work_interfere, family_history, treatment
