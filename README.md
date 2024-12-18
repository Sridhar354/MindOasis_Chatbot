🧠 Mental Health Counseling Chatbot
This project is a simple text-based mental health counseling chatbot powered by DistilGPT-2, fine-tuned on the Psych8k dataset using Hugging Face. It provides supportive, empathetic responses through a web interface built with Streamlit.

📦 Features
Text-Based Chatbot: Accepts text input and returns text responses.
Fine-Tuned Model: Trained for 45 epochs using the Psych8k dataset.
Web Interface: Simple and interactive chat interface using Streamlit.
Preprocessing: Automatic text cleaning (removing punctuation, stopwords, and normalizing text).

🛠️ Setup Instructions
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/mental-health-chatbot.git
cd mental-health-chatbot

3. Install Required Libraries
bash
Copy code
pip install transformers datasets torch streamlit huggingface_hub nltk

5. Log into Hugging Face Hub
python
Copy code
from huggingface_hub import login
login("your_huggingface_token")

7. Run the Chatbot
bash
Copy code
streamlit run mental_health_chatbot.py
📊 Model Details
Dataset: EmoCareAI/Psych8k
Model: DistilGPT-2 (fine-tuned for 45 epochs)
Batch Size: 4
Learning Rate: 5e-5

📜 Disclaimer
This chatbot is not a substitute for professional mental health services. It provides supportive responses based on pre-trained data but is not a replacement for therapy or clinical support.




