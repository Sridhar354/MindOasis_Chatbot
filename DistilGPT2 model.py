import os
import torch
import streamlit as st
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from huggingface_hub import login
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK Data
nltk.download("stopwords")
nltk.download("punkt")

# -------------------------------
# 1. Load and Preprocess the Dataset
# -------------------------------
def load_and_preprocess_data():
    """
    Load and preprocess the EmoCareAI/Psych8k dataset.
    """
    # Load dataset from Hugging Face
login("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")

dataset = load_dataset('EmoCareAI/Psych8k')

# Check the structure of the dataset
print(dataset)

    # Load stopwords
    stop_words = set(stopwords.words('english'))

    # Preprocess the text
    def preprocess_text(text):
        # Lowercase the text
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Join tokens back to form clean text
        clean_text = " ".join(filtered_tokens)

        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    # Extract valid text, preprocess, and filter
    processed_texts = []
    for item in dataset:
        if 'text' in item and item['text']:
            clean_text = preprocess_text(item['text'])
            
            # Filter texts that are too short or too long
            token_count = len(clean_text.split())
            if 5 <= token_count <= 100:  # Keep texts with token count between 5 and 100
                processed_texts.append(clean_text)

    return processed_texts

# -------------------------------
# 2. Tokenize the Dataset
# -------------------------------
def tokenize_data(texts, tokenizer):
    """
    Tokenize the dataset using the GPT2 tokenizer.
    """
    def tokenize_function(example):
        return tokenizer(example, truncation=True, padding="max_length", max_length=512)

    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_data = dataset.map(lambda x: tokenize_function(x['text']), batched=True)
    return tokenized_data

# -------------------------------
# 3. Fine-Tune DistilGPT-2 Model
# -------------------------------
def fine_tune_distilgpt2(tokenized_data, tokenizer):
    """
    Fine-tune the DistilGPT-2 model on the Psych8k dataset.
    """
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_distilgpt2_mental_health",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=45,  # Run for 45 epochs
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("./fine_tuned_distilgpt2_mental_health")
    tokenizer.save_pretrained("./fine_tuned_distilgpt2_mental_health")

    return model, tokenizer

# -------------------------------
# 4. Generate Chatbot Response
# -------------------------------
def generate_response(user_query, model, tokenizer):
    """
    Generate chatbot responses using the fine-tuned DistilGPT-2 model.
    """
    input_ids = tokenizer.encode(user_query, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=150,
            temperature=0.7,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# -------------------------------
# 5. Streamlit Web Application
# -------------------------------
def main():
    # Title
    st.title("MindOasis")
    st.write("Welcome! I am here to provide supportive, text-based conversations.")

    # Log into Hugging Face Hub
    if not os.path.exists("./fine_tuned_distilgpt2_mental_health"):
        st.info("Fine-tuning the model, please wait...")
        login("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")  # Replace with your HF token
        texts = load_and_preprocess_data()
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        tokenized_data = tokenize_data(texts, tokenizer)
        model, tokenizer = fine_tune_distilgpt2(tokenized_data, tokenizer)
    else:
        # Load fine-tuned model
        tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_distilgpt2_mental_health")
        model = GPT2LMHeadModel.from_pretrained("./fine_tuned_distilgpt2_mental_health")

    # Conversation History
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # User Input
    user_input = st.text_input("You:", placeholder="How are you feeling today?")

    # Generate and display response
    if st.button("Send") and user_input:
        response = generate_response(user_input, model, tokenizer)

        # Update conversation
        st.session_state.conversation.append(("You", user_input))
        st.session_state.conversation.append(("Chatbot", response))

        # Display conversation
        for role, text in st.session_state.conversation:
            st.write(f"**{role}:** {text}")

    # End Conversation
    if st.button("End Conversation"):
        st.session_state.conversation = []
        st.success("Conversation cleared.")

# Run Streamlit App
if __name__ == "__main__":
    main()
pip install transformers datasets torch streamlit huggingface_hub nltk
streamlit run mental_health_chatbot.py
