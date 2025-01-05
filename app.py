from langchain_openai import OpenAIEmbeddings
import streamlit as st
import numpy as np
import json
import faiss
from openai import OpenAI

# Page setup
st.set_page_config(page_title="FaceBase Chatbot", page_icon="FaceBase.png")

# Layout
col1, col2 = st.columns([1, 2])
with col1:
    st.write("")
    st.write("")  
    st.image("FaceBase.png", use_column_width=True)
with col2:
    st.write("") 
    st.title("FaceBase Chatbot")

# User input
user_input = st.text_input('Enter your prompt:')

try:
    # Load the dataset
    with open('isa_dataset (1).json', 'r') as file:
        data = json.load(file)

    def preprocess_descriptions(data):
        descriptions = [item['description'] for item in data]
        cleaned_descriptions = []
        for desc in descriptions:
            if desc:
                cleaned_desc = "".join(ch for ch in desc if ch.isprintable())
                cleaned_descriptions.append(cleaned_desc)
        return cleaned_descriptions

    def prepare_model_input(descriptions, question):
        input_text = " ".join(descriptions) + " [SEP] " + question
        return input_text

    def generate_response(input_text):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def postprocess_response(response):
        response = response.strip()
        max_length = 2000
        return response[:max_length]

    # Initialize OpenAI and embeddings with secrets
    openai_api_key = st.secrets["openai"]["api_key"]
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    client = OpenAI(api_key=openai_api_key)

    # Process descriptions and create embeddings
    descriptions = preprocess_descriptions(data)
    text_embeddings = embeddings_model.embed_documents(descriptions)
    text_embeddings = np.array(text_embeddings)
    
    # Initialize FAISS index
    dimension = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(text_embeddings)

    # Handle user input
    if st.button('send'):
        if user_input:
            question_embedding = embeddings_model.embed_documents([user_input])
            question_embedding = np.array(question_embedding)
            k = 3
            distances, indices = index.search(question_embedding, k)
            relevant_descriptions = [data[i]['description'] for i in indices[0]]
            
            model_input = prepare_model_input(relevant_descriptions, user_input)
            generated_response = generate_response(model_input)
            postprocessed_response = postprocess_response(generated_response)
            
            st.write("**Generated Response:**")
            st.write(postprocessed_response)
            
            st.write("**More From FaceBase:**")
            for description in relevant_descriptions:
                st.write(description)
        else:
            st.warning("Please enter a prompt before sending.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
