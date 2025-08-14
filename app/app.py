import streamlit as st
from app.rag_engine import SimpleRAG

# Initialize the RAG engine
rag = SimpleRAG("data/docs.txt")

# Page title
st.title("RAG Application")

# Form Inputs
with st.form("rag_form"):
    query = st.text_input("Enter your query:")
    input_text = st.text_area("Enter your input text:")

    submit_button = st.form_submit_button("Submit")

# Handle form submission
if submit_button:
    if not query or not input_text:
        st.warning("Please enter both query and input text.")
    else:
        with st.spinner("Processing..."):
            retrieved = rag.retrieve(query)
            response = rag.generate(input_text, retrieved)
        
        st.subheader("Response:")
        st.write(response)

        st.subheader("Retrieved Documents:")
        st.write(retrieved)
