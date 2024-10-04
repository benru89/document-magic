import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI  # Use ChatOpenAI for chat models
from langchain import LLMChain, PromptTemplate
import fitz  # PyMuPDF

# Load the API key from .env file
api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Use ChatOpenAI instead of OpenAI for chat models like gpt-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

def summarize_document(text, length, style, custom_instructions):
    # Create a dynamic prompt based on user inputs
    prompt = f"""
    Please read the following document carefully and summarize it in {length} sentences. 
    The summary should be {style}. 
    Here is the document: {text}. 
    {custom_instructions if custom_instructions else ''}
    """
    
    prompt_template = PromptTemplate.from_template(prompt)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run({"document_text": text})

# Extract text from PDF
def extract_text_from_pdf(uploaded_pdf):
    pdf_document = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Streamlit app layout
st.title("Customizable Document Summarizer")

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

# Customization options
length = st.slider("Select summary length (number of sentences)", min_value=1, max_value=30, value=10)
style = st.selectbox("Select summary style", ["concise", "detailed", "formal", "informal", "bullet points"])
custom_instructions = st.text_input("Additional instructions (optional)", "")

# Process file if uploaded
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    else:
        document_text = uploaded_file.read().decode("utf-8")
    
    # Show only the first 5 lines by default
    preview_lines = document_text.splitlines()[:5]
    preview_text = "\n".join(preview_lines)
    
    # Display a preview of the document (first 5 lines only)
    st.write("Document Preview (First 5 lines):")
    st.write(preview_text)

    # Collapsible section to expand and see the full document (optional)
    with st.expander("Show full document content", expanded=False):
        st.write(document_text)

    # Summarize button and the result display
    if st.button("Summarize"):
        # Summarize the document but don't display the full document
        summary = summarize_document(document_text, length, style, custom_instructions)
        st.write("Summary:")
        st.write(summary)
    
