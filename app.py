import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import streamlit as st
from PyPDF2 import PdfReader


app= Flask(__name__)

def main():
    load_dotenv()

    st.set_page_config(
        page_title="Chat with PDF",
        page_icon= ":books:",
        layout="centered"
    )

    st.title("Chat with PDF")
    st.header("Ask you PDFs anything you want to know about them")

    # upload file
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        texts=""
    
        for page in pdf_reader.pages:
            texts += page.extract_text()

        st.session_state["pdf_text"] = texts

        st.write("PDF text extracted successfully!")



if __name__== '__main__':
    main()

