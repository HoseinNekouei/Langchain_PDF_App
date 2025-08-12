import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate


@st.cache_resource
def load_embeddings():
    """
    Load Google Generative AI embeddings with proper event loop handling.
    """
    import asyncio
    api_key = st.secrets.get('GOOGLE_API_KEY')
    if not api_key:
        st.error("Google API key not found in Streamlit secrets.")
        st.stop()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )


def extract_pdf_text(pdf_file):
    """Extract text from PDF pages, skipping empty pages."""
    reader = PdfReader(pdf_file)
    text = ""
    error_page = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text += page_text
        else:
            error_page.append(i)

    if error_page:
        st.warning(f"⚠ Page {error_page} has no extractable text (might be scanned).")

    return text


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":books:", layout="centered")
    st.header("Ask your PDF anything")

    # Upload file
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if not pdf:
        st.info("📄 Please upload a PDF to get started.")
        return

    # Extract text
    texts = extract_pdf_text(pdf)
    if not texts.strip():
        st.error("No extractable text found in the PDF. It may be image-based or empty.")
        st.stop()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(texts)

    if not chunks or not any(c.strip() for c in chunks):
        st.error("No valid text chunks found — the PDF may be empty or scanned.")
        st.stop()

    # Load embeddings
    embeddings = load_embeddings()

    # Create vector store
    try:
        vector_store = FAISS.from_texts(chunks, embeddings)
    except IndexError:
        st.error("Embedding creation failed. Check your API key or embedding model.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while creating FAISS index: {e}")
        st.stop()

    # Get query
    query = st.text_input('Ask a question about your PDF:', key='question')
    if not query:
        return

    # Search similar docs
    try:
        docs = vector_store.similarity_search(query, k=3)
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        st.stop()

    if not docs:
        st.warning("No relevant content found for your query.")
        return

    # Create LLM and prompt
    llm = GoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the context below to answer the question:\n\n{context}\n\nQuestion: {question}"
    )

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template,
        document_variable_name="context",
    )

    # Get result
    try:
        result = chain.invoke({"context": docs, "question": query})
    except Exception as e:
        st.error(f"Error during LLM processing: {e}")
        st.stop()

    st.write("### Response:")
    st.write(result["output"] if isinstance(result, dict) and "output" in result else result)


if __name__ == '__main__':
    main()