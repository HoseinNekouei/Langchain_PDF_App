import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate


@st.cache_resource
def load_embeddings():
    """"
    grpc.aio (which GoogleGenerativeAIEmbeddings uses internally) needs an active asyncio event loop, 
    but Streamlit runs your script in a thread without one by default.
    """
    api_key = st.secrets['GOOGLE_API_KEY']

    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= api_key)


def main():
    st.set_page_config(
        page_title="Chat with PDF",
        page_icon= ":books:",
        layout="centered"
    )

    # st.title("Chat with PDF")
    st.header("Ask your PDF anything")

    # upload file
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        texts=""
        for page in pdf_reader.pages:
        
            texts += page.extract_text()
            
        text_spliter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks= text_spliter.split_text(texts)

        # create embeddings
        embeddings = load_embeddings()

        vector_store= FAISS.from_texts(chunks, embeddings)

        # show user input
        query= st.text_input('Ask a question about your PDF:', key='question')            
        if query:
            docs = vector_store.similarity_search(query, k=3)

            # Create LLM
            llm = GoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.2)

            # Create a prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="Use the context below to answer the question:\n\n{context}\n\nQuestion: {question}"
            )

            chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt_template,
                document_variable_name="context",
            )
            # Invoke the chain with the documents and query
            result = chain.invoke({"context": docs, "question": query})

            st.write("Response:")
            st.write(result["output"] if isinstance(result, dict) and "output" in result else result)


if __name__== '__main__':
    main()

