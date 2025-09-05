import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from rag_chatbot_backend_business_logic.rag_chatbot_business_logic import (
    load_pdf,
    split_documents,
    get_embeddings_model,
    create_vectorstore,
    create_qa_chain,
    run_qa_chain,
)

# Load environment variables (for OpenAI API key)
load_dotenv()

st.title("RAG Chatbot: PDF QA with OpenAI")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Step 2: Load and process PDF
        try:
            data = load_pdf(tmpdir, uploaded_file.name)
            st.write(f"Loaded {len(data)} document(s) from PDF.")
            chunks = split_documents(data)
            st.write(f"Split into {len(chunks)} chunks.")
            embeddings_model = get_embeddings_model()
            vectorstore = create_vectorstore(chunks, embeddings_model)
            st.write(f"Vectorstore created with {vectorstore.index.ntotal} vectors.")
            qa_chain = create_qa_chain(vectorstore)

            # Step 3: QA interface
            st.subheader("Ask a question about your PDF:")
            user_query = st.text_input("Enter your question")
            if user_query:
                with st.spinner("Retrieving answer..."):
                    response = run_qa_chain(qa_chain, user_query)
                st.markdown("**Answer:**")
                st.write(response["result"])
                st.markdown("**Source Documents:**")
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"Document {i+1}:")
                    st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a PDF file to begin.")
