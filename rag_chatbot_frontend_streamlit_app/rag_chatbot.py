import sys
import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
            user_query = st.text_input(
                "Ask a question about your PDF:",
                label_visibility="collapsed",
                value="Enter your question",
            )
            if "response" not in st.session_state:
                st.session_state["response"] = None
            if st.button("Submit Question"):
                with st.spinner("Retrieving answer..."):
                    response = run_qa_chain(qa_chain, user_query)
                st.session_state["response"] = response
                st.session_state["show_sources"] = False
            if st.session_state["response"]:
                st.subheader("Answer:")
                st.text_area(
                    "Answer:",
                    st.session_state["response"]["result"],
                    label_visibility="collapsed",
                    height=100,
                    disabled=False,
                    key="answer_area",
                )
                if "show_sources" not in st.session_state:
                    st.session_state["show_sources"] = False
                if st.button("Show/Hide Source Document Chunks"):
                    st.session_state["show_sources"] = not st.session_state[
                        "show_sources"
                    ]
                if st.session_state["show_sources"]:
                    # st.markdown("**Source Document Chunks:**")
                    for i, doc in enumerate(
                        st.session_state["response"]["source_documents"]
                    ):
                        st.write(f"**Document Chunk {i+1}:**")
                        st.write(doc.page_content)
        except Exception as e:
            st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a PDF file to begin.")
