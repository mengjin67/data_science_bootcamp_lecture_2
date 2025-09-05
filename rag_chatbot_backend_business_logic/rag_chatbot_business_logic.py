# %%
# Step 1: Load document
def load_pdf(folder, filename):
    import os
    from langchain_community.document_loaders import PyPDFLoader
    uploads_file_path = os.path.join(folder, filename)
    loader = PyPDFLoader(uploads_file_path)
    return loader.load()

# Step 2: Split document
def split_documents(data, chunk_size=500, chunk_overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(data)

# Step 3: Create embeddings model
def get_embeddings_model(model_name="text-embedding-3-small", dimensions=1536, organization=None):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name, dimensions=dimensions, organization=organization)

# Step 4: Create vectorstore
def create_vectorstore(chunks, embeddings_model):
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(chunks, embeddings_model)

# Step 5: Create QA chain
def create_qa_chain(vectorstore, llm_model_name="gpt-4o"):
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    model = ChatOpenAI(model_name=llm_model_name)
    template = """Use the following pieces of context to answer the question at the end.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\nUse three sentences maximum and keep the answer as concise as possible.\n\n{context}\nQuestion: {question}\nHelpful Answer:"""
    prompt = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Step 6: Run QA chain
def run_qa_chain(qa_chain, query):
    return qa_chain.invoke({"query": query})

# %%
# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    import numpy as np
    load_dotenv()
    data = load_pdf("./.local/uploads", "cover_letter_jin_meng.pdf")
    print(f"Loaded {len(data)} documents.")
    chunks = split_documents(data)
    print(f"Split into {len(chunks)} chunks.")
    embeddings_model = get_embeddings_model()
    vectorstore = create_vectorstore(chunks, embeddings_model)
    print(f"Vectorstore created with {vectorstore.index.ntotal} vectors.")
    qa_chain = create_qa_chain(vectorstore)
    query = "Help me conclude the personality of Jin Meng"
    response = run_qa_chain(qa_chain, query)
    print("Answer:", response["result"])
    print("Source documents:", response["source_documents"])

# %%
