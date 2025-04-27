from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Load embedding model and vectorstore
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings=embedding, allow_dangerous_deserialization=True)

# Load the Phi-2 model with llama.cpp
llm = LlamaCpp(
    model_path="models/phi-2.Q8_0.gguf",
    n_ctx=2048,
    temperature=0.7,
    top_p=0.95,
    n_gpu_layers=20,
    verbose=True,
)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# User query
query = input("🧠 Ask your question: ")

# Use RetrievalQA chain
response = qa_chain.invoke({ "query": query })

# Output
print("\n🧠 Answer:\n", response["result"])

print("\n📄 Source Chunks:")
for i, doc in enumerate(response["source_documents"], 1):
    print(f"\n--- Chunk {i} ---\n{doc.page_content}")
