from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embeddings=embedding, allow_dangerous_deserialization=True)

query = input("🔍 Enter your query: ")
results = db.similarity_search(query, k=3)

print("\n📄 Top matching chunks:")
for i, doc in enumerate(results, 1):
    print(f"\n--- Chunk {i} ---\n{doc.page_content}")
