import os
import time
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

# ──────────────────────────────────────────────────────────────────────────────
# Set environment to use only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ──────────────────────────────────────────────────────────────────────────────
# Initialize embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ──────────────────────────────────────────────────────────────────────────────
# Load and merge FAISS indexes
vectorstore_dir = "vectorstore"
merged_db = None

start_load_time = time.time()

for foldername in os.listdir(vectorstore_dir):
    folderpath = os.path.join(vectorstore_dir, foldername)
    if os.path.isdir(folderpath):
        print(f"📦 Loading vectorstore: {foldername}")
        db = FAISS.load_local(folderpath, embedding, allow_dangerous_deserialization=True)
        if merged_db is None:
            merged_db = db
        else:
            merged_db.merge_from(db)

end_load_time = time.time()
print(f"\n✅ Vectorstores loaded and merged in {end_load_time - start_load_time:.2f} seconds.")

# ──────────────────────────────────────────────────────────────────────────────
# Load Mistral 7B model using llama.cpp
llm = LlamaCpp(
    # model_path="models/phi-4-q4.gguf",
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=5012,
    temperature=0.7,
    top_p=0.95,
    n_threads=12,
    n_gpu_layers=0,
    verbose=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template for document + question
template = """[INST]
You are a helpful assistant. Use the following context to answer the question elaborately with bullet points.
If the answer is not in the context, just say "I don't know".

Context:
{context}

Question:
{question}
[/INST]"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# ──────────────────────────────────────────────────────────────────────────────
# Create StuffDocumentsChain manually
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

# ──────────────────────────────────────────────────────────────────────────────
# Interactive question-answer loop
while True:
    query = input("🧠 Ask your question (type 'exit' to quit): ")
    if query.strip().lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    # Step 1: Search manually using retriever
    # start = time.time()
    # retriever = merged_db.as_retriever(search_kwargs={"k": 1})
    # search_results = retriever.get_relevant_documents(query)
    # end = time.time()
    # print(f"⏱️ Retrieval Time: {end - start:.2f} seconds")

    # Step 2: Get similarity scores separately
    start = time.time()
    docs_with_scores = merged_db.similarity_search_with_score(query, k=2)
    end = time.time()
    # Step 3: Filter documents based on score
    filtered_docs = []
    # score_threshold = 0.0  # Accept only if similarity > 0.7
    for doc, score in docs_with_scores:
        print(f"📈 Score: {score:.4f} for document: {doc.metadata.get('source', 'unknown')}")
        filtered_docs.append(doc)
        # if score >= score_threshold:
        #     filtered_docs.append(doc)
        # else:
        #     print(f"⚠️ Discarded document with score {score:.4f}")

    # Step 4: Run the chain manually
    if not filtered_docs:
        print("\n⚠️ No documents passed the score threshold.")
        continue

    start = time.time()
    result = stuff_chain.invoke({
    "input_documents": filtered_docs,# # Use the filtered documents directly
    "question": query
    })
    # result = stuff_chain.invoke({
    #     "context": "\n\n".join(doc.page_content for doc in filtered_docs),
    #     "question": query
    # })
    end = time.time()

    print("\n🤖 Mistral Response:\n", result["output_text"])  # Corrected to 'text' key

    print(f"\n⏱️ Inference Time: {end - start:.2f} seconds")

    print("\n📄 Retrieved Source Documents:")
    for i, doc in enumerate(filtered_docs, 1):
        print(f"\n--- Document {i} ---\n{doc.page_content[:500]}...")

