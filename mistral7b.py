import os
import time
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
max_context_words = 4048  # Approx. token cap
# Load Mistral 7B model using llama.cpp
llm = LlamaCpp(
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=max_context_words,  # Limited context
    temperature=0.7,
    top_p=0.95,
    n_threads=12,
    n_gpu_layers=0,
    verbose=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt template
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
# Build the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

# ──────────────────────────────────────────────────────────────────────────────
# Initialize rolling conversation memory
conversation_history = []
# max_context_words = 4048  # Approx. token cap

# ──────────────────────────────────────────────────────────────────────────────
# Interactive loop
while True:
    query = input("🧠 Ask your question (type 'exit' to quit): ")
    if query.strip().lower() in ["exit", "quit"]:
        print("👋 Exiting.")
        break

    conversation_history.append(f"User: {query}")

    # Search relevant documents
    start = time.time()
    docs_with_scores = merged_db.similarity_search_with_score(query, k=10, score_threshold=1.0)
    # Combine last 5 conversation turns (User and Assistant)
    recent_history = "\n".join(conversation_history[-5:])
    # search_query = f"{recent_history}\nUser: {query}"

    # Semantic similarity threshold
    # similarity_threshold = 0.6

    # if len(conversation_history) >= 2:
    # previous_turns = "\n".join(conversation_history[-2:])
    # if is_short_or_generic(query):
    #     treat_as_follow_up = True
    #     print("🧠 Treating as follow-up due to vague/short query.")
    # else:
    #     similarity_score = compute_similarity(query, previous_turns, embedding)
    #     treat_as_follow_up = similarity_score > similarity_threshold
    #         print(f"🔍 Similarity Score: {similarity_score:.2f}")
    # else:
    #     treat_as_follow_up = False

    # # Prepare the search query
    # if treat_as_follow_up:
    #     recent_history = "\n".join(conversation_history[-5:])
    #     search_query = f"{recent_history}\nUser: {query}"
    # else:
    #     search_query = query


    # Perform similarity search with combined history
    # docs_with_scores = merged_db.similarity_search_with_score(search_query, k=10, score_threshold=1.0)

    end = time.time()

    filtered_docs = []
    for doc, score in docs_with_scores:
        print(f"📈 Score: {score:.4f} for document: {doc.metadata.get('source', 'unknown')}")
        filtered_docs.append(doc)

    # if not filtered_docs:
    #     print("⚠️ No relevant documents found.")
    #     continue

    # Build recent conversation summary (last 5 turns)
    summary = "\n".join(conversation_history[-5:])

    # Merge documents into one block
    doc_context = "\n\n".join(doc.page_content for doc in filtered_docs)

    # Merge summary + doc content
    total_context = f"{summary}\n\n{doc_context}"
    print("\n📝 Context:\n", total_context[:500], "...")  # Print first 500 chars of context
    # Truncate context if too long
    words = total_context.split()
    if len(words) > max_context_words:
        total_context = " ".join(words[:max_context_words])

    # Run the LLM chain
    start = time.time()
    # result = stuff_chain.invoke({
    #     "input_documents": filtered_docs,
    #     "context": total_context,
    #     "question": query
    # })
    # Run directly through LLMChain
    result = llm_chain.invoke({
        "context": total_context,
        "question": query
    })
    end = time.time()

    # response = result["output_text"]
    response = result["text"]

    print("\n🤖 Mistral Response:\n", response)

    # Append assistant's response to history
    conversation_history.append(f"Assistant: {response}")

    print(f"\n⏱️ Inference Time: {end - start:.2f} seconds")
    print("\n📄 Retrieved Source Documents:")
    for i, doc in enumerate(filtered_docs, 1):
        print(f"\n--- Document {i} ---\n{doc.page_content[:500]}...")
