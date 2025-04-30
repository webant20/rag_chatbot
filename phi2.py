from langchain_community.llms import LlamaCpp
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")


# Load the Phi-2 model with llama.cpp
llm = LlamaCpp(
    # model_path="models/phi-2.Q8_0.gguf",
    model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.7,
    top_p=0.95,
    n_gpu_layers=20,
    verbose=True,
)

# Print out the available properties of the LlamaCpp instance
print("LlamaCpp properties: ", dir(llm))

# User query
query = input("🧠 Ask your question: ")

# Direct model invocation
response = llm.invoke(query)

# Output
print("\n🧠 Model Response:\n", response)
