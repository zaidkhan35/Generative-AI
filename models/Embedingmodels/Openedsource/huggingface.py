# Import tool to create embeddings using a free Hugging Face model
from langchain_huggingface import HuggingFaceEmbeddings

# Set up the embedding model (runs locally, no API key needed)
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# List of sentences we want to convert into vectors
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Convert all sentences into embeddings (each is a 384-number vector)
vector = embedding.embed_documents(documents)

# Print the list of vectors
print(str(vector))

# Example of what output looks like (only first 5 numbers shown per sentence,
# real output has 384 numbers per sentence):
# [
#   [-0.023, 0.041, 0.087, -0.012, 0.056, ... ],   # vector for "Delhi..."
#   [-0.031, 0.028, 0.079, -0.019, 0.048, ... ],   # vector for "Kolkata..."
#   [ 0.015, -0.033, 0.062, 0.021, -0.044, ... ]   # vector for "Paris..."
# ]
