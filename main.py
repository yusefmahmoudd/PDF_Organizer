from sentence_transformers import SentenceTransformer

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# store abstracts in a list
abstract_list = ["abstract 1", "abstract 2"]

# calculate embeddings
embeddings = model.encode(abstract_list)