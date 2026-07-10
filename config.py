# AWS Configuration
AWS_REGION = "us-east-1"

# S3 Configuration
S3_BUCKET = "edi-documents-ajam-2026"
S3_KEY = "Kaveri_English_Text_Book_Class_9.pdf"

# Bedrock Models
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "amazon.nova-lite-v1:0"

# Local Storage
LOCAL_PDF_PATH = "data/textbook.pdf"

FAISS_INDEX_PATH = "vectorstore/faiss_index.bin"
CHUNKS_PATH = "vectorstore/chunks.json"

# Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval Settings
TOP_K = 5
