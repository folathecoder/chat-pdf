import time  # To measure execution time
import os  # To interact with the operating system (e.g., environment variables)
from dotenv import load_dotenv  # To load environment variables from a .env file
from langchain_community.document_loaders import PyPDFLoader  # For loading and parsing PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
from langchain_openai import OpenAIEmbeddings  # For generating embeddings using OpenAI models
from langchain_mongodb import MongoDBAtlasVectorSearch  # To interact with MongoDB for storing embeddings
from pymongo.mongo_client import MongoClient  # To connect and interact with MongoDB

# Start the timer to measure execution time
start_time = time.time()

# Specify the file path of the PDF document to be processed
file_path = "../documents/document-1.pdf"  # TODO: Change this to the desired file path

# Load environment variables from the .env file
load_dotenv()

# Retrieve MongoDB connection string and OpenAI API key from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the MongoDB client and define the database and collection to use
client = MongoClient(MONGODB_URI)
dbName = "rag"  # TODO: Change this to the desired database name
collectionName = "staff_handbook"  # TODO: Change this to the desired collection name
collection = client[dbName][collectionName]

# Function to load a PDF document
def load_document(file: str):
    try:
        # Use PyPDFLoader to load the PDF and return its content
        loader = PyPDFLoader(file)
        documents = loader.load()
        return documents
    except Exception as e:
        # Handle errors during PDF loading
        print(f"Error loading PDF file: {e}")

# Function to split the document into smaller chunks
def split_document(document):
    try:
        # Use RecursiveCharacterTextSplitter to create chunks with overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Maximum size of each chunk
            chunk_overlap=200  # Overlap between chunks to preserve context
        )
        return text_splitter.split_documents(document)
    except Exception as e:
        # Handle errors during document splitting
        print(f"Error splitting document: {e}")

# Function to store document chunks in the vector database
def store_in_vector_store(splits):
    try:
        # Initialize OpenAI embeddings using the API key
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        # Store the document chunks with their embeddings in MongoDB
        MongoDBAtlasVectorSearch.from_documents(
            splits, embedding, collection=collection
        )
    except Exception as e:
        # Handle errors during vector storage
        print(f"Error storing data in vector store: {e}")

# Main function to run the entire pipeline
def run(file: str):
    try:
        # Step 1: Load and clean the document
        document = load_document(file)
        print(document)

        # Step 2: Split the document into smaller chunks
        splits = split_document(document)

        # Step 3: Generate embeddings and store the chunks in MongoDB
        store_in_vector_store(splits)
    except Exception as e:
        # Handle errors during pipeline execution
        print(f"Error running data pipeline: {e}")

# Run the pipeline with the specified file
run(file_path)

# End the timer to measure execution time
end_time = time.time()

# Calculate and print the total elapsed time for execution
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")
