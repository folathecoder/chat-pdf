import time  # For measuring execution time
import os  # To interact with environment variables and file paths
from dotenv import load_dotenv  # To load environment variables from a .env file
from pymongo.mongo_client import MongoClient  # To connect and interact with MongoDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # For embeddings and chat-based AI models
from langchain_mongodb import MongoDBAtlasVectorSearch  # To interact with MongoDB for vector-based searches
from langchain.prompts import PromptTemplate  # For creating customizable prompts
from langchain_core.runnables import RunnablePassthrough, RunnableMap  # For creating a chain of tasks
from langchain_core.output_parsers import StrOutputParser  # For parsing string outputs from AI responses

# Start the timer to measure execution time
start_time = time.time()

# Load environment variables from a .env file
load_dotenv()

# Retrieve MongoDB connection string and OpenAI API key from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize MongoDB client and configure database and collection details
client = MongoClient(MONGODB_URI)
dbName = "rag"  # TODO: Change this to your database name
collectionName = "staff_handbook"  # TODO: Change this to your collection name
collection = client[dbName][collectionName]
index = "staff_handbook_index"  # TODO: Change this to your index name

# Define the user prompt/question
user_prompt = "What is Retrieval Augmented Generation?"

# Initialize the vector search object using MongoDB and OpenAI embeddings
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGODB_URI,
    dbName + "." + collectionName,
    OpenAIEmbeddings(disallowed_special=(), openai_api_key=OPENAI_API_KEY),
    index_name=index,
)

# Configure the retriever for similarity-based searches in the vector database
retriever = vector_search.as_retriever(
    search_type="similarity",  # Use similarity to find closely related results
    search_kwargs={"k": 3},  # Retrieve the top 3 most relevant documents
)

# Function to generate a prompt template for LLMs
def prompt_generator(context: str, question: str) -> PromptTemplate:
    instruction = (
        f"You are an AI assistant tasked with answering questions accurately and only "
        f"using the provided context. Avoid making assumptions or including unrelated information. "
        f"Here is the context:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n"
        f"1. Base your answer solely on the provided context and you can further explain in details in your own words without diverting from the original context.\n"
        f"2. If the question cannot be answered using the context, respond with: "
        f"'I cannot answer this question based on the provided context.'\n\n"
        f"Answer:"
    )
    # Return a PromptTemplate object created from the instruction text
    return PromptTemplate.from_template(instruction)

# Function to retrieve relevant data from the vector database
def retrieve_from_vector_store(query: str) -> str:
    try:
        # Retrieve the most relevant documents from the vector store
        documents = retriever.invoke(query)
        # Combine the content of all retrieved documents
        joined_page_content = "\n\n".join(
            doc.page_content for doc in documents if hasattr(doc, "page_content")
        )
        # Return the combined content or a default message if no content is found
        return joined_page_content or "No relevant content found."
    except Exception as e:
        # Handle errors during data retrieval
        print(f"Error retrieving data from vector store: {e}")

# Function to generate a response from the AI model based on the prompt and context
def response_generation(prompt: PromptTemplate, context: str, question: str) -> str:
    try:
        # Initialize ChatOpenAI with a low temperature for deterministic responses
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
        # Define an output parser for string-based outputs
        response_parser = StrOutputParser()

        # Create a chain of tasks to process the input, generate a response, and parse the output
        rag_chain = (
            RunnableMap(
                {
                    "context": RunnablePassthrough(),  # Pass the context as-is
                    "question": RunnablePassthrough(),  # Pass the question as-is
                }
            )
            | prompt  # Apply the prompt template
            | llm  # Generate a response using the AI model
            | response_parser  # Parse the AI model's response into a string
        )

        # Execute the chain with the provided context and question
        return rag_chain.invoke({"context": context, "question": question})
    except Exception as e:
        # Handle errors during response generation
        print(f"Error generating response for LLM: {e}")

# Main function to run the entire retrieval and response pipeline
def run(query: str):
    try:
        # Step 1: Retrieve relevant context from the vector store
        context = retrieve_from_vector_store(query)

        # Step 2: Generate a prompt template based on the retrieved context and query
        prompt = prompt_generator(context, question=query)

        # Step 3: Generate a response from the AI model using the prompt
        response = response_generation(prompt, context=context, question=query)

        # Print the generated response
        print(response)
    except Exception as e:
        # Handle errors during pipeline execution
        print(f"Error running retrieval pipeline: {e}")

# Run the pipeline with the user's prompt
run(user_prompt)

# End the timer to measure execution time
end_time = time.time()

# Calculate and print the total elapsed time for execution
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")
