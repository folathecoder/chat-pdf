# Chat-PDF: AI-Powered PDF Query System

Chat-PDF is a Python application for extracting, storing, and querying PDF content using OpenAI embeddings and MongoDB Atlas as a vector database. It allows you to parse PDF files, store content embeddings, and retrieve relevant information via similarity search.

## How to Run

1. **Clone the Repository**:
   
   ```bash
   git clone https://github.com/folathecoder/chat-pdf.git

2. **Install Dependencies:**:

   ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

3. **Run the Scripts:**:

   First run the data ingestion pipeline:
   
     ```bash
      python data_ingestion_pipeline.py

 Then run the data retrieval pipeline:
   
   ```bash
    python data_retrieval_pipeline.py
   
