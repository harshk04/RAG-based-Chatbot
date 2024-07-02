# RAG-based-Chatbot

## Introduction

It is a powerful tool that leverages Retrieval-Augmented Generation (RAG) to provide contextual responses based on PDF documents. By utilizing LLAMA3 (locally) as the language model and Qdrant as the vector database, this application ensures efficient and accurate semantic search and response generation. The embeddings are generated using the BAAI/bge-large-en model from HuggingFace, allowing for nuanced and context-aware interactions with the ingested documents.

<img width="800" alt="Screenshot 2024-06-24 at 3 57 18 PM" src="https://github.com/harshk04/BookFinder-LLM-Application/assets/115946158/e99d9a40-d1b6-40f1-93b0-442341ec6854">
<img width="800" alt="Screenshot 2024-06-24 at 3 57 30 PM" src="https://github.com/harshk04/BookFinder-LLM-Application/assets/115946158/e69997ed-01cc-426a-a5df-d6dcb501e410">
<img width="800" alt="Screenshot 2024-06-24 at 8 26 01 PM" src="https://github.com/harshk04/BookFinder-LLM-Application/assets/115946158/a743bebf-9040-4ff4-b4fe-4d158c640fbd">
<img width="800" alt="Screenshot 2024-06-24 at 3 57 40 PM" src="https://github.com/harshk04/BookFinder-LLM-Application/assets/115946158/1b2a99f2-3151-478e-874e-09892f59ee98">


## Features

- **PDF Document Ingestion:** Load and split PDF documents into chunks for semantic search.
- **Embedding Generation:** Use HuggingFace's BAAI/bge-large-en model for generating embeddings.
- **Contextual Search:** Perform semantic searches to retrieve relevant document sections.
- **Response Generation:** Use LLAMA3 to generate responses based on the retrieved context.
- **Streamlit Interface:** User-friendly interface with navigation options for Home, Generate Response, and Contact Us pages.

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Qdrant
- Ollama:LLAMA3
- Streamlit

## Installation

1. Clone the repository:

   `git clone https://github.com/harshk04/llama-gpt-contextual-search.git
   
   cd llama-gpt-contextual-search`

3. Install the required packages:
   
    `pip install -r requirements.txt`

4. Ensure Qdrant is running locally:
   
    `docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`

## Usage
### Ingesting Data
1. Place your PDF document in the project directory and load it in `qdrant-ingest.py`.
2. Run the `qdrant-ingest.py` script to load the PDF, split it into chunks, and store the embeddings in Qdrant:
   
`python ingest.py`

### Running the Streamlit App
1. Run the Streamlit app:
   `streamlit run app.py`
2. Open your browser and go to `http://localhost:8501` to access the application.

## Project Structure

- **qdrant-ingest.py**: Script to load PDF, split text, generate embeddings, and store them in Qdrant.
- **app.py**: Main application script with Streamlit interface for interacting with the search engine.
- **requirements.txt**: List of required Python packages.


## App Navigation

- **Home**: Overview of the application with an image and welcome message.
- **Generate Response**: Interactive chat interface to generate responses based on user input and document context.
- **Contact Us**: Form to contact the developer.


## 📬 Contact

If you want to contact me, you can reach me through below handles.

&nbsp;&nbsp;<a href="https://www.linkedin.com/in/harsh-kumawat-069bb324b/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="30"></img></a>

© 2024 Harsh
