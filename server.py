import os
import logging
from pathlib import Path
import shutil # For cleaning up ChromaDB

import nest_asyncio
nest_asyncio.apply()

from mcp.server.fastmcp import FastMCP

# LangChain components
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

# --- Configuration ---
DOC_DIR = "./docs_simple/"  # Folder for your Word and Excel files
CHROMA_DB_PATH = "./chroma_db_simple" # Directory to store ChromaDB
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Ollama embedding model
LLM_MODEL_NAME = "llama3"                # Ollama chat model
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more verbose LangChain/Chroma output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for LangChain components (initialized in main or reindex)
vector_store = None
llm = None
embeddings = None

# --- Helper Functions ---

def get_ollama_embeddings():
    """Initializes and returns OllamaEmbeddings."""
    logger.info(f"Initializing Ollama embeddings with model: {EMBEDDING_MODEL_NAME}")
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)

def get_ollama_llm():
    """Initializes and returns ChatOllama."""
    logger.info(f"Initializing Ollama LLM with model: {LLM_MODEL_NAME}")
    return ChatOllama(model=LLM_MODEL_NAME, base_url=OLLAMA_BASE_URL)

def load_documents_from_directory(directory_path: str):
    """Loads .docx and .xlsx documents from the specified directory."""
    docs = []
    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        logger.error(f"Document directory not found or is not a directory: {directory_path}")
        return []

    for file_path in path.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == ".docx":
                    logger.info(f"Loading Word document: {file_path.name}")
                    loader = Docx2txtLoader(str(file_path))
                    docs.extend(loader.load())
                elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                    logger.info(f"Loading Excel document: {file_path.name}")
                    # UnstructuredExcelLoader can sometimes be slow for large files
                    # mode="elements" or mode="single" can be chosen. "single" is simpler.
                    loader = UnstructuredExcelLoader(str(file_path), mode="single")
                    docs.extend(loader.load())
                # Add other loaders here if needed (e.g., PyPDFLoader for PDFs)
            except Exception as e:
                logger.error(f"Failed to load document {file_path.name}: {e}", exc_info=True)
    logger.info(f"Loaded {len(docs)} documents from {directory_path}")
    return docs

def split_documents(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(splits)} chunks.")
    return splits

# --- MCP Server Setup ---
mcp_agent = FastMCP(agent_name="DocumentQA-Ollama")

@mcp_agent.tool()
async def initialize_and_index() -> str:
    """
    Loads documents from the DOC_DIR, processes them, and creates/updates the vector store.
    This should be run first or when new documents are added.
    """
    global vector_store, embeddings

    logger.info("Starting document initialization and indexing...")
    try:
        # Ensure document directory exists
        doc_path_obj = Path(DOC_DIR)
        if not doc_path_obj.exists():
            doc_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created document directory: {DOC_DIR}")
            return f"Document directory {DOC_DIR} was created. Please add your .docx and .xlsx files and run this tool again."
        elif not any(doc_path_obj.iterdir()):
             return f"Document directory {DOC_DIR} is empty. Please add your .docx and .xlsx files and run this tool again."


        # 0. Initialize embeddings if not already done
        if embeddings is None:
            embeddings = get_ollama_embeddings()

        # 1. Load documents
        documents = load_documents_from_directory(DOC_DIR)
        if not documents:
            logger.warning("No documents loaded. Indexing cannot proceed.")
            return "No documents found or loaded. Please check DOC_DIR and logs."

        # 2. Split documents
        splits = split_documents(documents)
        if not splits:
            logger.warning("No text chunks generated after splitting. Indexing cannot proceed.")
            return "Documents were loaded but resulted in no text chunks after splitting."

        # 3. Create or update Chroma vector store
        # For simplicity, we'll delete and recreate the DB on each reindex.
        # For production, you might want an "update" strategy.
        if Path(CHROMA_DB_PATH).exists():
            logger.info(f"Removing existing ChromaDB at {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)

        logger.info(f"Creating new ChromaDB vector store at {CHROMA_DB_PATH} with {len(splits)} chunks.")
        try:
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )
            logger.info(f"Successfully created and persisted vector store with {vector_store._collection.count()} entries.")
            return f"Successfully indexed {len(splits)} document chunks. Ready to answer questions."
        except Exception as e:
            logger.error(f"Error creating Chroma vector store: {e}", exc_info=True)
            return f"Error creating vector store: {e}. Check Ollama embedding model connectivity and logs."

    except Exception as e:
        logger.error(f"Error during initialization and indexing: {e}", exc_info=True)
        return f"An error occurred during indexing: {e}"

@mcp_agent.tool()
async def ask_question(question: str) -> str:
    """
    Answers a question based on the indexed company documents.
    Make sure to run 'initialize_and_index' first.
    """
    global vector_store, llm

    if llm is None: # Initialize LLM on first use or if not already
        llm = get_ollama_llm()

    if vector_store is None:
        # Attempt to load existing ChromaDB if not in memory
        if Path(CHROMA_DB_PATH).exists() and embeddings is not None:
            logger.info(f"Loading existing ChromaDB from {CHROMA_DB_PATH}")
            try:
                vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
                entry_count = vector_store._collection.count()
                if entry_count > 0:
                    logger.info(f"Successfully loaded vector store with {entry_count} entries.")
                else:
                    logger.warning("Loaded vector store, but it's empty. Please run 'initialize_and_index'.")
                    return "Vector store is empty. Please run 'initialize_and_index' first."
            except Exception as e:
                logger.error(f"Error loading existing ChromaDB: {e}", exc_info=True)
                return f"Error loading vector store. Please run 'initialize_and_index' first. Details: {e}"
        else:
            logger.warning("Vector store not initialized and no persistent DB found or embeddings not ready. Please run 'initialize_and_index'.")
            return "Vector store not initialized. Please run 'initialize_and_index' first."

    logger.info(f"Received question: {question}")

    try:
        # 1. Retrieve relevant documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
        logger.info(f"Retrieving documents for question: {question}")
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            logger.warning("No relevant documents found for the question.")
            return "I couldn't find any relevant documents to answer your question."

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        logger.debug(f"Context for LLM (first 500 chars):\n{context[:500]}")

        # 2. Create prompt and call LLM
        prompt_template = f"""Based on the following context from company documents, please answer the question.
If the context does not contain the answer, say "I don't have enough information from the documents to answer that."

Context:
{context}

Question: {question}

Answer:"""

        logger.info("Sending prompt to LLM...")
        response = llm.invoke(prompt_template)
        answer = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        logger.info(f"LLM Answer: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Error during question answering: {e}", exc_info=True)
        return f"An error occurred while answering the question: {e}"

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Document Q&A MCP Agent Starting ---")
    logger.info(f"Document directory: {Path(DOC_DIR).resolve()}")
    logger.info(f"ChromaDB persist directory: {Path(CHROMA_DB_PATH).resolve()}")
    logger.info(f"Ollama Endpoint: {OLLAMA_BASE_URL}")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}, LLM Model: {LLM_MODEL_NAME}")
    logger.info("IMPORTANT: Ensure Ollama is running and models are downloaded:")
    logger.info(f"  ollama pull {EMBEDDING_MODEL_NAME}")
    logger.info(f"  ollama pull {LLM_MODEL_NAME}")

    # Create document directory if it doesn't exist
    Path(DOC_DIR).mkdir(parents=True, exist_ok=True)

    # Initialize Ollama components (can be deferred to first use too)
    # Doing it here to catch early errors with Ollama connection
    try:
        embeddings = get_ollama_embeddings()
        llm = get_ollama_llm()
        # Test connection by trying to embed a dummy text
        embeddings.embed_query("Test Ollama connection")
        logger.info("Successfully connected to Ollama and initialized embeddings/LLM.")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama components or connect: {e}", exc_info=True)
        logger.error("Please ensure Ollama is running and accessible at the specified OLLAMA_BASE_URL.")
        # Optionally, exit if Ollama is critical for startup
        # exit(1)

    # Note: We don't auto-index on startup. User explicitly calls 'initialize_and_index'.
    # This is often better for control.
    # If you want to auto-load an existing DB on startup:
    if Path(CHROMA_DB_PATH).exists() and embeddings is not None:
        logger.info(f"Found existing ChromaDB at {CHROMA_DB_PATH}. Loading...")
        try:
            vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            count = vector_store._collection.count()
            if count > 0:
                logger.info(f"Successfully loaded vector store with {count} entries.")
            else:
                logger.warning("Loaded vector store, but it's empty. Run 'initialize_and_index' to add documents.")
        except Exception as e:
            logger.error(f"Error loading existing ChromaDB on startup: {e}. You might need to re-index.", exc_info=True)
            # If loading fails, it's often best to clear it and require re-indexing
            # shutil.rmtree(CHROMA_DB_PATH)
            # logger.info(f"Cleared potentially corrupt ChromaDB at {CHROMA_DB_PATH}")


    logger.info("Starting MCP agent server...")
    mcp_agent.run(transport="stdio") # For MCP Inspector or mcp-cli
    logger.info("--- Document Q&A MCP Agent Stopped ---")
