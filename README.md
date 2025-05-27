# local_RAG_mcp
This is my attempt at creating a local RAG mcp server to chat with local .docx, .xlsx files.

# Local Document Q&A with Ollama & MCP

This project provides a Python-based MCP agent that allows you to chat with your local Word (.docx) and Excel (.xlsx) documents using Ollama for local language model inference, ensuring your private data stays on your machine.

It uses LangChain community components for document loading, text splitting, embeddings, and vector storage (ChromaDB), and `mcp-sdk` to expose this functionality as a set of tools.

**Features:**
*   **Private & Local:** All processing, embedding, and language model inference happens locally via Ollama. No data leaves your machine.
*   **Supported Document Types:** Currently supports Microsoft Word (.docx) and Excel (.xlsx) files. (Easily extensible for PDFs, .txt, etc.)
*   **Simple Indexing:** A dedicated MCP tool to scan a specified directory, process documents, and build a searchable vector index.
*   **Natural Language Q&A:** Ask questions in natural language about the content of your documents.
*   **MCP Integration:** Exposes functionality through MCP tools, usable with [MCP Inspector](https://github.com/mcp-ai/mcp-inspector) or `mcp-cli`.

## How it Works

1.  **Document Loading:** The agent scans a designated local folder for supported documents.
2.  **Text Chunking:** Document content is split into smaller, manageable chunks.
3.  **Embedding:** Each chunk is converted into a numerical representation (embedding) using a local Ollama embedding model (e.g., `nomic-embed-text`).
4.  **Vector Storage:** These embeddings and their corresponding text chunks are stored in a local ChromaDB vector store.
5.  **Querying (RAG - Retrieval Augmented Generation):**
    *   When you ask a question, it's also embedded.
    *   The system searches the vector store for document chunks with embeddings most similar to your question's embedding.
    *   These relevant chunks (context) are combined with your original question into a prompt.
    *   This prompt is sent to a local Ollama chat model (e.g., `llama3`) to generate an answer.

## Prerequisites

1.  **Python:** Python 3.9+
2.  **Ollama:** You need Ollama installed and running.
    *   Installation: [ollama.com](https://ollama.com/)
    *   Ensure Ollama is serving models. You can test this by running `ollama list` in your terminal.
3.  **Required Ollama Models:**
    *   Pull the embedding model: `ollama pull nomic-embed-text`
    *   Pull the chat model: `ollama pull llama3`

## Installation

1.  **Clone the Repository (or download the script):**
    ```bash
    # If you create a Git repository:
    # git clone https://github.com/ItsMistahJi/local_RAG_mcp
    # cd local_RAG_mcp
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Setup & Usage

1.  **Prepare Your Documents:**
    *   Create a folder named `docs_simple` in the same directory as the `server.py` script (or configure `DOC_DIR` in the script).
    *   Place your `.docx` and `.xlsx` files into this `docs_simple` folder.

2.  **Run the MCP Agent Script:**
    Open your terminal, navigate to the project directory, and run:
    ```bash
    python server.py
    ```
    The script will start, attempt to connect to Ollama, and then wait for MCP client connections (like MCP Inspector). You'll see log output in this terminal.

3.  **Interact using MCP Inspector (Recommended):**
    *   Download and install [MCP Inspector](https://github.com/mcp-ai/mcp-inspector/releases).
    *   Open MCP Inspector.
    *   **Configure the Agent:**
        *   If the agent isn't automatically detected, you may need to manually add it.
        *   Go to "File" > "Preferences" > "Agents" (or a similar section).
        *   Click "Add" or the "+" icon.
        *   **Name:** Give it a descriptive name (e.g., "My Local RAG Agent").
        *   **Command:** Enter the full command to run your script: `python /full/path/to/your/server.py`.
        *   **Transport:** Select `stdio`.
        *   Save the configuration.
    *   **Connect and Use:**
        *   Back in the main MCP Inspector window, find your configured agent in the list.
        *   Click "Connect" or the play icon next to it. MCP Inspector will run your script.
        *   Once connected, you'll see the available tools:
            *   `initialize_and_index`: Run this tool first. It takes no arguments. It will process the documents in `docs_simple` and create a local vector database in `chroma_db_simple`. Check the agent's terminal logs for progress.
            *   `ask_question`: After indexing is complete, use this tool. It takes one argument:
                ```json
                {
                  "question": "Your question about the documents here"
                }
                ```
                The agent will retrieve relevant information and generate an answer.

4.  **Interact using `mcp-cli` (Alternative):**
    Ensure your Python script (`server.py`) is **not** already running. `mcp-cli` will launch it for each command.

    *   **List available tools (optional check):**
        ```bash
        mcp tool list CompanyDocumentQA-Ollama --command "python server.py" --transport stdio
        ```
    *   **Index Documents:**
        ```bash
        mcp tool call CompanyDocumentQA-Ollama initialize_and_index --command "python server.py" --transport stdio
        ```
        *(Wait for this to complete. You'll see logs in your terminal.)*

    *   **Ask a Question:**
        ```bash
        mcp tool call CompanyDocumentQA-Ollama ask_question '{"question": "What is the company policy on annual leave?"}' --command "python server.py" --transport stdio
        ```

## Script Overview (`server.py`)

*   **Configuration:** Constants at the top for document directory, ChromaDB path, Ollama models, etc.
*   **Helper Functions:** For loading and splitting documents, initializing Ollama components.
*   **`initialize_and_index` (MCP Tool):**
    *   Loads `.docx` and `.xlsx` files.
    *   Splits them into chunks.
    *   Generates embeddings using Ollama.
    *   Stores chunks and embeddings in a persistent ChromaDB.
*   **`ask_question` (MCP Tool):**
    *   Takes a user question.
    *   Embeds the question.
    *   Retrieves relevant document chunks from ChromaDB.
    *   Constructs a prompt with the question and context.
    *   Gets an answer from the Ollama LLM.
*   **Main Block:** Sets up logging, initializes Ollama components, and starts the MCP agent server on `stdio`.

## Customization & Future Enhancements

*   **More Document Types:** Add loaders for PDFs (`PyPDFLoader`), text files (`TextLoader`), etc., in `load_documents_from_directory`. Remember to install necessary packages (e.g., `pypdf`, `unstructured`).
*   **Different Models:** Change `EMBEDDING_MODEL_NAME` and `LLM_MODEL_NAME` to use other Ollama models. Ensure they are pulled locally.
*   **Chunking Strategy:** Experiment with `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter` for better results.
*   **Retriever Options:** Modify `search_kwargs={"k": 3}` in `ask_question` to retrieve more/fewer chunks. Explore other retrieval modes if needed.
*   **Prompt Engineering:** Refine the prompt template in `ask_question` for better LLM responses.
*   **Error Handling:** Enhance error handling and user feedback.
*   **Web UI (e.g., Streamlit/Gradio):** Wrap the MCP agent or its core logic in a simple web UI for easier non-technical user access, potentially by having the UI call the MCP agent tools.

## Troubleshooting

*   **"Ollama not found" / Connection Errors:**
    *   Ensure Ollama is running (`ollama serve` or the Ollama desktop app).
    *   Verify `OLLAMA_BASE_URL` in the script matches your Ollama setup (default is `http://localhost:11434`).
    *   Make sure the models (`nomic-embed-text`, `llama3`) are pulled: `ollama list`.
*   **"No documents found" / "Vector store empty":**
    *   Double-check the `DOC_DIR` path in the script and ensure it points to the correct folder.
    *   Make sure your document files are in that folder and have supported extensions (.docx, .xlsx).
    *   Run the `initialize_and_index` tool. Check the terminal logs for errors during indexing.
*   **`mcp-cli` issues:**
    *   Ensure `mcp-cli` is installed correctly (`pip install "mcp-cli[cli]"`).
    *   Use the full `--command "python /path/to/script.py"` and `--transport stdio` flags.
*   **MCP Inspector doesn't see the agent:**
    *   Ensure you've configured the agent correctly in MCP Inspector preferences with the full path to the Python script and `stdio` transport.
    *   Make sure the script is not already running when MCP Inspector tries to start it.
