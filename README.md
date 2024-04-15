# LangChain Document QA

Interface for interacting with files using langchain. Inspired by [ollama](https://github.com/ollama/ollama) examples. It follows the usual structure:

1. Load documents
2. Chunk them
3. Create embeddings and store them into a vector datastore
4. Retrieve from the datastore using a LLM

## Setup

Run Llama 2 locally:
```
ollama run llama2
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Run

Run the project:
```
python main.py
```

Example query:
```
Query: Can you summarise the content of the file?
```
