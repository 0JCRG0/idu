# Intelligent Document Understanding API

This API seamlessly integrates state-of-the-art OCR technology, vector database retrieval, and large language models (LLMs) to efficiently process and understand JPG documents. Built with Django REST Framework for robust and scalable document processing.

---

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---

## Features

- **Advanced OCR**: Extracts text using both traditional and deep-learning-powered OCR engines.
- **Vector Database Retrieval**: Quickly finds the most relevant information from your document set.
- **LLM Integration**: Interprets, validates, and extracts structured information from document contents.

## Installation

1. **Install Dependency Manager**  
   This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Follow the [official instructions](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

2. **Install Dependencies**
   ```shell
   uv sync
   ```

3. **Set Up Environment Variables**  
   Copy the template `.env` file and fill in your credentials.  
   Important variables include your HuggingFace (HF) access token, OpenAI API key, and Anthropic API key.

4. **Set Up Model Endpoint**  
   - Register for a [Hugging Face](https://huggingface.co/) account and obtain your API token.
   - Deploy an instance of the [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) model via HuggingFace Inference Endpoints, and copy your endpoint URL.
   - Add your API token and endpoint URL to your `.env` file.

## Data Setup

**Important**: Before running the application (locally or via Docker), you must first populate the vector database.

1. **Authenticate with Kaggle**: To access and download datasets, refer to [KaggleHub instructions](https://github.com/Kaggle/kagglehub).

2. **Populate Vector Database**: Run the Django management command to download the [Real World Documents Collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections) dataset and populate the vector database:
   ```shell
   # Using Makefile (recommended)
   make populate-vectordb
   
   # Or run directly
   uv run manage.py populate_vectordb
   ```

This Django management command (`api/management/commands/populate_vectordb.py`) performs several key operations:
- Downloads the Real World Documents Collections dataset from Kaggle or from a specified local path
- Moves the dataset to the `data/` directory
- Splits the dataset into training (2% by default) and testing sets
- Processes images through OCR (using olmo_ocr by default)
- Generates embeddings and populates the ChromaDB vector database
- Implements batch processing and retry logic for failed files

### Command Options

The populate_vectordb command supports several options:
- `--dataset-path`: Use existing dataset instead of downloading (e.g., `--dataset-path data/test`)
- `--batch-size`: Set batch size for processing (default: 10)
- `--ocr-engine`: Choose OCR engine - "tesseract" or "olmo_ocr" (default: olmo_ocr)
- `--train-ratio`: Set training data ratio (default: 0.02 = 2%)

Example with custom options:
```shell
uv run manage.py populate_vectordb --batch-size 5 --ocr-engine tesseract --train-ratio 0.05
```

## OCR Service

This API supports two OCR engines:

- **tesseract**:  
  Open-source OCR engine. Best for simple documents, but may struggle with complex layouts and non-standard formats.
- **olmo_ocr**:  
  A fine-tuned Qwen2-VL-7B-Instruct model deployed via HuggingFace — recommended for most use cases:
  - **Advanced Extraction**: Handles complex layouts and noisy images with high accuracy.
  - **Anchor Functionality**: Uses text from `tesseract` as anchors to guide the `olmo_ocr` model for more context-aware extraction.

**Recommendation:** Choose `olmo_ocr` for best results, especially when dealing with structured or high-fidelity extraction needs.

## Vector Database

- Utilizes [Chroma](https://www.trychroma.com/) for embedding storage and similarity search.
- Embeddings are generated via OpenAI's `text-embedding-3-small` model, but the system is modular and can incorporate other vector databases or embedding models.
- Similarity scores are normalized with a sigmoid function to yield a confidence estimate, indicating the likelihood the extracted text matches the predicted document type.
- Results are further validated by an LLM to improve reliability, especially when the dataset expands.

## Processing Flow

1. **Upload**: The user uploads a document (JPEG, PNG or PDF).
2. **OCR**: The document is processed via the selected OCR service.
3. **Similarity Search**: The extracted text is compared against the vector database, identifying the nearest document type and providing a confidence score.
4. **LLM Validation**: The initial document type prediction and confidence score are validated by the LLM.
5. **Type Correction**: If the LLM disagrees with the initial prediction, it selects a new document type and loads the appropriate extraction prompt (confidence is set to `None` in this case).
6. **Entity Extraction**: Another LLM extracts the relevant fields/entities based on the validated document type.
7. **Response**: The API returns a structured response:

   ```python
   class DocumentModelResponse(BaseModel):
       """Response model for the document extraction endpoint."""

       document_type: str
       confidence: float | None
       entities: dict
       processing_time: float
   ```

## Running the Application

### Requirements

- **Local Development**: You must have `tesseract` and `poppler-utils` installed on your local machine
- **Docker**: All dependencies are included in the container

### Docker (Recommended)

**Note:** Run the Django management command.

Start the service:
```shell
docker compose up --build
# or use the Makefile command
make docker-up
```

Stop the service:
```shell
make docker-down
```

### Local Development

Start the Django development server:
```shell
python manage.py runserver 0.0.0.0:8000
# or use the Makefile command
make local-run
```

### Usage

Go to localhost:8000 to access the Django view and upload the document or documents.

## Makefile Commands

This project includes a Makefile with convenient commands for development:

### Database Management
- `make populate-vectordb` - Populate vector database with training data
- `make populate-vectordb-test` - Populate vector database with test data

### Code Quality
- `make lint` - Format code with ruff and fix linting issues
- `make pre-push` - Run linting checks and unit tests (recommended before pushing)

### Testing
- `make unit-test` - Run unit tests with coverage reporting
- `make integration-test` - Run integration tests

### Application
- `make local-run` - Start Django development server locally
- `make docker-up` - Start application in Docker container
- `make docker-down` - Stop Docker container

## Testing

Execute tests using pytest:
```shell
# Run all tests
uv run pytest -v tests/

# Run specific test suites
uv run pytest -v tests/unit/services/ocr/
uv run pytest -v tests/integration/

# Run tests with coverage
uv run pytest --cov=src --cov-report=term tests/

# Or use Makefile commands
make unit-test
make integration-test
```



## Future Improvements

- **Make the application truly async**: There are some blocking calls in the codebase that could be optimized for better performance.
- **Use other models for better performance**: Explore alternative OCR and LLM models for improved speed and accuracy. See [this OCR benchmark](https://huggingface.co/spaces/echo840/ocrbench-leaderboard) for candidate models
- **Production server deployment**: Implement proper production-grade server configuration with Gunicorn/uWSGI
- **Security improvements**: Implement malicious file detection before processing
- **Direct PDF parsing**: Parse PDFs natively instead of converting to images first

---