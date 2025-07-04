# Intelligent Document Understanding API

This API seamlessly integrates state-of-the-art OCR technology, vector database retrieval, and large language models (LLMs) to efficiently process and understand JPG documents.

## Features

- **Advanced OCR**: Extracts text using both traditional and deep-learning-powered OCR engines.
- **Vector Database Retrieval**: Quickly finds the most relevant information from your document set.
- **LLM Integration**: Interprets, validates, and extracts structured information from document contents.

## Installation

1. **Install Dependency Manager**  
   This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Follow the [official instructions](https://docs.astral.sh/uv/getting-started/installation/) to install `uv`.

2. **Set Up Environment Variables**  
   Copy the template `.env` file and fill in your credentials.  
   Important variables include your HuggingFace (HF) access token and the OCR endpoint URL.

3. **Set Up Model Endpoint**  
   - Register for a [Hugging Face](https://huggingface.co/) account and obtain your API token.
   - Deploy an instance of the [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) model via HuggingFace Inference Endpoints, and copy your endpoint URL.
   - Add your API token and endpoint URL to your `.env` file.

## Data Setup

1. Authenticate with Kaggle to access and download datasets. Refer to [KaggleHub instructions](https://github.com/Kaggle/kagglehub).
2. Run this command to download the [Real World Documents Collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections) dataset and to populate the vector database with training data:
   ```shell
   uv run setup.py
   ```
   - This command downloads the dataset, moves it to the `data` directory, splits it into training and testing sets, and populates the vector database with training data.

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

1. **Upload**: The user uploads a document (JPEG image).
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

## Running with Docker

**Note:** Run the setup script (`setup.py`) before building the Docker container.

Start the service:
```shell
docker compose up --build
```

Run integration tests:
```shell
uv run pytest -v tests/
```

Alternatively, send a manual request:
```python
import requests

with open("data/test/advertisement/660202.jpg", "rb") as f:
    image_bytes = f.read()
url = "http://localhost:8000/v1/extract-entities"
files = {"file": ("invoice.jpg", image_bytes, "image/jpg")}

response = requests.post(url, files=files)
data = response.json()
```

## Running Locally (Without Docker)

Start the FastAPI app directly:
```shell
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Testing

Execute tests using pytest:
```shell
uv run pytest -v tests/unit/services/ocr/
```

## Future Improvements

- Expand support for additional OCR engines to further improve extraction accuracy.  
  See [this OCR benchmark](https://huggingface.co/spaces/echo840/ocrbench-leaderboard) for candidate models.

---