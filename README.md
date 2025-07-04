# Intelligent Document Understanding API

This API combines state-of-the-art OCR technology, vector database retrieval, and large language models (LLMs) to help you understand and process any document efficiently.

## Features

- **Advanced OCR**: Extract text using traditional or deep-learning-powered OCR backends.
- **Vector Database Retrieval**: Quickly find relevant information.
- **LLM Integration**: Seamlessly interact with and understand document contents.

## Installation

1. **Install Dependency Manager**  
   This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Follow the [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) on their website.

2. **Environment Variables**  
   Copy the template environment file and fill in your values:
   
   Edit `.env` to set required variables such as your HuggingFace (HF) access token.

3. **Set Up Model Endpoint**  
   - Register for a [Hugging Face](https://huggingface.co/) account and get your HF API token.
   - Deploy an instance of the [olmOCR-7B-0225-preview](https://huggingface.co/allenai/olmOCR-7B-0225-preview) model on HuggingFace Inference Endpoints and copy your endpoint URL.
   - Add your API token and endpoint URL to your `.env` file.

## OCR Service

The API supports two OCR engines:

- **tesseract**: The open-source OCR engine. Suitable for basic text extraction, but limited on complex or non-standard layouts.
- **olmo_ocr**: A fine tuned from Qwen2-VL-7B-Instruct model deployed via HuggingFace.  
  This is recommended due to:
    - **Advanced Extraction**: Handles complex layouts and noisy images better.
    - **Anchor Functionality**: Uses `tesseract`-extracted text as “anchors” to guide the model (`olmo_ocr`) in extracting more accurate and context-aware text blocks.

**Recommendation:** Use `olmo_ocr` for best results, especially when high-fidelity or structured extraction is necessary.

## Testing

1. Authenticate to download the kaggle dataset. Follow the instructions [here](https://github.com/Kaggle/kagglehub).
2. Download the dataset [Real World Documents Collections](https://www.kaggle.com/datasets/shaz13/real-world-documents-collections) like so: 

```python
import kagglehub

path = kagglehub.dataset_download("shaz13/real-world-documents-collections")

print("Path to dataset files:", path)
```

3. Move the dataset to the `data` folder.

4. Run the tests with `pytest`

## Next steps

- Due to time restraints, only two OCR engines have been implemented. In the future we should add more to improve the accuracy of the results. This [benchmark](https://huggingface.co/spaces/echo840/ocrbench-leaderboard) is a good starting point.