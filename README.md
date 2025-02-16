# Document Analysis System

A robust document processing and summarization system that leverages LangChain to integrate with OpenAI and Anthropic language models. This system processes various document formats and generates intelligent summaries with different styles and depths.

## Features

### Document Processing
- Supports multiple file formats:
  - PDF (with text extraction)
  - DOCX (Microsoft Word)
  - TXT (Plain text)
- Intelligent text chunking for large documents
- Metadata extraction
- Error handling for corrupted files

### Summary Generation
- Multiple summary types:
  - Brief (concise overview)
  - Detailed (comprehensive analysis)
  - Bullet Points (key takeaways)
- Multi-language support
- Configurable output length
- Rate limiting and retry mechanisms

### Model Management
- Supports multiple LLM providers:
  - OpenAI (gpt-5)
  - Anthropic (claude-3.5-sonnet)
- Automatic provider switching
- Error handling and retries
- Rate limit management

### Technical Features
- RESTful API endpoints
- Comprehensive logging
- Input validation
- Error handling
- Type checking
- Performance monitoring

## Installation

### Prerequisites
- Python 3.11 or higher
- Poetry package manager
- OpenAI API key
- Anthropic API key

### Setup
1. Install Poetry if not already installed:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/zekeri-riya/aracor-ai-assignment-zekeri-riya
   cd aracor-ai-assignment-zekeri-riya
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Environment Variables
Required variables in `.env`:
```plaintext
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
DEFAULT_MODEL=openai  # or anthropic
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Usage

### Starting the API Server
```bash
poetry run uvicorn src.api.endpoints:app --reload
```

### API Endpoints

#### 1. Analyze Document
```http
POST /analyze/
```
Parameters:
- `file`: Document file (PDF, DOCX, or TXT)
- `summary_type`: Type of summary (brief, detailed, bullet_points)
- `model`: Model provider (openai, anthropic)
- `language`: Target language (optional)

Example using curl:
```bash
curl -X POST http://localhost:8000/analyze/ \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "summary_type=brief" \
  -F "model=openai"
```

#### 2. List Available Models
```http
GET /models/
```

#### 3. List Summary Types
```http
GET /summary-types/
```

#### 4. Health Check
```http
GET /health/
```

### Example Response
```json
{
  "summary": "This is a summary of the document...",
  "metadata": {
    "filename": "document.pdf",
    "file_type": "application/pdf",
    "file_size": 1234567,
    "page_count": 5,
    "created_at": "2024-02-16T10:30:00Z"
  },
  "provider": "openai",
  "summary_type": "brief",
  "processing_time": 2.5,
  "token_count": 150
}
```

## Development

### Running Tests
```bash
# Run all tests
poetry run pytest

# Run with coverage report
poetry run pytest --cov=src --cov-report=html

# Run specific test file
poetry run pytest tests/test_document_processor.py
```

### Code Quality Checks
```bash
# Format code
poetry run black .
poetry run isort .

# Lint code
poetry run pylint src tests

# Type checking
poetry run mypy src tests
```

## Project Structure
```
project_root/
├── src/
│   ├── api/           # API endpoints and configuration
│   ├── config/        # Application configuration
│   ├── models/        # Data models and schemas
│   ├── processors/    # Document processing logic
│   ├── services/      # Core business logic
│   └── utils/         # Utility functions
└── tests/             # Test files
```

## Security

### API Key Management
- API keys are stored in environment variables
- Keys are validated on startup
- Support for key rotation (planned)

### Rate Limiting
- Per-provider rate limits
- Configurable thresholds
- Automatic retries with exponential backoff

### Input Validation
- File size limits
- Supported format validation
- Content validation
- Pydantic model validation

## Performance

### Benchmarks
- Average processing time:
  - PDF (1MB): 2-3 seconds
  - DOCX (1MB): 1-2 seconds
  - TXT (1MB): < 1 second
- Concurrent requests: Up to 50/minute
- Memory usage: ~100MB baseline

### Optimization Features
- Text chunking for large documents
- Concurrent processing where applicable
- Caching (planned)
- Rate limit optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Troubleshooting

### Common Issues

1. Installation Problems
   ```bash
   # If poetry install fails, try:
   poetry config virtualenvs.create true
   poetry env remove --all
   poetry install
   ```

2. API Key Issues
   ```bash
   # Verify your .env file:
   cat .env
   # Check environment variables:
   poetry run python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

3. PDF Processing Issues
   ```bash
   # Install system dependencies:
   sudo apt-get install poppler-utils  # Ubuntu/Debian
   brew install poppler               # macOS
   ```

### Debug Mode
Enable debug logging in `.env`:
```plaintext
LOG_LEVEL=DEBUG
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain team for the excellent framework
- OpenAI and Anthropic for their LLM APIs
- Contributors and maintainers

## Contact

For support or questions, please open an issue in the GitHub repository.