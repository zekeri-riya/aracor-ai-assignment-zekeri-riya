# aracor-ai-assignment-zekeri-riya

This project processes documents (PDF, DOCX, TXT) and generates summaries using
either OpenAI or Anthropic language models, via LangChain.

## Features

- Document processing (PDF, DOCX, TXT)
- Summaries of different styles (brief, detailed, bullet)
- Simple model management (OpenAI, Anthropic)
- Error handling, logging, linting, type-checking, testing

## Getting Started

1. Install [Poetry](https://python-poetry.org/docs/).
2. Run `poetry install`.
3. Copy `.env.example` to `.env` and fill in real API keys.
4. Run tests: `poetry run pytest --cov=src --cov-report=html`.
5. Lint and format:  
   ```bash
   poetry run black .
   poetry run isort .
   poetry run pylint src tests
   poetry run mypy src tests
