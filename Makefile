.PHONY: setup train test infer clean help

# Default target
help:
	@echo "MEdi chain ai - Development Makefile"
	@echo "Available targets:"
	@echo "  setup    : Setup the Docker environment and install hooks"
	@echo "  train    : Run the training pipeline"
	@echo "  test     : Run all pytest tests"
	@echo "  infer    : Run the inference/demo"
	@echo "  clean    : Remove temporary files and logs"

setup:
	docker-compose -f deployment/docker-compose.yml up -d --build
	@echo "Setup complete. Vector DB (Milvus) and Dev Container are running."

test:
	pytest tests/ --cov=src --cov-report=term-missing

train:
	python src/main.py --mode train

infer:
	streamlit run src/ui/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
