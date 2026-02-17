.PHONY: install clean train evaluate inference

# Detect environment for Unsloth install (Optimized for Colab T4)
install:
	@echo "Installing Core Dependencies..."
	pip install -r requirements.txt
	@echo "Installing Unsloth..."
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	pip install --no-deps "xformers<0.0.27"
	pip install -e .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
	rm -rf outputs/ logs/

# Execution Shortcuts
train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

inference:
	python scripts/inference.py --config configs/default.yaml