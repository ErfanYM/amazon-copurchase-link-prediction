# Makefile for Amazon Co-Purchase Link Prediction

.PHONY: help install run clean

help:
	@echo "Amazon Co-Purchase Link Prediction - Available Commands:"
	@echo ""
	@echo "  make install    - Install Python dependencies"
	@echo "  make run        - Run the link prediction pipeline"
	@echo "  make clean      - Remove generated results and figures"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Note: Download datasets to data/ before running (see data/README.md)"

install:
	pip install -r requirements.txt

run:
	python src/run.py --data-dir data

clean:
	rm -rf results/*.csv figs/*.png
	@echo "Cleaned results and figures (data/ preserved)"
