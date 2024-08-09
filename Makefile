SHELL := /bin/zsh

genes:
	@echo "Running genetic traiing batch"
	python3.12 genetic_training.py

run:
	@echo "Running single iteration of network training"
	python3.12 network_basic.py

.PHONY: batch run