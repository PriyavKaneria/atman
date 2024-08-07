SHELL := /bin/zsh

# Define variables
Width := 400
Height := 300
X_Jobs := 4
Y_Jobs := 3

# Default values for X and Y
X ?= 0
Y ?= 0

batch:
	@for i in $$(seq 0 $$(($(X_Jobs) - 1))); do \
		for j in $$(seq 0 $$(($(Y_Jobs) - 1))); do \
			(make run X=$$((i * $(Width))) Y=$$((j * $(Height))) &); \
		done; \
	done; \
	wait

run:
	@echo "Running Python script with X=$(X) and Y=$(Y)"
	python3.12 network_basic.py $(X) $(Y)

.PHONY: batch run