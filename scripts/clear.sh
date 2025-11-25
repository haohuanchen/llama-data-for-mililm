#!/bin/bash

MODEL_DIR="./models/minilm-l6-custom"

if [ -d "$MODEL_DIR" ]; then
    find "$MODEL_DIR" -mindepth 1 -delete
    echo "Cleared: $MODEL_DIR"
else
    echo "Directory not found: $MODEL_DIR"
fi

RESULTS_DIR="./results"

if [ -d "$RESULTS_DIR" ]; then
    find "$RESULTS_DIR" -mindepth 1 -delete
    echo "Cleared: $RESULTS_DIR"
else
    echo "Directory not found: $RESULTS_DIR"
fi