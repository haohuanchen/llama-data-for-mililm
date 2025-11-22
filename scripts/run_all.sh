#!/bin/bash

models=(
    "Llama8B_sts:5e-6:1"
    # "Llama8B_bitext:2e-6:10"
    # "Llama8B:2e-6:10"

    # "Llama70B_sts:5e-6:10"
    # "Llama70B_bitext:2e-6:10"
    # "Llama70B:2e-6:10"
    
    # "QuoraDuplicates:5e-6:10"
    # "Allnli:2e-6:10"
    # "Public:2e-6:10"
)

train_model() {
    model="$1"
    lr="$2"
    epochs="$3"

    if [ -z "$model" ]; then
        echo "ERROR: Missing model name."
        exit 1
    fi

    if [ -z "$lr" ]; then
        lr="2e-6"
    fi

    if [ -z "$epochs" ]; then
        epochs="1"
    fi

    echo "Training model=${model}"

    data_path="data/${model}.jsonl"

    python -m train \
        --config configs/train.yaml \
        --model_name ${model} \
        --data_path ${data_path} \
        --epochs ${epochs} \
        --lr ${lr}

    if [ $? -ne 0 ]; then
        echo "ERROR: train model failed!"
        exit 1
    fi
}

evaluate_model() {
    model="$1"
    epochs="$2"

    if [ -z "$epochs" ]; then
        epochs="1"
    fi

    for ((i=1; i<=epochs; i++)); do
        echo "Evaluating model=${model}, Epoch=$i"

        model_revision="${model}_E${i}"

        python -m evalute \
            --config configs/evalute.yaml \
            --model_revision ${model_revision}

        if [ $? -ne 0 ]; then
            echo "ERROR: evaluate model failed at Epoch $i!"
            exit 1
        fi
    done
}

for item in "${models[@]}"; do
    IFS=":" read -r model lr epochs <<< "$item"

    echo "====================================="
    echo "START model=${model} epochs=${epochs}"
    echo "====================================="

    train_model "$model" "$lr" "$epochs"
    evaluate_model "$model" "$epochs"

    echo "FINISHED model=${model}"
    echo "====================================="
done