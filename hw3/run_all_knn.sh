#!/bin/bash

# Define the input parameters
TRAIN_FILE="examples/train.vectors.txt"
TEST_FILE="examples/test.vectors.txt"
OUTPUT_DIR="output_results"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Define different values for k and distance function
K_VALUES=(1 5 10)
DISTANCE_FUNCTIONS=("1" "2")  # 1 = Euclidean, 2 = Cosine

# Loop over different combinations of k and distance functions
for K in "${K_VALUES[@]}"; do
  for FUNC in "${DISTANCE_FUNCTIONS[@]}"; do
    OUTPUT_FILE="$OUTPUT_DIR/output_k${K}_func${FUNC}.txt"
    ACC_FILE="$OUTPUT_DIR/acc_k${K}_func${FUNC}.txt"
    
    echo "Running KNN with k=$K, function=$FUNC..."
    
    # Run the Python script with current parameters
    python3 build_knn.py --train "$TRAIN_FILE" --test "$TEST_FILE" --k "$K" --function "$FUNC" --output "$OUTPUT_FILE" > "$ACC_FILE"
    
    echo "Results saved in $OUTPUT_FILE"
  done
done

echo "All runs completed!"
