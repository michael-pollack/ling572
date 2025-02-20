#!/bin/bash

# Define three groups of input sets (each containing multiple sets with an identifier)
# Format: "<script_type>:<space-separated inputs>"
input_groups=(
    "zero: f1 0.05 200 1 1 2 | zero: f1 0.05 200 2 1 2 | zero: f1 0.05 200 3 1 2 | zero: f1 0.05 200 1 1 2.03 | zero: f1 0.05 200 2 1 2.03 | zero: f1 0.05 200 3 1 2.03 | grad: f1 0.05 200 1 2 | grad: f1 0.05 200 1 2.03 | zero: f1 0.05 200 1 100 20 | zero: f1 0.05 200 2 100 20 | grad: f1 0.05 200 100 20"  # Group 1
    "grad: g1 0.01 200 -1.8 | grad: g1 0.01 200 1.8 | grad: g2 0.01 200 -1.8 | grad: g2 0.01 200 1.8 | grad: g3 0.01 200 -1.8 | grad: g3 0.01 200 1.8 | grad: g4 0.01 200 -1.8 | grad: g4 0.01 200 1.8 | grad: g5 0.01 200 -1.8 | grad: g5 0.01 200 1.8 | grad: g6 0.01 200 -1.8 | grad: g6 0.01 200 0.77 | grad: g6 0.01 200 0.05 | grad: g6 0.01 200 0.3 | grad: g6 0.01 200 0.76 |  grad: f1 0.01 200 1 2 | grad: f2 0.01 200 1 2 | grad: f3 0.01 200 1 2 | grad: f4 0.01 200 1 2"  # Group 2
    "grad: g3 0.01 200 100 | grad: g3 0.1 200 100 | grad: g3 0.3 200 100 | grad: g3 0.6 200 100 | grad: g3 1.0 200 100 | grad: g3 1.01 200 100"  # Group 3
)

# Define corresponding folder names
group_folders=("q2" "q4" "q6")
file_prefixes=("Z" "E" "L")

# Create group-level folders
for folder in "${group_folders[@]}"; do
    mkdir -p "results/$folder"
done

# Process each group separately
for group_idx in "${!input_groups[@]}"; do
    IFS='|' read -ra input_sets <<< "${input_groups[$group_idx]}"  # Split sets using '|'
    group_folder="results/${group_folders[$group_idx]}"  # Define folder for this group
    file_prefix="${file_prefixes[$group_idx]}"
    
    # Create subfolders for each input set inside the group folder
    echo "${input_sets}"
    for set_idx in "${!input_sets[@]}"; do
        set_data="${input_sets[$set_idx]}"  # Extract set (e.g., "grad:0.1 0.5 1.0")
        
        # Extract identifier and input values
        script_type=$(echo "$set_data" | cut -d':' -f1)  # "grad" or "zero"
        inputs=$(echo "$set_data" | cut -d':' -f2-)  # Remaining input values

        output="$group_folder/${file_prefix}$((set_idx + 1))"

        # Run grad_desc.sh if the identifier is "grad"
        if [[ "$script_type" == "grad" ]]; then
            echo "Running grad_desc.sh for Group $((group_idx + 1)), Set $((set_idx + 1))..."
            eval "./grad_desc.sh $inputs" > "$output"
        fi

        # Run zero_order.sh if the identifier is "zero"
        if [[ "$script_type" == "zero" ]]; then
            echo "Running zero_order.sh for Group $((group_idx + 1)), Set $((set_idx + 1))..."
            eval "./grad_desc.sh $inputs" > "$output"
        fi
    done
done

echo "Script execution completed."
