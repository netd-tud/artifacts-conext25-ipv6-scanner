#!/bin/bash 

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

input_file="$1"
output_file="$2"

export output_file

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file not found!"
    exit 1
fi

# Function to perform reverse DNS lookup on a single IP address
perform_rdns_lookup() {
    local ip="$1"

    # Perform reverse DNS lookup using dig
    rdns=$(dig +short -x "$ip")

    # If no result is returned, output "No PTR record found"
    if [ -z "$rdns" ]; then
        rdns="No PTR record found"
    fi

    # Write the IP and its reverse DNS result to the output file
    echo "$ip|$rdns" >> "$output_file"
}

# Use parallel to perform reverse DNS lookups in parallel
export -f perform_rdns_lookup
cat "$input_file" | parallel perform_rdns_lookup {} "$output_file"

echo "Reverse DNS lookup completed. Results are saved in $output_file."

