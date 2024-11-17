import json

def count_mismatches(json_file_path):
    """
    Parses the specified JSON file, counts matches and mismatches between 'prediction' and 'target',
    and outputs statistics on the number of items by prediction label.
    Also counts empty fields separately.
    
    Parameters:
        json_file_path (str): Path to the wei_predictions_cleaned.json file.
    
    Returns:
        tuple: (mismatch_counts dict, match_counts dict, empty_fields_count dict)
    """
    mismatch_counts = {}
    match_counts = {}
    empty_fields = {'empty_predictions': 0, 'empty_targets': 0}
    total_mismatches = 0
    total_matches = 0
    total_entries = 0

    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        for entry in data:
            total_entries += 1
            prediction = entry.get('prediction', '').strip()
            target = entry.get('target', '').strip()
            
            # Count empty fields
            if not prediction:
                empty_fields['empty_predictions'] += 1
            if not target:
                empty_fields['empty_targets'] += 1
            
            # Skip comparison if either field is empty
            if not prediction or not target:
                continue
                
            if prediction != target:
                if prediction in mismatch_counts:
                    mismatch_counts[prediction] += 1
                else:
                    mismatch_counts[prediction] = 1
                total_mismatches += 1
            else:
                if prediction in match_counts:
                    match_counts[prediction] += 1
                else:
                    match_counts[prediction] = 1
                total_matches += 1
        print("--------------------------------")
        print(f"Total Entries: {total_entries}")
        print(f"Total Matches: {total_matches}")
        print(f"Total Mismatches: {total_mismatches}")
        print(f"Accuracy: {(total_matches/(total_matches + total_mismatches))*100:.2f}%")
        
        print(f"\nEmpty Fields:")
        print("--------------------------------")
        for field, count in empty_fields.items():
            print(f"{field}: {count}")
        
        print(f"\nMatches by Prediction Label:")
        print("--------------------------------")
        for label, count in match_counts.items():
            total = count + mismatch_counts.get(label, 0)
            accuracy = (count/total)*100
            print(f"{label}: {count} (Accuracy: {accuracy:.2f}%)")
        
        print(f"\nMismatches by Prediction Label:")
        print("--------------------------------")
        for label, count in mismatch_counts.items():
            total = count + match_counts.get(label, 0)
            error_rate = (count/total)*100
            print(f"{label}: {count} (Error Rate: {error_rate:.2f}%)")
        
        return mismatch_counts, match_counts, empty_fields

    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return {}, {}, {}
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not a valid JSON file.")
        return {}, {}, {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}, {}, {}

def convert_predictions_to_json(input_file: str, output_file: str):
    """
    Reads a predictions text file, removes specific tokens from the 'prediction' and 'target' fields,
    and writes the cleaned data to a JSON file.

    Args:
        input_file (str): Path to the input predictions text file.
        output_file (str): Path to the output JSON file.
    """
    cleaned_data = []
    token_to_remove = "<|end_of_text|>"
    token_to_replace = "Units"
    new_token= "Unit"

    try:
        with open(input_file, 'r') as infile:
            for line in infile:
                # Parse each JSON line
                record = json.loads(line.strip())
                
                # Remove the specified token from 'prediction' and 'target'
                record['prediction'] = record['prediction'].replace(token_to_remove, "").strip()
                record['target'] = record['target'].replace(token_to_remove, "").strip()
                record['prediction'] = record['prediction'].replace(token_to_replace, new_token)
                record['target'] = record['target'].replace(token_to_replace, new_token)
                
                cleaned_data.append(record)

        # Write the cleaned data to the output JSON file
        with open(output_file, 'w') as outfile:
            json.dump(cleaned_data, outfile, indent=2)
        
        print(f"Successfully converted {input_file} to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

convert_predictions_to_json("/home/weili3/VLSI-LLM-Graph/predictions/0_val__3B_lora.txt", "/home/weili3/VLSI-LLM-Graph/predictions/0_val__3B_lora.json")
count_mismatches("/home/weili3/VLSI-LLM-Graph/predictions/0_val__3B_lora.json")