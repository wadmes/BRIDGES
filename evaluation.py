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
    id2result = {'netlist_id': [], 'correct': []}
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
                
            id2result['netlist_id'].append(entry['netlist_id'])
            if prediction != target:
                if prediction in mismatch_counts:
                    mismatch_counts[prediction] += 1
                else:
                    mismatch_counts[prediction] = 1
                total_mismatches += 1
                id2result['correct'].append(0)
            else:
                if prediction in match_counts:
                    match_counts[prediction] += 1
                else:
                    match_counts[prediction] = 1
                total_matches += 1
                id2result['correct'].append(1)
        # print("--------------------------------")
        # print(f"Total Entries: {total_entries}")
        # print(f"Total Matches: {total_matches}")
        # print(f"Total Mismatches: {total_mismatches}")
        print(f"Accuracy: {(total_matches/(total_matches + total_mismatches))*100:.2f}%")
        
        # print(f"\nEmpty Fields:")
        # print("--------------------------------")
        # for field, count in empty_fields.items():
        #     print(f"{field}: {count}")
        
        # print(f"\nMatches by Prediction Label:")
        # print("--------------------------------")
        # for label, count in match_counts.items():
        #     total = count + mismatch_counts.get(label, 0)
        #     accuracy = (count/total)*100
        #     print(f"{label}: {count} (Accuracy: {accuracy:.2f}%)")
        
        # print(f"\nMismatches by Prediction Label:")
        # print("--------------------------------")
        # for label, count in mismatch_counts.items():
        #     total = count + match_counts.get(label, 0)
        #     error_rate = (count/total)*100
        #     print(f"{label}: {count} (Error Rate: {error_rate:.2f}%)")
        
        # save id2result as a csv with the same name as the json file (end with .csv)
        with open(json_file_path.replace('.json', '.csv'), 'w') as f:
            f.write("netlist_id,correct\n")
            for i, (netlist_id, correct) in enumerate(zip(id2result['netlist_id'], id2result['correct'])):
                f.write(f"{netlist_id},{correct}\n")
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
        
        # print(f"Successfully converted {input_file} to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

import pandas as pd

def calculate_accuracies(csv1_path, csv2_path):
    """
    Calculates overall accuracy and accuracies for different node count ranges.
    
    Args:
        csv1_path (str): Path to the first CSV file with columns "netlist_id" and "correct".
        csv2_path (str): Path to the second CSV file with columns "netlist_id" and "#nodes".
        
    Returns:
        dict: A dictionary containing overall accuracy and accuracies for each node range.
    """
    # Load the CSV files into DataFrames
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Merge the two DataFrames on netlist_id (df1) and id (df2)
    merged_df = pd.merge(df1, df2, left_on="netlist_id", right_on="id", how="inner")

    # Define node ranges
    ranges = [
        ("<10", lambda x: x < 10),
        ("[10, 100)", lambda x: 10 <= x < 100),
        ("[100, 1000)", lambda x: 100 <= x < 1000),
        ("[1000, 10000)", lambda x: 1000 <= x < 10000),
        (">10000", lambda x: x > 10000)
    ]

    # Calculate overall accuracy
    overall_accuracy = merged_df["correct"].mean()

    # Calculate accuracy for each range
    range_accuracies = {}
    for range_name, condition in ranges:
        filtered_df = merged_df[merged_df["#node"].apply(condition)]
        if len(filtered_df) > 0:
            range_accuracies[range_name] = float(round(filtered_df["correct"].mean(),4))
        else:
            range_accuracies[range_name] = None  # No data points in this range

    # Combine results into a dictionary. keep overall_accuracy for 2 decimal places
    accuracies = {"overall": float(round(overall_accuracy, 4))}
    accuracies.update(range_accuracies)

    return accuracies

# Example usage:
# accuracies = calculate_accuracies("csv1.csv", "csv2.csv")
# print(accuracies)


# convert_predictions_to_json("/home/weili3/VLSI-LLM-Graph/predictions/2_val__3B_lora.txt", "/home/weili3/VLSI-LLM-Graph/predictions/2_val__3B_lora.json")
# count_mismatches("/home/weili3/VLSI-LLM-Graph/predictions/2_val__3B_lora.json")

# for all .txt file in /home/weili3/VLSI-LLM-Graph/predictions, run convert_predictions_to_json and count_mismatches
import os
all_results = {}
all_results = {"epoch":[], "run_name":[], "accuracy":[], "<10": [], "[10, 100)": [], "[100, 1000)": [], "[1000, 10000)": [], ">10000": []}
for file in os.listdir("/home/weili3/VLSI-LLM-Graph/predictions"):
    if file.endswith(".txt"):
        print(f"Processing {file}...")
        input_file = os.path.join("/home/weili3/VLSI-LLM-Graph/predictions", file)
        output_file = input_file.replace(".txt", ".json")
        convert_predictions_to_json(input_file, output_file)
        count_mismatches(output_file)
        csv_file = output_file.replace(".json", ".csv")
        result = calculate_accuracies(csv_file,"/home/weili3/VLSI-LLM/data_collection/BRIDGES.csv")
        print(result)
        epoch = file.split("__")[0].split("_")[0]
        run_name = file.split("__")[1].replace(".txt", "")
        all_results["epoch"].append(epoch)
        all_results["run_name"].append(run_name)
        all_results["accuracy"].append(result["overall"])
        all_results["<10"].append(result["<10"])
        all_results["[10, 100)"].append(result["[10, 100)"])
        all_results["[100, 1000)"].append(result["[100, 1000)"])
        all_results["[1000, 10000)"].append(result["[1000, 10000)"])
        all_results[">10000"].append(result[">10000"])

        print("##################\n")

df = pd.DataFrame(all_results)
df.to_csv("/home/weili3/VLSI-LLM-Graph/predictions/summary.csv", index=False)