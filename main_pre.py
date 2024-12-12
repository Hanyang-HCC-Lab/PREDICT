import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
import subprocess
from config.environment import set_environment_variables
from src.utils.retriever import RAG, init_vectorstore
import time

# Maximum retry attempts for RAG API calls during dataset processing
MAX_RETRIES = 3
# Waiting time between API call retry attempts
WAIT_TIME = 5

def parse_args():
    # CLI argument parser for flexible dataset processing configuration
    # Supports running single dataset or multiple predefined configurations
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_path", required=False, help="Path to input hate speech dataset") 
    parser.add_argument("--output_path", required=False, help="Path to save RAG processed results")  
    parser.add_argument("--dataset_name", required=False, help="Specify the embedding vector source for retrieval (e.g., Unsmile)")
    parser.add_argument("--prompt_name", required=False, help="Prompt_Specify agent persona for analysis (e.g., someen/unsmile)")
    parser.add_argument("--run-all", action="store_true", help="Process all predefined dataset configurations")
    return parser.parse_args()

def run_script(command):
    subprocess.run(command, shell=True)

# Set up environment variables
set_environment_variables()

def update_json_file(file_path, data_chunk):
    """
    Update a JSON file with new data chunks, creating it if it doesn't exist
    
    Args:
        file_path (str): Path to the JSON file
        data_chunk (dict): New data to be added to the JSON file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data.update(data_chunk)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def call_api_with_retry(text, agent_name):
    """
    Call RAG API with retry mechanism
    
    Args:
        text (str): Input text for RAG
        agent_name (str): Name of the agent to use
    
    Returns:
        Response from RAG or None if all retries fail
    """
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Attempt to call RAG API
            response = RAG(text, agent_name)
            return response
        except Exception as e:
            print(f"Error occurred: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{MAX_RETRIES})")
            time.sleep(WAIT_TIME)  # Wait before retrying
    print("Max retries reached. Failed to call API.")
    return None

def process_dataset(dataset_path, output_path, dataset_name, agent_name, batch_size=10):
    """
    Process a dataset by applying RAG to each text entry

    Args:
        dataset_path (str): Path to input CSV dataset
        output_path (str): Path to save processed results
        dataset_name (str): Embedding vector source for retrieval
        agent_name (str): Specific agent persona for analysis
        batch_size (int, optional): Number of entries to process before writing to file
    """
    dataset = pd.read_csv(dataset_path)
    length = len(dataset)
    init_vectorstore(dataset_name)
    data_chunk = {}
    for i in tqdm(range(length)):
        # Call RAG with retry mechanism
        response = call_api_with_retry(dataset['text'][i], agent_name)
        if response:
            # Store response with index as key
            data_chunk[str(i)] = response
        # Write to file in batches or at the end
        if (i + 1) % batch_size == 0 or i == length - 1:
            update_json_file(output_path, data_chunk)
            data_chunk = {}


if __name__ == "__main__":
    # Parse command-line arguments for dataset processing
    args = parse_args()
    
    # Run predefined hate speech dataset configurations for multi-agent analysis
    # PRE phase
    if args.run_all:
        dataset_configs = [
            ## Pre-processing configuration for hate speech dataset
            ## PRE phase output
            {
                # IMPORTANT: Path to the CSV dataset that will be evaluated for hate speech analysis
                "dataset_path": "Dataset/khaters/khaters_sample.csv",
                "output_path": "output/Dataset_A/PRE/Agent_E.json",
                # IMPORTANT: Embedding vector source for retrieval matching the specific dataset
                "dataset_name": "Unsmile",
                # IMPORTANT: Specific agent persona whose perspective/analysis is being sought for this dataset
                "agent_name": "someen/unsmile"
            }
        ]
        for config in dataset_configs:
            process_dataset(config["dataset_path"], config["output_path"], config["dataset_name"], config["agent_name"])
    else:
        process_dataset(args.dataset_path, args.output_path, args.dataset_name, args.agent_name)
