import os
import json
import pandas as pd
import argparse

def agent_concat_5(data_name, evaluation_data):
    # Define agent names (A through E)
    agent_names = ['A', 'B', 'C', 'D', 'E']
    
    # Load ground truth data from CSV file
    gt_data_path = f'Dataset/{data_name}/{data_name}_sample.csv'
    gt_df = pd.read_csv(gt_data_path)
    
    # Read and combine prediction results from each agent's JSON file
    for agent in agent_names:
        predict_file_path = f'output/Dataset_{evaluation_data}/PRE/Agent_{agent}.json'
        
        # Load JSON data from each agent's prediction file
        with open(predict_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Parse JSON data and add predictions to the main dataframe
        for index, row in gt_df.iterrows():
            key = str(index)  # Convert index to string for JSON key matching
            if key in data:   # Check if the key exists in JSON data
                item = data[key]
                gt_df.loc[index, f'Agent_{agent}_Label'] = item['Label']
                gt_df.loc[index, f'Agent_{agent}_Reason'] = item['Reason']

    # Define classification criteria for each agent
    hate_criteria = {
        'Agent_A': ['Offensive'],
        'Agent_B': ['Hate Speech'],
        'Agent_C': ['Offensive'],
        'Agent_D': ['Offensive'],
        'Agent_E': ['Hate Speech']
    }
    not_hate_criteria = {
        'Agent_A': ['Not Offensive'],
        'Agent_B': ['Not Hate Speech'],
        'Agent_C': ['Not Offensive'],
        'Agent_D': ['Not Offensive'],
        'Agent_E': ['Not Hate Speech']
    }

    # Calculate binary indicators for hate and non-hate predictions
    for agent in agent_names:
        full_agent_name = f'Agent_{agent}'
        gt_df[f'{full_agent_name}_Hate'] = gt_df[f'{full_agent_name}_Label'].apply(
            lambda x: x in hate_criteria[full_agent_name]
        )
        gt_df[f'{full_agent_name}_Not_Hate'] = gt_df[f'{full_agent_name}_Label'].apply(
            lambda x: x in not_hate_criteria[full_agent_name]
        )

    # Sum up total hate and non-hate votes from all agents
    gt_df['Hate_count'] = gt_df[[f'Agent_{agent}_Hate' for agent in agent_names]].sum(axis=1)
    gt_df['Not_Hate_count'] = gt_df[[f'Agent_{agent}_Not_Hate' for agent in agent_names]].sum(axis=1)

    # Determine final label based on majority vote
    gt_df['Final_Label'] = gt_df.apply(lambda x: '혐오' if x['Hate_count'] > x['Not_Hate_count'] else '비혐오', axis=1)

    # Combine reasoning from agents based on their classifications
    gt_df['Hate_Reason'] = ''
    gt_df['Not_Hate_Reason'] = ''
    for index, row in gt_df.iterrows():
        hate_explain = []
        not_hate_explain = []
        for agent in agent_names:
            full_agent_name = f'Agent_{agent}'
            label = row[f'{full_agent_name}_Label']
            if label in hate_criteria[full_agent_name]:
                hate_explain.append(row[f'{full_agent_name}_Reason'])
            elif label in not_hate_criteria[full_agent_name]:
                not_hate_explain.append(row[f'{full_agent_name}_Reason'])
        gt_df.at[index, 'Hate_Reason'] = ' '.join(hate_explain)
        gt_df.at[index, 'Not_Hate_Reason'] = ' '.join(not_hate_explain)

    # Select and organize final columns for output
    base_columns = ['text', 'label']
    label_explain_columns = [f'Agent_{agent}_Label' for agent in agent_names] + [f'Agent_{agent}_Reason' for agent in agent_names]
    additional_columns = ['Hate_count', 'Not_Hate_count', 'Final_Label', 'Hate_Reason', 'Not_Hate_Reason']
    final_columns = base_columns + label_explain_columns + additional_columns
    
    # Create output directory and save results to CSV
    output_dir = f'output/Dataset_{evaluation_data}/PRE_to_DICT/'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/reference.csv'
    gt_df[final_columns].to_csv(output_path, index=False)
    
    return gt_df[final_columns]

def parse_args():
    parser = argparse.ArgumentParser(description="Combine and process agent outputs for hate speech detection.")
    parser.add_argument("-d", "--data-name", required=True, help="Name of the dataset directory (e.g., 'khaters')")
    parser.add_argument("-e", "--evaluation-data", required=True, help="Type of evaluation data being processed (e.g., 'A')")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = agent_concat_5(args.data_name, args.evaluation_data)
