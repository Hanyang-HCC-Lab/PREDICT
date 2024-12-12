## Overview
[PREDICT: Multi-Agent-based Debate Simulation for Generalized Hate Speech Detection](https://aclanthology.org/2024.emnlp-main.1166/)

## Run experiments
### 1. Environmental Setup
#### 1.1. Install Requirements
```bash
pip install -r requirements.txt
```
#### 1.2. Verify File Structure
```bash
PREDICT/
├── Dataset/      # Change this directory to your desired dataset
│   ├── khaters/  # Change this folders to your desired dataset
│   ├── kmhas/
│   ├── kodoli/
│   ├── kold/
│   └── unsmile/
├── config/      # Configuration files folder
│    └── _pycache_/
│    └── environment.py
├── faiss/       # FAISS-based database construction (for embedded database construction)
│   └── README.md
├── output/      # Experimental results folder
│   └── Dataset_A/ # Dataset specific output folder
│       ├── PRE/       # Folder for main_pre.py results
│       ├── PRE_to_DICT/ # Folder for main_pre_to_dict.py results
│       └── DICT/      # Folder for main_dict.py results
├── src/        # Source code folder
│   └── _pycache_/
│   ├── main_pre.py
│   ├── main_dict.py
│   └── ...
├── utils/    # Utility scripts and resources
│   └── _pycache_/
│   ├── agent_debate.py
│   ├── debate_prompt.json
│   ├── openai_utils.py
│   └── retriever.py
├── .gitignore
├── README.md
├── main_pre.py
├── main_pre_to_dict.py
├── main_dict.py
└── requirements.txt
```
### 2. Data Preprocessing
#### 2.1. Select Data
- **Choose the Dataset**  
   - Select the dataset you want to use for the experiments.
   - Ensure the dataset is saved in the `Dataset/` directory in the following format: `<dataset_name>_sample.csv`.

- **Dataset Format**  
   - The dataset consists of two columns:
     - **Text**: The text data for analysis.
     - **Label**: The corresponding label for the text.

#### 2.2. Build the Database
- **Create Embeddings**  
   - Convert the selected dataset into embeddings and build a database.
   - This step involves embedding the **train set** of the selected dataset.

- **Follow Instructions**  
   - By default, you can refer to the instructions in `faiss/README.md` to complete the embedding process and build the database.

- **Alternative Methods**  
   - While FAISS is a common choice for creating the embedding database, you are free to use other embedding storage methods that fit your requirements

- **Purpose of the Database**  
   - The database is used for **RAG (Retrieve-then-Generate)** to retrieve 'similar context,' as described in the paper.

### 3. Run Experiments
#### 3.1. Run the PRE phase
Execute `main_pre.py` for each dataset and agent. You can use command-line arguments to specify the dataset, output path, embedding source, and agent prompt.
```bash
python main_pre.py -i Dataset/<evaluation_data>/<evaluation_data>_sample.csv \
                   -o output/Dataset_<evaluation_data>/PRE/Agent_<agent_name>.json \
                   -d <embedding_source> \
                   -a <agent_prompt>
## Example command:
# python main_pre.py -i Dataset/khaters/khaters_sample.csv \
#                    -o output/Dataset_A/PRE/Agent_E.json \
#                    -d Unsmile \
#                    -a someen/unsmile
```
- **Agent Prompt in LangChain**
   - The `<agent_prompt>` is stored in LangChain.
   - You can connect LangChain to access the pre-defined prompts. *(Add details if specific setup is required.)*
   - Visit the [LangChain Hub](https://smith.langchain.com/hub?organizationId=f1542bb4-2843-5b16-80f0-dd1d85524b88) and search for the following prompts to update the `<agent_prompt>` parameter:   
     - `someen/khaters`
     - `someen/kmhas`
     - `someen/kold`
     - `someen/kodoli`
     - `someen/unsmile`
- **Important Note:**
   - The `<agent_prompt>` and `<embedding_source>` must correspond to the **same dataset** to ensure consistency. For example: If `<agent_prompt>` is `someen/khaters`, then `<embedding_source>` must also be `khaters`.
   - Failing to match these parameters may lead to inconsistent results or errors during the experiment.


#### 3.2. Aggregate Agents' outputs
After running `main_pre.py` for all agents, merge the outputs using `main_pre_to_dict.py`.
```bash
python main_pre_to_dict.py -d <evaluation_data> -e <dataset_alias>

## Example command:
# python pre_to_dict.py -d khaters -e A
```
- <evaluation_data>: The name of the dataset being processed (e.g., khaters, kmhas, kold, kodoli, unsmile).
- <dataset_alias>: The corresponding alias used in the paper for the dataset:
  - khaters: A
  - kmhas: B
  - kold: C
  - kodoli: D
  - unsmile: E

#### 3.3. Run the DICT phase
The DICT phase involves a debate between two representative agents: one representing the **Hate** perspective and the other representing the **Non-Hate** perspective. This step processes the aggregated results to generate the final debate-based output. Execute the debate using `main_dict.py`.
```bash
python main_dict.py -i output/Dataset_<dataset_alias>/PRE_to_DICT/reference.csv \
                    -o output/Dataset_<dataset_alias>/DICT/

## Example command:
# python main_dict.py -i output/Dataset_A/PRE_to_DICT/reference.csv \
#                     -o output/Dataset_A/DICT/
```
- `-i` / `--input-file`
   - Specify the path to the merged CSV file generated by `main_pre_to_dict.py`. This file contains the aggregated outputs from the PRE phase.
   - Although this file is located in the `output/` folder, it serves as the input for this phase. The agent opinions in the PRE phase are used as references in the DICT phase.
   **Example**:  
   ```plaintext
   output/Dataset_A/PRE_to_DICT/reference.csv
   ```

- `-o` / `--output-dir`
   - Specify the directory where the final JSON results will be saved.
   
   **Example**:  
   ```plaintext
   output/Dataset_A/DICT/
   ```


## Reference

This code is based on [**Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate**](https://arxiv.org/abs/2305.19118).  
The datasets used in this project are provided as follows:  

- [`K-HATERS`](https://aclanthology.org/2023.findings-emnlp.952/): Link to the paper or resource describing the `K-HATERS` dataset.  
- [`K-MHaS`](https://aclanthology.org/2022.coling-1.311/): Link to the paper or resource describing the `K-MHaS` dataset.  
- [`KOLD`](https://aclanthology.org/2022.emnlp-main.744/): Link to the paper or resource describing the `KOLD` dataset.  
- [`KODOLI`](https://aclanthology.org/2023.findings-eacl.85/): Link to the paper or resource describing the `KODOLI` dataset.  
- [`UnSmile`](https://arxiv.org/abs/2204.03262): Link to the paper or resource describing the `UnSmile` dataset.  

Thank you to the contributors for releasing these valuable resources!  

## Citation
If you use this framework in your research, please cite:
```bibtex
@inproceedings{park2024predict,
  title={PREDICT: Multi-Agent-based Debate Simulation for Generalized Hate Speech Detection},
  author={Park, Someen and Kim, Jaehoon and Jin, Seungwan and Park, Sohyun and Han, Kyungsik},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={20963--20987},
  year={2024}
}
```
