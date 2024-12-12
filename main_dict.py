import os
import json
import random
import argparse
from langcodes import Language
from src.utils.agent_debate import Agent
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# A predefined list of player names for the debate participants
NAME_LIST = [
    "Non Hate side",
    "Hate side",
]

class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        """
        Initialize a debate player with specific configuration parameters.
        
        This class extends the base Agent class to create a player for an AI-driven debate.
        
        Args:
            model_name (str): The name of the AI model to be used (e.g., 'gpt-3.5-turbo')
            name (str): The name of the player/side in the debate
            temperature (float): Controls the randomness of the model's responses
                - Lower values (closer to 0) make responses more focused and deterministic
                - Higher values increase creativity and randomness
            openai_api_key (str): API key for accessing OpenAI's services
            sleep_time (float): Delay between API calls to manage rate limits
        """
        super().__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key

class Debate:
    def __init__(self,
            model_name: str = 'gpt-3.5-turbo-0125', 
            temperature: float = 0, 
            num_players: int = 2, 
            save_file_dir: str = None,
            openai_api_key: str = None,
            prompts_path: str = None,
            max_round: int = 2,
            sleep_time: float = 0
        ) -> None:
        """
        Initialize a debate simulation with configurable parameters.
        
        This class manages the entire debate process, including:
        - Creating debate players
        - Managing debate rounds
        - Generating final judgment
        - Saving debate results
        
        Args:
            model_name (str): AI model to be used for debate
            temperature (float): Response randomness control
            num_players (int): Number of debate participants
            save_file_dir (str): Directory to save debate results
            openai_api_key (str): OpenAI API key
            prompts_path (str): Path to JSON file containing debate prompts
            max_round (int): Maximum number of debate rounds
            sleep_time (float): Delay between API calls
        """
        # Store configuration parameters
        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.save_file_dir = save_file_dir
        self.openai_api_key = openai_api_key
        self.max_round = max_round
        self.sleep_time = sleep_time

        # Initialize a structured save file to track debate details
        self.save_file = {
            'num_players': num_players,
            'success': False,
            'text': '',
            'ground_truth': '',
            'players': {},
        }
        # Load debate prompts from a JSON configuration file
        prompts = json.load(open(prompts_path, encoding='utf-8'))
        self.save_file.update(prompts)
        
        # Prepare and customize prompts
        self.init_prompt()

        # Create debate agents and initialize their settings
        self.create_agents()
        self.init_agents()

    def init_prompt(self):
        """
        Prepare and customize debate prompts by replacing placeholders with actual content.
        
        This method replaces generic placeholders in prompts with specific text,
        enabling dynamic prompt generation for each debate scenario.
        """
        
        def prompt_replace(key):
            # Replace generic text placeholder with specific debate text
            self.save_file[key] = self.save_file[key].replace("##text##", self.save_file["text"])
        prompt_replace("text")
        prompt_replace("NonHate_player_meta_prompt")
        prompt_replace("Hate_player_meta_prompt")
        prompt_replace("judge_prompt_2")
        
        def script_replace(key):
            # Replace reference placeholders with specific reasoning
            self.save_file[key] = self.save_file[key].replace("##Non_Hate_Reference##", self.save_file["Not_Hate_Reason"])
            self.save_file[key] = self.save_file[key].replace("##Hate_Reference##", self.save_file["Hate_Reason"])
        script_replace("NonHate_prompt_1")
        script_replace("Hate_prompt_1")

    def create_agents(self):
        """
        Create debate players using the predefined configuration.
        
        Initializes two players with specific model parameters:
        1. A "Non Hate" side player
        2. A "Hate" side player
        """
        # Create players
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature, openai_api_key=self.openai_api_key, sleep_time=self.sleep_time) for name in NAME_LIST
        ]
        # Convenience references to specific players
        self.nothate = self.players[0]
        self.hate = self.players[1]

    def init_agents(self):
        """
        Initialize debate agents by setting their meta prompts and conducting the first round of debate.
        
        First round involves:
        1. Setting meta prompts for each player
        2. Having the "Non Hate" side state their initial argument
        3. Having the "Hate" side respond to that argument
        """        
        # Set meta prompts for each player
        self.nothate.set_meta_prompt(self.save_file['NonHate_player_meta_prompt'])
        self.hate.set_meta_prompt(self.save_file['Hate_player_meta_prompt'])

        # First round debate: state initial opinions
        print(f"===== Debate Round-1 =====\n")
        self.nothate.add_event(self.save_file['NonHate_prompt_1'])
        self.not_ans = self.nothate.ask()
        self.nothate.add_memory(self.not_ans)

        self.hate.add_event(self.save_file['Hate_prompt_1'].replace('##non_arg##', self.not_ans))
        self.hate_ans = self.hate.ask()
        self.hate.add_memory(self.hate_ans)

    def debate_round(self):
        """
        Conduct the second round of the debate.
        
        In this round:
        1. Each side receives and responds to the other side's previous argument
        2. Players add the received arguments to their memory
        3. Players generate responsive arguments
        """
        
        print(f"===== Debate Round-2 =====\n")
        
        # Non-Hate side responds to Hate side's argument
        self.nothate.add_memory(self.save_file['NonHate_arg_prompt'].replace('##non_arg##', self.not_ans))
        self.nothate.add_event(self.save_file['NonHate_prompt_2'].replace('##hate_arg##', self.hate_ans))
        self.not_res = self.nothate.ask()
        self.nothate.add_memory(self.not_res)

        # Hate side responds to Non-Hate side's argument
        self.hate.add_memory(self.save_file['Hate_arg_prompt'].replace('##hate_arg##', self.hate_ans))
        self.hate.add_event(self.save_file['Hate_prompt_2'].replace('##non_res##', self.not_res))
        self.hate_res = self.hate.ask()
        self.hate.add_memory(self.hate_res)

    def final_judgment(self):
        """
        Generate a final judgment for the debate.
        
        Process:
        1. Create a judge agent
        2. Compile debate history from all players
        3. Generate a judgment using predefined prompts
        4. Attempt to parse the judgment as JSON
        5. Update save file with judgment results
        """
        
        print(f"===== Final Judgment =====\n")
        
        # Create a judge agent with the same model configuration
        judge_player = DebatePlayer(model_name=self.model_name, name='Judge', temperature=self.temperature, openai_api_key=self.openai_api_key, sleep_time=self.sleep_time)

        # Compile debate history as JSON
        debate_history = json.dumps({player.name: player.memory_lst for player in self.players}, ensure_ascii=False)
        # Generate judgment
        judge_player.add_event(self.save_file['judge_prompt_1'].replace('##history##', debate_history))
        judge_player.add_event(self.save_file['judge_prompt_2'])
        judgment = judge_player.ask()
        judge_player.add_memory(judgment)

        # Parse judgment and update save file
        try:
            result = json.loads(judgment)
            self.save_file['success'] = True
            self.save_file.update(result)
        except json.JSONDecodeError:
            print("Error parsing judge's response.")

    def save_file_to_json(self, id):
        """
        Save the debate results to a JSON file.
        
        Args:
            id (int/str): Unique identifier for the debate session
        """
        
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    def run(self):
        """
        Execute the entire debate process.
        
        Workflow:
        1. Conduct the debate round
        2. Generate final judgment
        3. Save player memories to the save file
        """
        
        self.debate_round()
        self.final_judgment()
        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst

def parse_args():
    """
    Parse command-line arguments for the script.
    
    Required Arguments:
    - input-file: Path to the CSV file containing debate input data
    - output-dir: Directory to save debate result files
    
    Returns:
        Parsed command-line arguments
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory to store results")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Determine script and configuration paths
    current_script_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_script_path)
    config_path = os.path.join(current_directory, "src/utils", "debate_prompt.json")

    # Read the configuration file
    with open(config_path, "r", encoding='utf-8') as config_file:
        config = json.load(config_file)

    # Read input data using pandas
    inputs = pd.read_csv(args.input_file)

    # Ensure output directory exists
    save_file_dir = args.output_dir
    os.makedirs(save_file_dir, exist_ok=True)

    # Iterate through input data and run debates
    for id, row in tqdm(inputs.iterrows(), total=inputs.shape[0]):
        # Create a unique configuration file for each debate
        prompts_path = os.path.join(save_file_dir, f"{id}-config.json")

        # Update configuration with specific row data
        config['text'] = str(row['text'])
        config['ground_truth'] = str(row['label'])
        config['Not_Hate_Reason'] = str(row['Not_Hate_Reason'])
        config['Hate_Reason'] = str(row['Hate_Reason'])

        # Prepare player meta prompts
        config['NonHate_player_meta_prompt'] = config['NonHate_player_meta_prompt'].replace("##text##", config['text'])
        config['Hate_player_meta_prompt'] = config['Hate_player_meta_prompt'].replace("##text##", config['text'])

        # Save the debate-specific configuration
        with open(prompts_path, 'w', encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        # Run the debate for this specific input
        debate = Debate(save_file_dir=save_file_dir, num_players=2, prompts_path=prompts_path, temperature=0, sleep_time=0)
        debate.run()
        debate.save_file_to_json(id)