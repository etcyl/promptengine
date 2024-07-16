import pandas as pd
import logging
from prompt_engine import LLMInterface

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLMInterface
llm_interface = LLMInterface(model_name='gpt2')

# Load data
data = pd.read_csv('dataset.csv')
X = data.drop('target', axis=1)
y = data['target']

def prompt_llm_for_features(dataset_description):
    prompt = f"""
    Given the dataset with the following description: {dataset_description}, suggest new features and transformations to improve model performance.
    """
    response = llm_interface.generate_text(prompt, max_length=200, num_return_sequences=1)
    return response[0] if isinstance(response, list) else response

dataset_description = "The dataset contains features such as age, salary, education level, and work experience."
new_features_code = prompt_llm_for_features(dataset_description)
logger.info("Generated Feature Engineering Code:\n%s", new_features_code)

# Ensure the generated code is reviewed and safe to execute before running it
exec(new_features_code)
