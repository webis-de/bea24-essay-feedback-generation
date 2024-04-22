import argparse
import copy
import json

from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from transformers import set_seed

from helpfulness_example_dataset import samples
from llm_functions import collect_llm_output, load_model

template = """
{model_prefix}
You are given an essay and feedback from a teacher for this essay. Your task is to evaluate the helpfulness of the feedback. 

# Task:
Evaluate the helpfulness of the feedback. Helpful feedback should explain what the errors are, why they are errors, and how to fix them.
Give a score between 1 and 10, where 1 means the feedback is not helpful at all, and 10 means the feedback is very helpful. 

#### Essay: 
'''{essay}'''
#### Feedback: 
'''{feedback}'''

Provide the output in the following output: 
{format_instructions}

{model_suffix}
"""


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def analyse_helpfulness(llm, config, prompt_template):
    """
    Analyzes the helpfulness of feedback on essays using a specified Language Model (LM).

    This function loads a dataset of essays and their corresponding feedback, presents them to the LM alongside a prompt
    asking to evaluate the helpfulness of the feedback, and logs the results.

    Parameters:
        llm: The loaded Language Model to be used for generating outputs.
        config (Config): Configuration object containing the settings for the analysis.
        prompt_template (PromptTemplate): Template object for generating prompts to the LM.

    Results are saved to a specified path in the configuration object.
    """
    with open(config.input_data, "r") as f:
        data = json.load(f)

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[
        ResponseSchema(name="score", description="the helpfulness score of the feedback", type="int"),
    ])

    results = []
    for fold in tqdm(data):
        inputs = [{"essay": item['essay'], "feedback": item['output']} for item in data[fold]]
        outputs = collect_llm_output(llm, inputs, output_parser, prompt_template)
        results += outputs

    # log the data
    with open(config.logging_data_path, "w") as f:
        json.dump(results, f)


def analyse_reference_helpfulness(llm, config, prompt_template):
    """
    Analyzes the helpfulness of feedback on a predefined set of example essays and feedback using a specified LM.

    This function uses a sample dataset (`samples` from `helpfulness_example_dataset.py`) to evaluate feedback helpfulness.
    Each sample consists of an essay and its corresponding feedback. The LM is tasked with assessing the feedback's helpfulness
    based on predefined criteria.

    Parameters:
        llm: The loaded Language Model to be used for generating outputs.
        config (Config): Configuration object containing the settings for the analysis.
        prompt_template (PromptTemplate): Template object for generating prompts to the LM.

    The analysis results are saved to a specified path in the configuration object.
    """
    data = copy.deepcopy(samples)

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas=[
        ResponseSchema(name="score", description="the helpfulness score of the feedback", type="int"),
    ])

    results = collect_llm_output(llm, data, output_parser, prompt_template)

    # log the data
    with open(config.logging_data_path, "w") as f:
        json.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default=None)
    parser.add_argument("--logging_data_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="holistic_scoring_prompt1")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "mistral"])
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b"])
    parser.add_argument("--temperature", type=float, default=0.00)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference_dataset", action="store_true")

    args = parser.parse_args()

    # configure wandb
    load_dotenv()
    config = {
        "temperature": args.temperature,
        "model": args.model,
        "model_size": args.model_size,
        "seed": args.seed,
        "input_data": args.input_data,
        "logging_data_path": args.logging_data_path,
        "prompt": args.prompt,
        "max_length": args.max_length,
        "reference_dataset": args.reference_dataset
    }
    set_seed(args.seed)

    config = Config(**config)

    prompt_template = PromptTemplate(template=template,
                                     input_variables=["format_instructions", "essay", "feedback", "model_prefix",
                                                      "model_suffix"])

    llm, prompt_template = load_model(config, prompt_template)

    if config.reference_dataset:
        analyse_reference_helpfulness(llm, config, prompt_template)
    else:
        analyse_helpfulness(llm, config, prompt_template)


if __name__ == '__main__':
    main()
