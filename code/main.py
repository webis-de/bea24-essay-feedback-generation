import argparse

from transformers import set_seed

from evaluation_helpers import load_dataset
from llm_functions import analyse_dataset, load_model
from prompts import holistic_scoring_instructions, holistic_scoring_instructions_with_small_rubric, \
    holistic_scoring_instructions_with_full_rubric, explanation_somewhere_instructions, explain_first_instructions, \
    feedback_somewhere_instructions, feedback_first_instructions, feedback_and_explanation_instructions, \
    persona_instructions, chain_of_thought_instructions, chain_of_thought_detailed_instructions, one_shot_instructions, \
    few_shot_instructions, prompt_template_1, prompt_template_2, prompt_template_3, prompt_template_4, \
    feedback_only_instructions, scoreless_prompt_template_1, scoreless_prompt_template_2, scoreless_prompt_template_3, \
    scoreless_prompt_template_4


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logging_data_path", type=str, default=None)
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "mistral"])
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b"])
    parser.add_argument("--temperature", type=float, default=0.00)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--prompt", type=str, default="holistic_scoring_prompt1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--setting", type=str, default="one-shot", choices=["one-shot", "few-shot"])
    parser.add_argument("--full-rubric", action="store_true")
    parser.add_argument("--prompt-template", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--instruction-variant", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    extract_scores, instruction, prompt_template = load_prompt(args)

    if args.full_rubric:
        instruction = instruction.replace("{rubric_small}", "{rubric_full}")

    config = {
        "temperature": args.temperature,
        "max_length": args.max_length,
        "model": args.model,
        "model_size": args.model_size,
        "prompt_template": prompt_template,
        "instruction": instruction,
        "seed": args.seed,
        "dataset_split": args.dataset_split,
        "setting": args.setting
    }
    config = Config(**config)

    set_seed(args.seed)

    llm, prompt_template = load_model(config, prompt_template)

    dataset = load_dataset("./data/datasets/folds")
    analyse_dataset(llm, dataset, instruction, prompt_template, logging_data_path=args.logging_data_path,
                    split=args.dataset_split, setting=config.setting, extract_scores=extract_scores)


def load_prompt(args):
    """
    This function loads the correct prompt template and instruction based on the provided arguments.

    Args:
        args (argparse.Namespace): The arguments parsed from the command line. It should include:
            - prompt_template: An integer representing the prompt template to be used.
            - prompt: A string representing the type of prompt to be used.
            - instruction_variant: An integer representing the variant of instruction to be used.

    Returns:
        tuple: A tuple containing:
            - extract_scores (bool): A boolean indicating whether to extract scores or not.
            - instruction (str): The instruction to be used.
            - prompt_template (str): The prompt template to be used.

    Raises:
        ValueError: If the provided prompt is not supported.
    """
    # load the correct prompt template
    if args.prompt_template == 1:
        prompt_template = prompt_template_1
    elif args.prompt_template == 2:
        prompt_template = prompt_template_2
    elif args.prompt_template == 3:
        prompt_template = prompt_template_3
    elif args.prompt_template == 4:
        prompt_template = prompt_template_4

    # load the correct instruction
    extract_scores = True
    if args.prompt == "holistic_scoring_prompt1":
        instruction = holistic_scoring_instructions[args.instruction_variant - 1]
    elif args.prompt == "holistic_scoring_prompt2":
        instruction = holistic_scoring_instructions_with_small_rubric[args.instruction_variant - 1]
    elif args.prompt == "holistic_scoring_prompt3":
        instruction = holistic_scoring_instructions_with_full_rubric[args.instruction_variant - 1]
    elif args.prompt == "explanation_somewhere_prompt":
        instruction = explanation_somewhere_instructions[args.instruction_variant - 1]
    elif args.prompt == "explanation_first_prompt":
        instruction = explain_first_instructions[args.instruction_variant - 1]
    elif args.prompt == "feedback_somewhere_prompt":
        instruction = feedback_somewhere_instructions[args.instruction_variant - 1]
    elif args.prompt == "feedback_first_prompt":
        instruction = feedback_first_instructions[args.instruction_variant - 1]
    elif args.prompt == "feedback_and_explanation_prompt":
        instruction = feedback_and_explanation_instructions[args.instruction_variant - 1]
    elif args.prompt == "persona_prompt":
        instruction = persona_instructions[args.instruction_variant - 1]
    elif args.prompt == "chain_of_thought_prompt":
        instruction = chain_of_thought_instructions[args.instruction_variant - 1]
    elif args.prompt == "chain_of_thought_detailed_prompt":
        instruction = chain_of_thought_detailed_instructions[args.instruction_variant - 1]
    elif args.prompt == "one_shot_prompt":
        instruction = one_shot_instructions[args.instruction_variant - 1]
    elif args.prompt == "few_shot_prompt":
        instruction = few_shot_instructions[args.instruction_variant - 1]
    elif args.prompt == "feedback_only_prompt":
        instruction = feedback_only_instructions[args.instruction_variant - 1]
        # change the template for scoreless prompts
        if args.prompt_template == 1:
            prompt_template = scoreless_prompt_template_1
        elif args.prompt_template == 2:
            prompt_template = scoreless_prompt_template_2
        elif args.prompt_template == 3:
            prompt_template = scoreless_prompt_template_3
        elif args.prompt_template == 4:
            prompt_template = scoreless_prompt_template_4
        extract_scores = False
    else:
        raise ValueError("Prompt not supported")
    return extract_scores, instruction, prompt_template


if __name__ == '__main__':
    main()
