# run_experiments_fewshot.py
import os
from dataset2 import HumanVsMachineDataset
from zero_shot_inference import ZS_Inference
from zero_shot_evaluation import EvalHumanVsMachine as Eval

class Args:
    """Unified argument holder for few-shot experiments"""
    def __init__(
        self,
        model="HuggingFaceTB/SmolLM3-3B",
        csv_path="ground_truth.csv",
        prompt_style=1,
        shots=3,  # default few-shot examples
        save_path="./fs_preds",
        call_limit=50,
        prompt_lang="ar",
        resume=False,
    ):
        self.model = model
        self.csv_path = csv_path
        self.prompt_style = prompt_style
        self.shots = shots
        self.save_path = save_path
        self.call_limit = call_limit
        self.prompt_lang = prompt_lang
        self.resume = resume

def main():
    # ----------------------------
    # Step 1 — Load dataset
    # ----------------------------
    dataset_builder = HumanVsMachineDataset(csv_path="ground_truth.csv")
    dataset_dict = dataset_builder.load_dataset()
    print(f"Dataset loaded: {len(dataset_dict['train'])} train, "
          f"{len(dataset_dict['validation'])} validation, {len(dataset_dict['test'])} test samples")

    # ----------------------------
    # Step 2 — Sample few-shot examples from training set
    # ----------------------------
    args = Args()
    few_shot_examples = None
    if args.shots > 0:
        few_shot_examples = dataset_builder.sample_few_shot_examples(dataset_dict['train'], k=args.shots)
        print(f"Using {args.shots} few-shot examples:")
        for i, (text, label) in enumerate(few_shot_examples):
            print(f"{i+1}. {label}: {text[:60]}...")

    # ----------------------------
    # Step 3 — Format test dataset with few-shot prompts
    # ----------------------------
    formatted = dataset_builder.format_for_training(
        dataset_dict,
        few_shot_examples=few_shot_examples,
        prompt_style=args.prompt_style,
        test_mode=True
    )
    test_dataset = formatted["test"]



    # ----------------------------
    # Step 4 — Run inference
    # ----------------------------
    inference = ZS_Inference(args)


    inference.run_inference()

    # ----------------------------
    # Step 5 — Run evaluation
    # ----------------------------
    evaluator = Eval(
        model_name=args.model,
        prompt_lang=args.prompt_lang,
        preds_folder=args.save_path
    )
    results = evaluator.classification()
    print("\nFinal Evaluation Results:")
    print(results)

if __name__ == "__main__":
    main()
