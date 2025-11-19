# run_experiments_fewshot.py
import os
from dataset2 import HumanVsMachineDataset
from few_shot_inference import ZS_Inference
from zero_shot_evaluation import EvalHumanVsMachine as Eval

class Args:
    """Unified argument holder for few-shot experiments"""
    def __init__(
        self,
        model="HuggingFaceTB/SmolLM3-3B",
        csv_path="ground_truth.csv",
        prompt_style=1,
        shots=2,  # default few-shot examples
        save_path="./fs_preds2",
        call_limit=720,
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
    # Step 1 — Load dataset
    dataset_builder = HumanVsMachineDataset(csv_path="ground_truth.csv")
    dataset_dict = dataset_builder.load_dataset()

    print(f"Dataset loaded with {len(dataset_dict['train'])} training samples.")

    # Step 2 — Run zero-shot inference
    args = Args()
    inference = ZS_Inference(args)
    inference.run_inference()


    # Step 3 — Run evaluation
    evaluator = Eval(
        model_name=args.model,
        prompt_lang=args.prompt_lang,
        preds_folder=args.save_path
    )
    results = evaluator.classification()

    print("\nEvaluation Results:")
    print(results)

if __name__ == "__main__":
    main()