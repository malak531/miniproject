import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset2 import HumanVsMachineDataset
from utils import Logger
from huggingface_hub import login
from peft import PeftModel



class ZS_Inference:
    def __init__(self, args):
        self.model_name = args.model
        self.csv_path = args.csv_path
        self.prompt_style = args.prompt_style
        self.shots = args.shots
        self.save_path = args.save_path
        self.call_limit = args.call_limit
        self.resume = args.resume
        self.prompt_lang = "ar",
        self.batch_size = args.batch_size


        # ----- DEVICE SELECTION -----
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # ----- CREATE OUTPUT DIRS -----
        os.makedirs(self.save_path, exist_ok=True)
        self.preds_file_path = os.path.join(
            self.save_path,
            f"{self.model_name.replace('/', '_')}_human_vs_machine_ar"
        )
        os.makedirs(self.preds_file_path, exist_ok=True)

        # ----- LOAD MODEL & DATA -----
        self.load_model()
        self.load_data()

    # ------------------------------------------------------------
    # Load model with MPS/CPU safe options
    # ------------------------------------------------------------
# inside ZS_Inference class
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        print(f" Loading model: {self.model_name}")
        access_token = ""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True,token=access_token)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        

        # Determine dtype for GPU
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Some models (Qwen, InternLM, etc.) require trust_remote_code=True
        trust_remote_code = any(x in self.model_name.lower() for x in ["qwen", "internlm"])


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code = True,
            device_map = "auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            token=access_token
        )


        # Move to GPU explicitly if device_map didn't already do it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If a LoRA adapter path is provided, load it
#         if self.lora_path:
#             print(f"Loading LoRA adapter from {self.lora_path}")
#             self.model = PeftModel.from_pretrained(self.model, self.lora_path)

        # Move to the selected device
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on device: {self.device}")


    # ------------------------------------------------------------
    # Load dataset & format prompts
    # ------------------------------------------------------------
    def load_data(self):
        print("Loading dataset...")
        ds_builder = HumanVsMachineDataset(self.csv_path)
        ds = ds_builder.load_dataset()

        few_shots = None
        if self.shots > 0:
            few_shots = ds_builder.sample_few_shot_examples(ds['train'], k=self.shots)

        formatted = ds_builder.format_for_training(
            ds,
            few_shot_examples=few_shots,
            prompt_style=self.prompt_style,
            test_mode=True,
        )
        self.dataset = formatted["test"]
        self.dataset_size = len(self.dataset)
        print(f"Loaded {self.dataset_size} samples for inference.")
        # 💾 Save test dataset to a CSV file for evaluation alignment
        import pandas as pd
        test_df = self.dataset.to_pandas() if hasattr(self.dataset, "to_pandas") else pd.DataFrame(self.dataset)
        test_df.to_csv("test_split.csv", index=False)
        
        
        
#         self.dataset = formatted["validation"]
#         self.dataset_size = len(self.dataset)
#         print(f"Loaded {self.dataset_size} samples for validation.")
#         # 💾 Save test dataset to a CSV file for evaluation alignment
#         import pandas as pd
#         test_df = self.dataset.to_pandas() if hasattr(self.dataset, "to_pandas") else pd.DataFrame(self.dataset)
#         test_df.to_csv("test_split.csv", index=False)
        
        print("Saved test split to test_split.csv for evaluation consistency.")


    # ------------------------------------------------------------
    # Run inference locally and save predictions
    # ------------------------------------------------------------
    def run_inference(self):
        print("Starting batched inference...")
        batch_size = getattr(self, "batch_size", 8)  # default if missing
        start_idx = 0
        if self.resume and os.path.exists(self.preds_file_path):
            start_idx = len(os.listdir(self.preds_file_path))
            print(f"Resuming from index {start_idx}")

        # Iterate over dataset in batches
        for batch_start in tqdm(range(start_idx, min(self.dataset_size, self.call_limit), batch_size)):
            batch_end = min(batch_start + batch_size, self.dataset_size)
            batch = self.dataset[batch_start:batch_end]
            # If dataset slicing returns a dict of lists
            if isinstance(batch, dict):
                prompts = batch["text"]
            else:
                prompts = [example["text"] for example in batch]


            # Tokenize batch with padding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding = True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    use_cache=False,
                    max_new_tokens=32,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Save predictions
            for i, response in enumerate(decoded):
                logger = Logger(os.path.join(self.preds_file_path, f"{batch_start + i}.txt"))
                logger(prompts[i])
                logger("=" * 80)
                logger(response)

            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

        print(f"Inference complete. Files saved to: {self.preds_file_path}")



# ------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", default="tiiuae/Falcon3-1B-Instruct")
    parser.add_argument("--csv_path", dest="csv_path", default="ground_truth.csv")
    parser.add_argument("--prompt_style", dest="prompt_style", type=int, default=1)
    parser.add_argument("--shots", dest="shots", type=int, default=0)
    parser.add_argument("--save_path", dest="save_path", default="./zs_preds")
    parser.add_argument("--call_limit", dest="call_limit", type=int, default=1000)
    parser.add_argument("--resume", dest="resume", type=int, default=0)
    parser.add_argument("--prompt_lang", dest="prompt_lang", type=str, default="ar")
    args = parser.parse_args()

    zs = ZS_Inference(args)
    zs.run_inference()
