import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset2 import HumanVsMachineDataset
from utils import Logger
from huggingface_hub import login


class ZS_Inference:
    def __init__(self, args):
        self.model_name = args.model
        self.csv_path = args.csv_path
        self.prompt_style = args.prompt_style
        self.shots = args.shots
        self.save_path = args.save_path
        self.call_limit = args.call_limit
        self.resume = args.resume
        self.prompt_lang = getattr(args, "prompt_lang", "ar")  # default to "ar" if missing

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
            f"{self.model_name.replace('/', '_')}_human_vs_machine_{self.prompt_lang}"
        )
        os.makedirs(self.preds_file_path, exist_ok=True)

        # ----- LOAD MODEL & DATA -----
        self.load_model()
        self.load_data()

    # ------------------------------------------------------------
    # Load model with MPS/CPU safe options
    # ------------------------------------------------------------
    def load_model(self):
     

        print(f"🦙 Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Use low_cpu_mem_usage to reduce RAM spikes
        # Avoid float16 on CPU
        dtype = torch.float16 if self.device.type == "mps" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=None,  # manual device placement
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

        # Move model to the selected device
        self.model.to(self.device)
        print(f"✅ Model loaded on {self.device}")

    # ------------------------------------------------------------
    # Load dataset & format prompts
    # ------------------------------------------------------------
    def load_data(self):
        print("📘 Loading dataset...")
        ds_builder = HumanVsMachineDataset(self.csv_path)
        ds = ds_builder.load_dataset()

        few_shots = None
        if self.shots > 0:
            few_shots = ds_builder.sample_few_shot_examples(k=self.shots)

        formatted = ds_builder.format_for_training(
            ds,
            few_shot_examples=few_shots,
            prompt_style=self.prompt_style,
            test_mode=True,
        )
        self.dataset = formatted["test"]
        self.dataset_size = len(self.dataset)
        print(f"✅ Loaded {self.dataset_size} samples for inference.")
        # 💾 Save test dataset to a CSV file for evaluation alignment
        import pandas as pd
        test_df = self.dataset.to_pandas() if hasattr(self.dataset, "to_pandas") else pd.DataFrame(self.dataset)
        test_df.to_csv("test_split.csv", index=False)
        print("💾 Saved test split to test_split.csv for evaluation consistency.")


    # ------------------------------------------------------------
    # Run inference locally and save predictions
    # ------------------------------------------------------------
    def run_inference(self):
        print("🚀 Starting zero-shot / few-shot inference...")
        start_idx = 0
        if self.resume and os.path.exists(self.preds_file_path):
            start_idx = len(os.listdir(self.preds_file_path))
            print(f"🔄 Resuming from index {start_idx}")

        for i in tqdm(range(start_idx, min(self.dataset_size, self.call_limit))):
            example = self.dataset[i]
            prompt = example["text"]

            # Move inputs to the same device as model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=32,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Save prediction
            logger = Logger(os.path.join(self.preds_file_path, f"{i}.txt"))
            logger(prompt)
            logger("=" * 80)
            logger(response)
            del inputs, outputs
            if torch.cuda.is_available():
              torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
              torch.mps.empty_cache()

        print(f"✅ Inference complete. Files saved to: {self.preds_file_path}")


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
