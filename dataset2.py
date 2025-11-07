import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split



class HumanVsMachineDataset:
    def __init__(self, csv_path, eos_token=""):
        self.csv_path = csv_path
        self.eos_token = eos_token

    def load_dataset(self, test_size=0.15, val_size=0.15, random_state=777):
        # Load CSV
        df = pd.read_csv(self.csv_path)

        # Check columns
        assert "prompt" in df.columns and "label" in df.columns, \
            "CSV must have columns named 'prompt' and 'label'"

        # Train/val/test split
        train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=random_state, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=random_state, stratify=temp_df["label"])

        # Convert to HF Datasets
        train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

        return DatasetDict({
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        })
    
    def sample_few_shot_examples(self, k=3):
        df = pd.read_csv(self.csv_path)
        samples = df.sample(n=k, random_state=777)
        return list(zip(samples['prompt'], samples['label']))


    def format_for_training(self, dataset_dict, few_shot_examples=None, prompt_style=1, test_mode=False):
        """
            Adds Arabic prompts with optional few-shot examples.
            prompt_style: integer 1–5 selecting which question phrasing to use.
         """
        prompt_variants = {
            1: "هل النص التالي من إنتاج إنسان أم نموذج لغة كبير؟",
            2: "هل كتب هذا النص شخص حقيقي أم تم توليده آليًا؟",
            3: "اقرأ النص التالي وحدد ما إذا كان بشريًا أم مولدًا من قبل نموذج لغة.",
            4: "تم جمع بعض النصوص من الإنترنت وبعضها من نماذج لغوية. صنّف النص التالي.",
            5: "صِف النص التالي بأنه بشري أو مولد."
        }
        question = prompt_variants.get(prompt_style, prompt_variants[1])

        examples_block = ""
        if few_shot_examples:
            examples_block = "أمثلة:\n"
            for text, label in few_shot_examples:
                examples_block += f"النص: {text}\nالإجابة: {label}\n\n"

        def _format(example):
            prompt = (
                    f"{examples_block}"
                    f"{question}\n"
                    f"النص: {example['prompt']}\n"
                    f"الإجابة: <answer>{example['label'] if not test_mode else ''}</answer>"
                    f"{self.eos_token}"
                )
            return {"text": prompt}

        formatted = {}
        for split, ds in dataset_dict.items():
            formatted[split] = ds.map(_format)
        return DatasetDict(formatted)

if __name__ == "__main__":
    ds_builder = HumanVsMachineDataset("data/arabic_llm_detection.csv")
    ds = ds_builder.load_dataset()
    formatted = ds_builder.format_for_training(ds)

    print("✅ Dataset splits:", ds)
    print("📘 Example formatted text:\n", formatted["train"][0]["text"])


