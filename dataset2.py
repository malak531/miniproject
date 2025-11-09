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

      # Ensure we have the required columns
      assert "prompt" in df.columns and "label" in df.columns, \
          "CSV must have columns named 'prompt' and 'label'"

      # Add unique ID for each row (so we can track later)
      df["id"] = range(len(df))

      # Split into train/val/test
      train_df, temp_df = train_test_split(
          df,
          test_size=test_size + val_size,
          random_state=random_state,
          stratify=df["label"]
      )

      val_df, test_df = train_test_split(
          temp_df,
          test_size=test_size / (test_size + val_size),
          random_state=random_state,
          stratify=temp_df["label"]
      )

      # Convert to HuggingFace datasets (keep the ID!)
#       N = 50  # for example
#       test_ds = Dataset.from_pandas(test_df.sample(n=N, random_state=777).reset_index(drop=True))
      train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
      val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
      test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

      return DatasetDict({
          "train": train_ds,
          "validation": val_ds,
          "test": test_ds
      })

    
    def sample_few_shot_examples(self, dataset_split, k=3):
        """
        Sample k few-shot examples from the given dataset split (e.g., 'train').
        """
        df = dataset_split.to_pandas()  # Convert HuggingFace dataset to pandas
        samples = df.sample(n=k, random_state=777)
        return list(zip(samples['prompt'], samples['label']))



    def format_for_training(self, dataset_dict, few_shot_examples=None, prompt_style=1, test_mode=False):
        """
        Adds Arabic prompts with optional few-shot examples.
        few_shot_examples: list of (text, label) tuples from train set
        test_mode: if True, only include <answer> tag without the gold label
        """
        question = (
            "صنّف النص التالي بدقة. ضع إجابتك فقط داخل الوسوم <answer>...</answer>. "
            "اكتب كلمة واحدة فقط: 'بشري' أو 'آلة'. لا تضف أي شرح آخر."
        )

        # Build few-shot block
        examples_block = ""
        if few_shot_examples:
            examples_block = "أمثلة:\n"
            for text, label in few_shot_examples:
                # Map label to Arabic
                label_map = {"human": "بشري", "machine": "آلة", "Human": "بشري", "Machine": "آلة",
                             "بشري": "بشري", "مولد": "آلة"}
                mapped_label = label_map.get(label, label)
                examples_block += f"النص: {text}\nالإجابة: <answer>{mapped_label}</answer>\n\n"

        def _format(example):
            label_map = {"human": "بشري", "machine": "آلة", "Human": "بشري", "Machine": "آلة",
                         "بشري": "بشري", "مولد": "آلة"}
            mapped_label = label_map.get(example['label'], example['label'])

            if test_mode:
                # Leave answer empty for test prompts
                prompt = f"{examples_block}{question}\nالنص: {example['prompt']}\nالإجابة: <answer>"
            else:
                # Training prompt includes the gold label
                prompt = f"{examples_block}{question}\nالنص: {example['prompt']}\nالإجابة: <answer>{mapped_label}</answer>"

            return {"text": prompt}

        formatted = {}
        for split, ds in dataset_dict.items():
            formatted[split] = ds.map(_format)
        return DatasetDict(formatted)



        formatted = {}
        for split, ds in dataset_dict.items():
            formatted[split] = ds.map(_format)
        return DatasetDict(formatted)

if __name__ == "__main__":
    ds_builder = HumanVsMachineDataset("data/arabic_llm_detection.csv")
    ds = ds_builder.load_dataset()
    formatted = ds_builder.format_for_training(ds, test_mode=True)


    print("✅ Dataset splits:", ds)
    print("📘 Example formatted text:\n", formatted["train"][0]["text"])


