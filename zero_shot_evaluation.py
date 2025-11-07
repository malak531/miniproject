import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional helper if you want logs
class Logger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
    def __call__(self, text):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

class EvalHumanVsMachine:
    def __init__(self, model_name="LLaMA3-3B", prompt_lang="ar", preds_folder="./zs_preds"):
        self.model_name = model_name
        self.prompt_lang = prompt_lang
        self.preds_folder = preds_folder
        self.separator = "================================================================================="
        self.eos_token = "<｜end▁of▁sentence｜>"

        # prediction folder name
        self.preds_dir = os.path.join(preds_folder, f"{model_name.replace('/', '_')}_human_vs_machine_{prompt_lang}")
        self.scores_path = os.path.join(self.preds_dir, "scores.txt")

    def get_preds(self):
      txt_files = sorted([f for f in os.listdir(self.preds_dir) if f.endswith(".txt") and f != "scores.txt"],
                        key=lambda x: int(x.split('.')[0]))

      self.preds = []
      self.answers = []

      for f in txt_files:
        with open(os.path.join(self.preds_dir, f), encoding="utf-8") as pred_file:
            text = pred_file.read()

        # Extract the gold label (from your test set filename order)
        # Assuming ground truth is available in same order
        # If your ground truth is in CSV, we’ll match it separately later.
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if answer_match:
            pred = answer_match.group(1).strip()
            self.preds.append(pred)
        else:
            self.preds.append("<none>")

        # For now, we’ll load the gold label directly from your CSV later


    def classification(self):
        self.get_preds()

        import pandas as pd
        df = pd.read_csv("test_split.csv")
        self.answers = df['label'].tolist()



        # Clean text
        self.preds = [p.replace(self.eos_token, "").replace("\n", "").strip() for p in self.preds]
        self.answers = [a.replace(self.eos_token, "").replace("\n", "").strip() for a in self.answers]

        # Normalize possible variations of labels
        label_map = {
            "بشري": "بشري",
            "آلة": "آلة",
            "human": "بشري",
            "machine": "آلة",
            "Human": "بشري",
            "Machine": "آلة"
        }
        self.preds = [label_map.get(p, p) for p in self.preds]
        self.answers = [label_map.get(a, a) for a in self.answers]

        return self.calculate_F1(self.preds, self.answers)

    def calculate_F1(self, preds, answers):
        acc = accuracy_score(answers, preds)
        prec = precision_score(answers, preds, average='macro', zero_division=0)
        rec = recall_score(answers, preds, average='macro', zero_division=0)
        f1 = f1_score(answers, preds, average='macro', zero_division=0)

        logger = Logger(self.scores_path)
        logger(f"Accuracy: {acc:.4f}")
        logger(f"Precision: {prec:.4f}")
        logger(f"Recall: {rec:.4f}")
        logger(f"F1 Score: {f1:.4f}")

        print("✅ Evaluation Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1: {f1:.4f}")

        return acc, prec, rec, f1


if __name__ == "__main__":
    # Example usage
    e = EvalHumanVsMachine(model_name="LLaMA3-3B", prompt_lang="ar", preds_folder="./zs_preds")
    e.classification()
