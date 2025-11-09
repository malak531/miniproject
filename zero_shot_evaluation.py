import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import unicodedata


# Optional helper if you want logs
class Logger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
    def __call__(self, text):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")

class EvalHumanVsMachine:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", prompt_lang=("ar",), preds_folder="./fs_preds2"):
        self.model_name = model_name
        self.prompt_lang = prompt_lang
        self.preds_folder = preds_folder
        self.separator = "================================================================================="
        self.eos_token = "<｜end▁of▁sentence｜>"

        # prediction folder name
        self.preds_dir = os.path.join(preds_folder, f"{model_name.replace('/', '_')}_human_vs_machine_{prompt_lang}")        
        self.scores_path = os.path.join(self.preds_dir, "scores.txt")

    def get_preds(self):
        txt_files = [
            f for f in os.listdir(self.preds_dir)
            if f.endswith(".txt") and f.split('.')[0].isdigit()
        ]
        txt_files = sorted(txt_files, key=lambda x: int(x.split('.')[0]))

        self.preds = []

        for f in txt_files:
            path = os.path.join(self.preds_dir, f)
            with open(path, encoding="utf-8") as file:
                text = file.read()

            pred = self.extract_answer(text)
            self.preds.append(pred)

    def extract_answer(self, text):
        # 1) Try <answer>...</answer>
        pattern = r"<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>"
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[-1].strip()

        # 2) After 'الإجابة:'
        if "الإجابة:" in text:
            after = text.split("الإجابة:")[-1].strip()
            for line in after.splitlines():
                line = line.strip()
                if line:
                    return line

        # 3) Fallback: last non-empty line
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                return line

        return "<none>"

    # ---------------- Normalize predictions and gold labels ----------------
    def normalize_text(self, text):
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        # remove invisible chars but keep punctuation
        text = text.replace("\u200f", "").replace("\u202c", "").replace("\xa0", " ").strip()
        # remove common trailing punctuation
        text = text.rstrip(".،!؟").strip()
        # map English labels
        mapping = {"human": "بشري", "Human": "بشري", "machine": "آلة", "Machine": "آلة"}
        for k, v in mapping.items():
            if text.lower() == k.lower():
                return v
        return text

    # ---------------- Main classification method ----------------
    def classification(self):
        self.get_preds()

        df = pd.read_csv("test_split.csv")
        self.answers = df['label'].tolist()

        # normalize both preds and answers
        self.preds = [self.normalize_text(p) for p in self.preds]
        self.answers = [self.normalize_text(a) for a in self.answers]

        # debug file
        debug_path = os.path.join(self.preds_dir, "debug_predictions.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("INDEX\tPREDICTION\tGOLD\n")
            f.write("="*60 + "\n")
            for i, (p, g) in enumerate(zip(self.preds, self.answers)):
                f.write(f"{i}\t{repr(p)}\t{repr(g)}\n")
        print(f"📝 Cleaned predictions saved to: {debug_path}")



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

        print("Evaluation Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1: {f1:.4f}")

        return acc, prec, rec, f1


if __name__ == "__main__":
    # Example usage
    e = EvalHumanVsMachine(model_name="Qwen/Qwen2.5-1.5B-Instruct", prompt_lang=("ar",), preds_folder="./fs_preds2")
    e.classification()
