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
        self.preds_dir = os.path.join(preds_folder, f"{model_name}_human_vs_machine_{prompt_lang}")
        self.scores_path = os.path.join(self.preds_dir, "scores.txt")

    def get_preds(self):
        txt_files = sorted([f for f in os.listdir(self.preds_dir) if f.endswith(".txt") and f != "scores.txt"],
                           key=lambda x: int(x.split('.')[0]))

        self.preds = []
        self.answers = []

        for f in txt_files:
            with open(os.path.join(self.preds_dir, f), encoding="utf-8") as pred_file:
                lines = pred_file.readlines()

            # Find answer bounds
            bounds = [i for i, line in enumerate(lines) if line.strip() == self.separator]
            if len(bounds) < 2:
                continue

            gold = " ".join(lines[bounds[0]+1: bounds[1]]).strip()
            self.answers.append(gold.replace("\n", ""))

            pred_text = " ".join(lines[bounds[1]+1:]).strip()
            # Extract only text after </think>, if present
            think_match = re.search(r"</think>(.*)", pred_text, re.DOTALL)
            if think_match:
                pred_text = think_match.group(1).strip()

            # Extract <answer>...</answer>
            answer_match = re.search(r"<answer>(.*?)</answer>", pred_text, re.DOTALL)
            pred = answer_match.group(1).strip() if answer_match else "<none>"
            self.preds.append(pred)

    def classification(self):
        self.get_preds()
        self.answers = self.answers[:len(self.preds)]

        # Clean text
        self.preds = [p.replace(self.eos_token, "").replace("\n", "").strip() for p in self.preds]
        self.answers = [a.replace(self.eos_token, "").replace("\n", "").strip() for a in self.answers]

        # Normalize possible variations of labels
        label_map = {
            "بشري": "بشري",
            "مولد": "مولد",
            "human": "بشري",
            "machine": "مولد",
            "Human": "بشري",
            "Machine": "مولد"
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
