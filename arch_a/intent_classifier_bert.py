import torch
from transformers import AutoTokenizer, ModelSeqClassification
from pathlib import Path
from shared.config import MODELS_DIR

# Оценка с использованием BERT, основа
class BERTIntentClassifier:
    # Инициализация
    def __init__(self, model_path: Path = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_path and model_path.exists():
            self.model = ModelSeqClassification.from_pretrained(str(model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        else:
            self.model = ModelSeqClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=5)
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self.model.to(self.device)
        self.model.eval()

    # Предсказание/оценка
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()
        return pred_class, confidence