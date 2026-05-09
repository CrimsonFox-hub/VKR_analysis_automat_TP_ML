from .text_processor import TextProcessor
import pandas as pd

class TaskTagger:
    def __init__(self):
        self.processor = TextProcessor()

    def generate_tags(self, text, category):
        # Генерация тегов на основе текста и категории
        if pd.isna(text) or not text:
            return ""
            
        systems = self.processor.extract_systems(text)
        problems = self.processor.extract_problems(text)
        phrases = self.processor.extract_key_phrases(text)
        all_tags = []
        
        all_tags.extend(systems)
        all_tags.extend(problems)
        all_tags.extend(phrases)

        unique_tags = []
        for tag in all_tags:
            if (tag not in unique_tags and 
                len(tag) > 2 and len(tag) < 30 and
                not any(stop in tag for stop in ['добрый', 'день', 'пожалуйста', 'спасибо'])):
                unique_tags.append(tag)
        
        return ', '.join(unique_tags[:4])