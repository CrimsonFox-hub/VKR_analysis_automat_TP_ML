import re
import pymorphy3
from nltk.corpus import stopwords
import nltk

# Загрузка стоп-слов
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextProcessor:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stop_words = set(stopwords.words('russian'))
        # Системы и паттерны
        self.systems = {
            'лкк': ['лкк', 'личный кабинет', 'личном кабинете', 'лк'],
            'crm': ['crm', 'крм', 'срм'],
            'орппп': ['орппп'],
            'фрешпринт': ['фрешпринт', 'freshprint'],
            'сарфа': ['сарфа', 'sarfa'],
            'биллинг': ['биллинг', 'billing'],
            'оракл': ['оракл', 'oracle'],
            'предбиллинг': ['предбиллинг', 'prebilling'],
            'почта': ['почта', 'email', 'емейл'],
            'веб': ['сайт', 'веб', 'web'],
            'мобильный': ['мобильный', 'приложение'],
            'админка': ['админка', 'админ панель'],
            'отчеты': ['отчет', 'отчеты']
        }
        self.problem_patterns = {
            'пароль': ['пароль', 'password', 'авторизация', 'логин'],
            'доступ': ['доступ', 'вход', 'зайти', 'не заходит'],
            'ошибка': ['ошибка', 'error', 'баг', 'глюк'],
            'данные': ['данные', 'информация', 'сведения'],
            'квитанция': ['квитанция', 'платеж', 'оплата', 'счет'],
            'рассылка': ['рассылка', 'отправка', 'письмо'],
            'показания': ['показания', 'счетчик', 'прибор'],
            'консультация': ['консультация', 'вопрос', 'помощь'],
            'настройка': ['настройка', 'настроить'],
            'не работает': ['не работает', 'не открывается', 'не запускается'],
            'ответ на обращение': ['ответ на обращение', 'отразить ответ', 'внести ответ']
        }

    def text_cleaner(self, text: str) -> str:
        # Очистка текста
        if not text:
            return ""
        # Удаление лишних пробелов и спецсимволов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.strip()
        # Лемматизация
        words = text.lower().split()
        lemmas = []
        for w in words:
            if w not in self.stop_words and len(w) > 2:
                lemma = self.morph.parse(w)[0].normal_form
                lemmas.append(lemma)
        return ' '.join(lemmas)
    
    def clean_text_for_search(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        text = text.strip().lower()
        return text

    def extract_systems(self, text: str) -> list:
        # Извлечение упоминаний систем из текста
        found = []
        text_lower = text.lower()
        for sys_name, patterns in self.systems.items():
            if any(p in text_lower for p in patterns):
                found.append(sys_name)
        return found

    def extract_problems(self, text: str) -> list:
        # Извлечение типов проблем
        found = []
        text_lower = text.lower()
        for prob_name, patterns in self.problem_patterns.items():
            if any(p in text_lower for p in patterns):
                found.append(prob_name)
        return found

    def extract_key_phrases(self, text: str) -> list:
        # Извлечение ключевых фраз
        words = text.lower().split()
        phrases = []
        for w in words:
            if w not in self.stop_words and len(w) > 2:
                parsed = self.morph.parse(w)[0]
                if 'NOUN' in parsed.tag:
                    phrases.append(parsed.normal_form)
        return list(set(phrases))[:5]