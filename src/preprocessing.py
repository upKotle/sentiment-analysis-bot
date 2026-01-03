import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^а-яё\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
