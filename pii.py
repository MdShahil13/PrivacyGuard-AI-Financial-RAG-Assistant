import re

def mask_pii(text):
    text = str(text)

    # Email
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)

    # Phone (10 digit)
    text = re.sub(r'\b\d{10}\b', '[PHONE]', text)

    # Aadhaar (12 digit)
    text = re.sub(r'\b\d{12}\b', '[AADHAAR]', text)

    # Names (basic - optional)
    text = re.sub(r'\b[A-Z][a-z]+\b', '[NAME]', text)

    return text