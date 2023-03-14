import unicodedata

from pysus.online_data import FTP_SINAN

DISEASES = FTP_SINAN.diseases


def normalize_str(disease: str) -> str:
    """
    Animais Peçonhentos -> animais_peconhentos
    """
    non_ascii = (
        unicodedata.normalize("NFKD", disease).encode("ascii", "ignore").decode()
    )
    disease = non_ascii.lower().replace(" ", "_")
    return disease
