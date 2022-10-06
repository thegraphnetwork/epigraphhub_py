import pytest

from epigraphhub.data.data_collection.owid import (
    compare_data,
    download_data,
    load_into_db,
)

from epigraphhub.data.data_collection.config import (
    OWID_CSV_PATH,
    OWID_FILENAME,
)

from epigraphhub.settings import env


def test_download_csv_file_and_its_existence():
    import os
    download_data.download_csv()
    assert os.path.exists(f"{OWID_CSV_PATH}/{OWID_FILENAME}")

def test_csv_file_integrity_size_shouldnt_be_zero():
    size = compare_data.csv_size()
    assert size > 0

def test_load_into_db():
    load_into_db.load(remote=False)
    csv_size = compare_data.csv_size()
    db_size = compare_data.database_size(remote=False)
    assert csv_size == db_size

def test_delete_csv():
    import os
    download_data.remove_csv()
    assert os.path.exists(f"{OWID_CSV_PATH}/{OWID_FILENAME}") == False