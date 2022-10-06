import pytest

from epigraphhub.data.data_collection.foph import (
    compare_data,
    download_data,
    load_into_db,
)

from epigraphhub.data.data_collection.config import (
    FOPH_CSV_PATH,
)

from epigraphhub.settings import env

tables = [[t, u] for t, u in download_data.get_csv_relation()]

def test_download_csv_files_and_its_existence():
    for table in tables:
        _, url = table
        download_data.download_csv(url)
    import os
    assert len(os.listdir(FOPH_CSV_PATH)) == 14

def test_csv_files_integrity():
    import subprocess
    import os
    for table in tables:
        _, url = table
        filename = str(url).split("/")[-1]
        csv_count = subprocess.Popen(
            f"wc -l {os.path.join(FOPH_CSV_PATH, filename)}",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout    
        assert csv_count > 0

def test_load_csvs_into_db_and_comparison_between_data_update_dates():
    for table in tables:
        tablename, url = table
        filename = str(url).split("/")[-1]
        load_into_db.load(tablename, filename)
        db_last_update = compare_data.table_last_update(tablename)
        csv_last_update = compare_data.csv_last_update(filename)
        assert db_last_update == csv_last_update

def test_delete_csvs_directory():
    import os
    download_data.remove_csvs()
    assert os.path.exists(FOPH_CSV_PATH) == False

