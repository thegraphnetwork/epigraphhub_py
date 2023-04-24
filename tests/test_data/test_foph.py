import pandas as pd
import pytest

from epigraphhub.data.foph import extract


def test_extracting_metadata_tables():
    metadata_tables = list(extract.metadata().table)

    expected_tables = [
        'DailyIncomingData',
        'DailyCasesVaccPersonsIncomingData',
        'DailyIncomingData',
        'DailyHospVaccPersonsIncomingData',
        'DailyIncomingData',
        'DailyDeathVaccPersonsIncomingData',
        'DailyIncomingData',
        'DailyIncomingData',
        'WeeklyIncomingData',
        'WeeklyIncomingData',
        'WeeklyIncomingData',
        'WeeklyIncomingData',
        'WeeklyIncomingData',
        'WeeklyIncomingData',
        'WeeklyCasesVaccPersonsAgeRangeIncomingData',
        'WeeklyIncomingData',
        'WeeklyHospReasonIncomingData',
        'WeeklyDeathReasonIncomingData',
        'WeeklyHospVaccPersonsAgeRangeIncomingData',
        'WeeklyIncomingData',
        'WeeklyDeathVaccPersonsAgeRangeIncomingData',
        'WeeklyDeathBreakthroughVaccPersonsAgeRangeIncomingData',
        'WeeklyIncomingData',
        'WeeklyIncomingData',
        'WeeklyCasesVaccPersonsSexIncomingData',
        'WeeklyIncomingData',
        'WeeklyHospVaccPersonsSexIncomingData',
        'WeeklyIncomingData',
        'WeeklyDeathVaccPersonsSexIncomingData',
        'WeeklyIncomingData',
        'WeeklyReportIncomingData',
        'SentinellaWeeklyVirusTypesData',
        'SentinellaWeeklyConsulationsData',
        'AdditionalGeoRegionDailyIncomingData',
        'AdditionalGeoRegion14dPeriodIncomingData',
        'DailyReportIncomingData',
        'ContactTracingIncomingData',
        'HospCapacityDailyIncomingData',
        'HospCapacityCertStatusIncomingData',
        'HospCapacityWeeklyIncomingData',
        'InternationalQuarantineIncomingData',
        'InternationalDailyIncomingData',
        'ReDailyIncomingData',
        'VaccinationIncomingData',
        'VaccinationDosesReceivedDeliveredVaccineIncomingData',
        'VaccinationIncomingData',
        'VaccinationVaccineIncomingData',
        'VaccPersonsIncomingData',
        'VaccPersonsVaccineIncomingData',
        'VaccinationWeeklyIncomingData',
        'VaccPersonsWeeklyIncomingData',
        'VaccPersonsWeeklyAgeRangeVaccineIncomingData',
        'VaccinationWeeklyIncomingData',
        'VaccPersonsWeeklyIncomingData',
        'VaccPersonsWeeklyIndicationIncomingData',
        'VaccinationWeeklyIndicationIncomingData',
        'VaccinationWeeklyLocationIncomingData',
        'VaccinationSymptomsIncomingData',
        'VaccinationContingentIncomingData',
        'VirusVariantsWgsDailyIncomingData',
        'VirusVariantsHospWeeklyIncomingData',
        'CovidCertificatesDailyIncomingData',
        'DailyEpiRawIncomingData',
        'DailyCasesAgeRangeRawIncomingData',
        'WeeklyEpiAgeRangeSexRawIncomingData',
        'PopulationAgeRangeSexData',
        'WasteWaterDailyViralLoadData',
        'WasteWaterViralLoadOverview',
        'WeeklyHospBreakthroughVaccPersonsAgeRangeIncomingData'
    ]

    assert all([t in metadata_tables for t in expected_tables])

    half_of_amt = int(len(expected_tables)/2)
    for tablename in expected_tables[half_of_amt:]: # takes too long
        df = extract.metadata(table=tablename)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

def test_fetching_tables_for_daily_data():
    dataset_tables = [
        t for t,url in extract.fetch(freq='daily')
    ]
    expected_tables = [
        'cases',
        'casesVaccPersons',
        'hosp',
        'hospVaccPersons',
        'death',
        'deathVaccPersons',
        'test',
        'testPcrAntigen',
        'hospCapacity',
        'hospCapacityCertStatus',
        're',
        'intCases',
        'virusVariantsWgs',
        'covidCertificates'
    ]
    assert all(map(lambda t: t in dataset_tables, expected_tables))

def test_fetching_tables_for_weekly_data_default():
    dataset_tables = [
        t for t,url in extract.fetch(freq='weekly')
    ]
    expected_tables = [
        'cases',
        'hosp',
        'death',
        'test',
        'detections',
        'consultations',
        'testPcrAntigen',
        'hospCapacity',
        'virusVariantsHosp'
    ]
    assert all(map(lambda t: t in dataset_tables, expected_tables))

def test_fetching_tables_for_weekly_data_by_age():
    dataset_tables = [
        t for t,url in extract.fetch(freq='weekly', by='age')
    ]
    expected_tables = [
        'cases',
        'casesVaccPersons',
        'hosp',
        'hospReason',
        'hospVaccPersons',
        'hospBreakthroughVaccPersons',
        'death',
        'deathReason',
        'deathVaccPersons',
        'deathBreakthroughVaccPersons',
        'test'
    ]
    assert all(map(lambda t: t in dataset_tables, expected_tables))

def test_fetching_tables_for_weekly_data_by_sex():
    dataset_tables = [
        t for t,url in extract.fetch(freq='weekly', by='sex')
    ]
    expected_tables = [
        'cases',
        'casesVaccPersons',
        'hosp',
        'hospVaccPersons',
        'death',
        'deathVaccPersons',
        'test'
    ]
    assert all(map(lambda t: t in dataset_tables, expected_tables))

def test_downloading_daily_csv_cases():
    _, url = [x for x in extract.fetch()][0]
    file = extract.download(url)

    assert file == '/tmp/foph/releases/COVID19Cases_geoRegion.csv'

    df = pd.read_csv(file)

    expected_columns = [
        'geoRegion', 'datum', 'entries',
        'sumTotal', 'timeframe_14d', 'timeframe_all',
        'offset_last7d', 'sumTotal_last7d', 'offset_last14d',
        'sumTotal_last14d', 'offset_last28d', 'sumTotal_last28d',
        'sum7d', 'sum14d', 'mean7d', 'mean14d', 'entries_diff_last_age',
        'pop', 'inz_entries', 'inzsumTotal', 'inzmean7d', 'inzmean14d',
        'inzsumTotal_last7d', 'inzsumTotal_last14d', 'inzsumTotal_last28d',
        'inzsum7d', 'inzsum14d', 'sumdelta7d', 'inzdelta7d', 'type',
        'type_variant', 'version', 'datum_unit', 'entries_letzter_stand',
        'entries_neu_gemeldet', 'entries_diff_last'
    ]

    assert all(map(lambda c: c in df.columns, expected_columns))

    del df


def test_downloading_weekly_csv_default_cases():
    _, url = [x for x in extract.fetch(freq='weekly')][0]
    file = extract.download(url)

    assert file == '/tmp/foph/releases/COVID19Cases_geoRegion_w.csv'

    df = pd.read_csv(file)

    expected_columns = [
        'geoRegion', 'datum', 'entries', 'timeframe_all', 'sumTotal', 'freq',
       'prct', 'pop', 'inz_entries', 'inzsumTotal', 'type', 'type_variant',
       'version', 'datum_unit', 'datum_dboardformated', 'entries_diff_abs',
       'entries_diff_inz', 'entries_diff_pct', 'prct_diff', 'timeframe_2w',
       'timeframe_4w', 'sum2w', 'sum4w', 'mean2w', 'mean4w', 'sumTotal_last2w',
       'sumTotal_last4w', 'inzmean2w', 'inzmean4w', 'inzsumTotal_last2w',
       'inzsumTotal_last4w', 'inzsum2w', 'inzsum4w', 'entries_letzter_stand',
       'entries_neu_gemeldet', 'entries_diff_last_age', 'entries_diff_last'
    ]

    assert all(map(lambda c: c in df.columns, expected_columns))

    del df

def test_downloading_weekly_csv_by_age_cases():
    _, url = [x for x in extract.fetch(freq='weekly', by='age')][0]
    file = extract.download(url)

    assert file == '/tmp/foph/releases/COVID19Cases_geoRegion_AKL10_w.csv'

    df = pd.read_csv(file)

    expected_columns = [
        'altersklasse_covid19', 'geoRegion', 'datum', 'entries',
        'timeframe_all', 'sumTotal', 'freq', 'prct', 'pop', 'inz_entries',
        'inzsumTotal', 'type', 'type_variant', 'version', 'datum_unit',
        'datum_dboardformated', 'entries_diff_abs', 'entries_diff_inz',
        'entries_diff_pct', 'prct_diff'
    ]

    assert all(map(lambda c: c in df.columns, expected_columns))

    del df

def test_downloading_weekly_csv_by_sex_cases():
    _, url = [x for x in extract.fetch(freq='weekly', by='sex')][0]
    file = extract.download(url)

    assert file == '/tmp/foph/releases/COVID19Cases_geoRegion_sex_w.csv'

    df = pd.read_csv(file)

    expected_columns = [
        'sex', 'geoRegion', 'datum', 'entries', 'timeframe_all', 'sumTotal',
        'freq', 'prct', 'pop', 'inz_entries', 'inzsumTotal', 'type',
        'type_variant', 'version', 'datum_unit', 'datum_dboardformated',
        'entries_diff_abs', 'entries_diff_inz', 'entries_diff_pct',
        'prct_diff'
    ]

    assert all(map(lambda c: c in df.columns, expected_columns))

    del df

def test_removing_all_downloaded_data():
    extract.remove(entire_dir=True)
