raw_path = '/blue/yonghui.wu/hoyinchan/Data/data2022raw/'
ct_names = ['KUMC', 'UPITT']
pat_id_cols = ["CENTER_NAME", "PATID",  "ONSETS_ENCOUNTERID"]
race_mapping = \
{
    '01': 'American Indian or Alaska Native',
    'RACE:amer. indian': 'American Indian or Alaska Native',
    '02': 'Asian',
    'RACE:asian': 'Asian',
    '03': 'Black',
    'RACE:black': 'Black', 
    '04': 'Native Hawaiian',
    'RACE:asian/pac. isl': 'Native Hawaiian',
    'RACE:white': 'White',
    '05': 'White',
    '06': 'More Than One Race',
    '07': 'Other',
    'RACE:ot': 'Patient Refused',
    'OT': 'Patient Refused',
    'NI': 'No Information',
    'RACE:ni': 'No Information',
    'nan': 'No Information',
    'UN': 'Unknown',
    'RACE:unknown':  'Unknown'
}