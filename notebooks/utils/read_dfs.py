import pandas as pd
from common_var import raw_path, ct_names, race_mapping
from tqdm import tqdm

# return the path to the data folder
def get_data_path(ct_name: str, raw_path: str) -> str:
    # KUMC requires extra suffix
    if ct_name == "KUMC":
        data_path = raw_path + 'KUMC_ORCALE' + '/raw/'
    else:
        data_path = raw_path + ct_name + '/raw/'
    return data_path

###############Demographics data####################

#read patients' demographical data
def read_DEMO(ct_names: list, raw_path: str) -> dict:
    DEMO_dict = dict()
    use_cols = ['ONSETS_ENCOUNTERID', 'AGE', 'PATID', 'SEX', 'RACE']

    for ct_name in ct_names:
        
        data_path = get_data_path(ct_name, raw_path)
        
        if (ct_name == 'UPITT') or (ct_name == 'UTHSCSA') or (ct_name == 'UIOWA') or (ct_name == 'KUMC'):
            DEMO_df = pd.read_csv(data_path + "AKI_DEMO.csv", delimiter = ',', usecols = use_cols)
        elif (ct_name == 'UTSW'):
            DEMO_df = pd.read_csv(data_path + "AKI_DEMO.dsv", delimiter = '|', usecols = use_cols)
        elif (ct_name == 'MCW'):
            DEMO_df = pd.read_csv(data_path + "AKI_DEMO.dsv", delimiter = '|')
            DEMO_df.columns = [col.upper() for col in DEMO_df.columns] 
            DEMO_df = DEMO_df[use_cols]
        elif (ct_name == 'UMHC'):
            DEMO_df = pd.read_csv(data_path + "DEID_AKI_DEMO.csv", delimiter = ',', usecols = use_cols)
        elif (ct_name == 'UofU'):
            DEMO_df = pd.read_csv(data_path + "AKI_DEMO.csv", delimiter = '|', header=None, skiprows = 1, usecols=[0, 1, 2, 5, 17])
            DEMO_df.columns = use_cols
    
        DEMO_df["CENTER_NAME"] = ct_name
        DEMO_dict[ct_name] = DEMO_df
        
    return DEMO_dict

def format_DEMO_dict(DEMO_dict: dict, race_mapping: dict) -> dict:
    processed_DEMO_dict = dict()
    for ct_name, DEMO_df in DEMO_dict.items():    
        #convert id columns to string
        DEMO_df[['ONSETS_ENCOUNTERID', 'PATID']] = DEMO_df[['ONSETS_ENCOUNTERID', 'PATID']].astype(str)
        
        DEMO_df['RACE'] = DEMO_df['RACE'].replace(race_mapping)
        
        DEMO_df["CENTER_NAME"] = ct_name

        processed_DEMO_dict[ct_name] = DEMO_df
    return processed_DEMO_dict

def concat_dfs_to_one(info_dict: dict) -> pd.DataFrame:
    dfs_to_concat = []
    for df in info_dict.values():
        dfs_to_concat.append(df)
    one_df = pd.concat(dfs_to_concat, axis = 0)
    return one_df

def read_and_format_DEMO(ct_names: list, raw_path: str, race_mapping: dict) -> pd.DataFrame:
    DEMO_dict = read_DEMO(ct_names, raw_path)
    processed_DEMO_dict = format_DEMO_dict(DEMO_dict, race_mapping)
    DEMO_df = concat_dfs_to_one(processed_DEMO_dict)
    # drop duplicates to keep only one record per encounter
    DEMO_df.drop_duplicates(subset = ['CENTER_NAME', 'PATID', 'ONSETS_ENCOUNTERID'], inplace = True)
    return DEMO_df

###############SCr data####################

def format_SCR_dict(SCR_dict: dict) -> dict:
    processed_SCR_dict = dict()
    for ct_name, SCR_df in tqdm(SCR_dict.items()):
        SCR_df['PATID'] = SCR_df['PATID'].astype(str)
        SCR_df[['ONSETS_ENCOUNTERID', 'ENCOUNTERID']] = SCR_df[['ONSETS_ENCOUNTERID', 'ENCOUNTERID']].astype(str)
        SCR_df['SPECIMEN_DATE'] = pd.to_datetime(SCR_df['SPECIMEN_DATE'], format='mixed')
        if ct_name == 'UMHC':
            SCR_df['SPECIMEN_DATE'] = SCR_df['SPECIMEN_DATE'].dt.date
            SCR_df['SPECIMEN_DATE'] = pd.to_datetime(SCR_df['SPECIMEN_DATE'])
        SCR_df['CENTER_NAME'] = ct_name
        processed_SCR_dict[ct_name] = SCR_df
    return processed_SCR_dict

#read Scr records, here we kept the historical records(DAYS_SINCE_ADMIT < 0)
def read_SCR(ct_names: list, raw_path: str) -> dict:
    SCR_dict = dict()
    use_cols = ['ONSETS_ENCOUNTERID','PATID','ENCOUNTERID','SPECIMEN_DATE','RESULT_NUM', 'DAYS_SINCE_ADMIT']

    for ct_name in tqdm(ct_names):
        
        data_path = get_data_path(ct_name, raw_path)
        
        if (ct_name == 'UPITT') or (ct_name == 'UTHSCSA') or (ct_name == 'UIOWA'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.csv", delimiter = ',', usecols=use_cols)
        elif (ct_name == 'UTSW'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.dsv", delimiter = '|', usecols=use_cols)
        elif (ct_name == 'MCW'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.dsv", delimiter = '|')
            SCR_df.columns = [col.upper() for col in SCR_df.columns] 
            SCR_df = SCR_df[use_cols]
        elif (ct_name == 'UMHC'):
            SCR_df = pd.read_csv(data_path + "DEID_AKI_LAB_SCR.csv", delimiter = ',', usecols=use_cols)
        elif (ct_name == 'UofU'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.csv", delimiter = '|', usecols=use_cols)
        elif (ct_name == 'KUMC'):
            SCR_df = pd.read_csv(data_path + "AKI_LAB_SCR.csv", delimiter = ',')
            SCR_cols = SCR_df.columns.tolist()
            SCR_cols = [s[:-len('"+PD.DATE_SHIFT"')]  if s.endswith('"+PD.DATE_SHIFT"') else s for s in SCR_cols]
            SCR_df.columns = SCR_cols
            SCR_df = SCR_df[use_cols]

        SCR_dict[ct_name] = SCR_df
        
    return SCR_dict

def read_and_format_SCR(ct_names: list, raw_path: str) -> pd.DataFrame:
    SCR_dict = read_SCR(ct_names, raw_path)
    processed_SCR_dict = format_SCR_dict(SCR_dict)
    SCR_df = concat_dfs_to_one(processed_SCR_dict)
    return SCR_df

###############Diagnose data####################
#centers do not have a DX_DATE: UTHSCSA, UTSW, UofU
def read_DX(ct_names: list, raw_path: str) -> dict:
    DX_dict = dict()
    use_cols = ['ONSETS_ENCOUNTERID', 'PATID', 'DX_DATE', 'DX', 'DX_TYPE', 'DAYS_SINCE_ADMIT']
    ct_missing_DX_DATE = ['UTHSCSA', 'UTSW', 'UofU']
    
    for ct_name in tqdm(ct_names):
        
        data_path = get_data_path(ct_name, raw_path)
        
        if (ct_name == 'UPITT') or (ct_name == 'UTHSCSA') or (ct_name == 'UIOWA'):
            DX_df = pd.read_csv(data_path + "AKI_DX.csv", delimiter = ',', usecols=use_cols)
            
            #adjust the col order of UIOWA
            if ct_name == 'UIOWA':
                DX_df = DX_df[use_cols]
                
        elif (ct_name == 'UTSW'):
            DX_df = pd.read_csv(data_path + "AKI_DX.dsv", delimiter = '|', usecols=use_cols)
            
        elif (ct_name == 'MCW'):
            DX_df = pd.read_csv(data_path + "AKI_DX.dsv", delimiter = '|')
            DX_df.columns = [col.upper() for col in DX_df.columns] 
            DX_df = DX_df[use_cols]
            
        elif (ct_name == 'UMHC'):
            DX_df = pd.read_csv(data_path + "DEID_AKI_DX.csv", delimiter = ',', usecols=use_cols)
            
        elif (ct_name == 'UofU'):
            DX_df = pd.read_csv(data_path + "AKI_DX.csv", delimiter = '|', header=None, skiprows = 1, usecols=[0, 2, 6, 8, 9, 20])
            DX_df.columns = use_cols
            
        elif (ct_name == 'KUMC'):
            DX_df = pd.read_csv(data_path + "AKI_DX.csv", delimiter = ',')
            DX_cols = DX_df.columns.tolist()
            DX_cols = [s[:-len('"+PD.DATE_SHIFT"')] if s.endswith('"+PD.DATE_SHIFT"') else s for s in DX_cols]
            DX_df.columns = DX_cols
            DX_df = DX_df[use_cols]
            
        DX_dict[ct_name] = DX_df
        
    return DX_dict

def format_DX_dict(DX_dict: dict, pat_df: pd.DataFrame) -> dict:
    processed_DX_dict = dict()
    ct_missing_DX_DATE = ['UTHSCSA', 'UTSW', 'UofU']
    
    for ct_name, DX_df in tqdm(DX_dict.items()):
        DX_df['PATID'] = DX_df['PATID'].astype(str)
        pat_ct_df = pat_df[pat_df.CENTER_NAME == ct_name]
        pat_ct_df = pat_ct_df.merge(DX_df[['PATID', 'DX_DATE', 'DX', 'DX_TYPE', 'DAYS_SINCE_ADMIT']], 
                                    on = 'PATID', how = 'left')
        pat_ct_df.dropna(subset=['DX'], inplace = True)
        
        if ct_name not in ct_missing_DX_DATE:
            pat_ct_df['DX_DATE'] = pd.to_datetime(pat_ct_df['DX_DATE'], format = 'mixed')
            pat_ct_df['DX_DATE'] = pat_ct_df['DX_DATE'].dt.strftime('%Y-%m-%d')
            pat_ct_df['DX_DATE'] = pd.to_datetime(pat_ct_df['DX_DATE'], format = 'mixed')
        else:
            pat_ct_df.loc[:, 'DX_DATE'] = pat_ct_df.loc[:, 'ADMIT_DATE'] + \
            pd.to_timedelta(pat_ct_df.loc[:, 'DAYS_SINCE_ADMIT'], unit='D')

        #make type consistent, and make sure it is before the admit date
        pat_ct_df = pat_ct_df[pat_ct_df.DX_DATE < pat_ct_df.ADMIT_DATE]
        pat_ct_df['DX_TYPE'] = pat_ct_df['DX_TYPE'].astype(str)
        pat_ct_df['DX_TYPE'] = pat_ct_df['DX_TYPE'].replace('09', '9')
        pat_ct_df['DX_TYPE'] = pat_ct_df['DX_TYPE'].replace('9.0', '9')
        pat_ct_df['DX_TYPE'] = pat_ct_df['DX_TYPE'].replace('10.0', '10')
        pat_ct_df = pat_ct_df[["CENTER_NAME", "PATID", "ONSETS_ENCOUNTERID", 'DX_DATE', 'DX_TYPE', 'DX']]
        processed_DX_dict[ct_name] = pat_ct_df
        
    return processed_DX_dict

def read_and_format_DX(ct_names: list, raw_path: str, pat_df: pd.DataFrame) -> pd.DataFrame:
    DX_dict = read_DX(ct_names, raw_path)
    processed_DX_dict = format_DX_dict(DX_dict, pat_df)
    DX_df = concat_dfs_to_one(processed_DX_dict)
    # drop duplicates in all columns
    DX_df.drop_duplicates(inplace = True)
    return DX_df

###############Procedures data####################
def read_procedures(ct_names: list, raw_path: str) -> dict:
    PX_dict = dict()
    use_cols = ['PATID', 'PX_DATE', 'PX', 'PX_TYPE']
    
    for ct_name in tqdm(ct_names):
        
        data_path = get_data_path(ct_name, raw_path)
        
        if (ct_name == 'UPITT') or (ct_name == 'UTHSCSA') or (ct_name == 'UIOWA'):
            PX_df = pd.read_csv(data_path + "AKI_PX.csv", delimiter = ',', usecols = use_cols)
            
        elif (ct_name == 'UTSW'):
            PX_df = pd.read_csv(data_path + "AKI_PX.dsv", delimiter = '|', usecols = use_cols)
            
        elif (ct_name == 'MCW'):
            PX_df = pd.read_csv(data_path + "AKI_PX.dsv", delimiter = '|')
            PX_df.columns = [col.upper() for col in PX_df.columns] 
            PX_df = PX_df[use_cols]
            
        elif (ct_name == 'UMHC'):
            PX_df = pd.read_csv(data_path + "DEID_AKI_PX.csv", delimiter = ',', usecols = use_cols)
            
        elif (ct_name == 'UofU'):
            PX_df = pd.read_csv(data_path + "AKI_PX.csv", delimiter = '|', usecols = use_cols)
            
        elif (ct_name == 'KUMC'):
            PX_df = pd.read_csv(data_path + "AKI_PX.csv", delimiter = ',', usecols = ['PATID', 'PX_DATE"+PD.DATE_SHIFT"', 'PX','PX_TYPE'])
            PX_df.columns = use_cols

        PX_dict[ct_name] = PX_df
        
    return PX_dict

###############Duplication check####################
def dup_check(pat_df: pd.DataFrame, id_col_names: list) -> None:
    df_for_check_dup = pat_df.drop_duplicates(subset=id_col_names)
    print("# of rows before dropping dups: ", len(pat_df))
    print("# of rows after dropping dups: ",len(df_for_check_dup))
    assert(len(df_for_check_dup) == len(pat_df))