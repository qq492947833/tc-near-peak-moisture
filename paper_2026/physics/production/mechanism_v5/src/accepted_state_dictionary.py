import pandas as pd
def state(usa,nature,matched):
    usa='' if pd.isna(usa) else str(usa).strip().upper(); nature='' if pd.isna(nature) else str(nature).strip().upper()
    if not matched:return 'C','unmatched'
    if usa in {'TD','TS','TY','ST','TC','HU','HR'}:return 'A','USA_STATUS'
    if usa in {'DB','SD','SS','EX','PT','IN','DS','LO','WV','MD'}:return 'B','USA_STATUS'
    if usa:return 'C','USA_STATUS unknown/ambiguous'
    if nature=='TS':return 'A','NATURE fallback'
    if nature in {'DS','ET','SS'}:return 'B','NATURE fallback'
    return 'C','NATURE missing/unknown/mixed'
