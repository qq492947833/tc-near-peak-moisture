import numpy as np
import pandas as pd

def args_from_group(sid,g):
    name=g.NAME.dropna().astype(str);season=pd.to_numeric(g.SEASON,errors='coerce').dropna()
    args={'time':g.ISO_TIME.to_numpy(),'lat':g.LAT.to_numpy(),'lon':g.LON_180.to_numpy(),'wind':g.WIND.to_numpy(),'pressure':g.PRES.to_numpy(),'basin':g.BASIN.to_numpy(),'is_tropical':g.IS_TROPICAL.to_numpy(),'dist2land':g.DIST2LAND.to_numpy(),'landfall':g.LANDFALL.to_numpy(),'wind_filled_flag':g.WIND_WAS_FILLED.to_numpy(),'sid':sid,'name':name.iloc[0] if len(name) else '', 'season':float(season.iloc[0]) if len(season) else np.nan}
    return args
