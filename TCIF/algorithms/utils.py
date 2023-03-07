import numpy as np
from tqdm.auto import tqdm


def preare(df, tid_list, verbose=True):
    id = []
    lat = []
    lon = []
    time = []
    classe = []

    for _id, _classe in tqdm(df[df.tid.isin(tid_list)][["tid", "class"]].groupby(
            by=["tid", "class"]).max().reset_index().values, disable=not verbose, desc="Preparing data"):
        df_result = df[df.tid == _id]
        id.append(_id)
        classe.append(_classe)

        _lat = []
        _lon = []
        _time = []

        start_t = df_result.t.min()

        for _lat_el, _lon_el, _time_el in df_result[["c1", "c2", "t"]].values:
            if _time_el-start_t in _time:
                continue
            _time.append(_time_el-start_t)
            _lat.append(_lat_el)
            _lon.append(_lon_el)

        lat.append(np.array(_lat))
        lon.append(np.array(_lon))
        time.append(np.array(_time))

    return id, classe, lat, lon, time