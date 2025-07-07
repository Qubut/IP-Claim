from pprint import pp

import pandas as pd
import polars as pl

df = pd.read_feather('./data/hupd_metadata_2022-02-22.feather')
pp(df)
