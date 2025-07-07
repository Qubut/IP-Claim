from pprint import pp

import polars as pl

df = pl.read_csv_batched(
    source='./data/g_us_patent_citation.tsv',
    has_header=True,
    separator='\t',
    infer_schema_length=10_000,
)
_df: pl.DataFrame = pl.concat(df.next_batches(10))


pp(
    _df.with_columns(citation_date=pl.col('citation_date').cast(pl.Date)).filter(
        pl.col('citation_date') > pl.date(2017, 12, 30)
    )
)
