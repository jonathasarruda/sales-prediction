import pandas as pd

def filter_sales_data(filepath, store_nbr=1, family="BEVERAGES", start="2016-01-01", end="2017-12-31", output_path=None):
    """
    Filters sales data by store, product family, and date range.
    Filtra os dados de vendas por loja, família de produto e intervalo de datas.

    Parameters / Parâmetros:
    - filepath: path to input CSV file / caminho do arquivo CSV de entrada
    - store_nbr: store number / número da loja
    - family: product category / categoria de produto
    - start: start date (YYYY-MM-DD) / data inicial
    - end: end date (YYYY-MM-DD) / data final
    - output_path: optional path to save filtered data / caminho opcional para salvar o subset

    Returns / Retorna:
    - Filtered DataFrame / DataFrame filtrado
    """
    df = pd.read_csv(filepath)
    subset = df[
        (df['store_nbr'] == store_nbr) &
        (df['family'] == family) &
        (df['date'] >= start) &
        (df['date'] <= end)
    ]
    if output_path:
        subset.to_csv(output_path, index=False)
    return subset
