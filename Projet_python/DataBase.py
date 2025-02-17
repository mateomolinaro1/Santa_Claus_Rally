import yfinance as yf
import pandas as pd
import os


#Classe d'import de données, en fonction du ticker yahoo finance demandé, date de début et de fin
class YahooFinanceAPI:
    def __init__(self, tickers, start_date, end_date):

        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.start_date = start_date
        self.end_date = end_date

    def fetch_prices(self,export_data: bool = True):

        try:
            data = yf.download(
                tickers=self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )

            close_prices = data['Close']

            if close_prices.empty:
                raise ValueError("Aucune donnée récupérée. Vérifiez les tickers ou les dates.")

            if export_data:
                self.export_to_excel(close_prices)

            close_prices.index = pd.to_datetime(close_prices.index)
            return close_prices

        except Exception as e:
            print(f"Erreur lors de la récupération des données : {e}")
            return pd.DataFrame()

    # Exporte le DataFrame en fichier Excel avec un nom basé sur les tickers et les dates
    def export_to_excel(self, df):

        try:

            tickers_str = "_".join(self.tickers)
            filename = f"{tickers_str}_{self.start_date}_{self.end_date}.xlsx"

            output_dir = os.path.join(os.getcwd(), "Data")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, filename)

            df.to_excel(file_path, index=True)
            print(f"Données exportées avec succès vers le fichier : {file_path}")
        except Exception as e:
            print(f"Erreur lors de l'exportation vers Excel : {e}")



# Classe qui gère les données, construit les DataFrames de prix et de rendement
class DataCompute(YahooFinanceAPI):
    def __init__(self, tickers, start_date, end_date,export_data: bool = True):

        super().__init__(tickers, start_date, end_date) #Appel du constructeur de la classe YahooFinanceApi
        self.df_prices = self.fetch_prices(export_data)
        self.df_returns = self.compute_return(self.df_prices)
        self.data_dict = self.compute_data_dict()

    def compute_data_dict(self):
        dict ={}
        for ticker in self.tickers:
            df_price_ticker = pd.DataFrame()
            df_price_ticker[ticker] = self.df_prices[ticker].dropna(how='any',axis= 0)
            df_price_ticker.sort_index()

            df_return_ticker = pd.DataFrame()
            df_return_ticker[ticker] = self.compute_return(df_price_ticker).dropna(how='any', axis=0)
            df_return_ticker.sort_index()

            dict[ticker] = {"df_prices" : df_price_ticker, "df_returns" : df_return_ticker}

        return dict

    def compute_return(self,df):
        df_result = df.pct_change(fill_method=None) #Ajout
        df_result = df_result.iloc[1:]
        df_result.index = df_result.index.strftime('%Y-%m-%d')
        return  df_result

    @staticmethod
    def modif_df(df,start_date,end_date):
        filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
        return filtered_df



class FactorsDataBase:
    def __init__(self, start_date:str = "1998-01-01", end_date:str="2023-12-31", file_name:str=r"Data\Betting Against Beta Equity Factors Daily.xlsx",
                 sheet_names=None, starting_line_data:int = 18):

        self.start_date= start_date
        self.end_date = end_date

        if sheet_names is None:
            sheet_names = ["MKT", "SMB", "HML FF", "HML Devil", "UMD"]

        self.sheet_names = sheet_names
        self.file_name = file_name
        self.starting_line_data = starting_line_data
        self.factors_non_formatted = self.load_data()
        self.factors_formatted = self.format_data()

    def load_data(self):
        factors_dict = {factor: pd.DataFrame() for factor in self.sheet_names}
        for sheet_name in self.sheet_names:
            factors_dict[sheet_name] = pd.read_excel(self.file_name, sheet_name=sheet_name, header=self.starting_line_data)

        return factors_dict

    def format_data(self):
        factors_formatted = {factor: pd.DataFrame for factor in self.sheet_names}
        for sheet_name in self.sheet_names:
            df_current = self.factors_non_formatted[sheet_name]
            # Set index as datetime
            df_current.set_index(pd.to_datetime(df_current.iloc[:,0], format="%m/%d/%Y"), inplace=True)
            # Drop "DATE" column
            df_current.drop(columns="DATE", inplace=True)
            # Aligning all factors to the first all non-missing values
            first_valid_index = df_current.dropna(how="any").index[0]
            df_current = df_current.loc[first_valid_index:]
            # Deleting 1997 and 2024 because not full years
            start_date = pd.to_datetime(self.start_date)
            end_date = pd.to_datetime(self.end_date)
            df_current = df_current.loc[start_date:end_date,:]

            factors_formatted[sheet_name] = df_current

        return factors_formatted