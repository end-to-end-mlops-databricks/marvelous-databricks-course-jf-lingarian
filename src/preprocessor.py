import polars as pl
import datetime as dt
import os

class DataProcessor:
    """
    A class used to process data files.

    Attributes:
        file_path (str): The path to the directory containing the file.
        file_name (str): The name of the file to be processed.
    """

    def __init__(self, file_path: str, file_name: str):
        """
        Initializes the DataProcessor with the specified file path and file name.

        Args:
            file_path (str): The path to the directory containing the file.
            file_name (str): The name of the file to be processed.
        """
        self.file_path = file_path
        self.file_name = file_name
        self.preprocessed_data_file_path = f'data/preprocessed/{file_name}.parquet'

    def process_files(self):
        """
        Process files in the specified directory that match the file name pattern.

        Lists all files in the specified directory, filters for .csv files that match the file name pattern,
        and calls the merge_new_data_with_preprocessed_data method for each matching file.
        """
        # List all files in the specified directory
        files = os.listdir(self.file_path)

        # Filter and process the names of .csv files
        for file in files:
            if file.startswith(self.file_name) and file.endswith('.csv'):
                print(file)
                self.merge_new_data_with_preprocessed_data(file)
        return pl.read_parquet(self.preprocessed_data_file_path)

    def prepare_raw_csv(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare the raw CSV data by transforming and reshaping it.

        Args:
            df (pl.DataFrame): The input DataFrame containing raw data.

        Returns:
            pl.DataFrame: The transformed DataFrame with categorical columns and a unique identifier.
        """
        return (
            df
            .with_columns(
                [
                    pl.col("Client").cast(pl.Utf8).cast(pl.String),
                    pl.col("Warehouse").cast(pl.Utf8).cast(pl.String),
                    pl.col("Product").cast(pl.Utf8).cast(pl.String),
                    pl.concat_str([pl.col('Client'), pl.lit('/'), pl.col('Warehouse'), pl.lit('/'), pl.col('Product')]).cast(pl.String).alias('unique_id')
                ]
            )
            .unpivot(index=['unique_id','Product','Warehouse','Client'], variable_name='ds', value_name='y')
            [['ds','y','unique_id','Client','Warehouse','Product']]
            .with_columns(pl.col('ds').cast(pl.Date))
            .sort(by=['unique_id', 'ds'])
        )

    def merge_new_data_with_preprocessed_data(
        self,
        new_data_file_name: str,
        original_cat_cols: list = ['Client','Warehouse','Product']      
    ) -> None:
        """
        Merge new data with preprocessed data and save the result.

        Args:
            new_data_file_name (str): The name of the new data file (without extension).
            preprocessed_data_file_path (str, optional): The path to the preprocessed data file. Defaults to 'data/preprocessed/sales.parquet'.
            original_cat_cols (list, optional): The list of original categorical columns. Defaults to ['Client','Warehouse','Product'].

        Returns:
            None
        """
        # Read new file
        new_df = pl.read_csv(f'data/raw/{new_data_file_name}').with_columns([pl.col('Client').cast(pl.String).cast(pl.Categorical), pl.col('Warehouse').cast(pl.String).cast(pl.Categorical), pl.col('Product').cast(pl.String).cast(pl.Categorical)])

        # Read the preprocessed data
        old_df = pl.read_parquet(self.preprocessed_data_file_path)

        # Get only date columns from the new files
        dates_from_new_file = [col for col in new_df.columns if col not in original_cat_cols]

        # Convert the date columns to datetime format
        date_from_new_file_datetime_format = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates_from_new_file]

        # Get the unique dates from the old file
        dates_from_old_file = old_df['ds'].unique().to_list()

        # Get the dates that are not overlapping
        dates_that_are_not_overlapping = [col for col in date_from_new_file_datetime_format if col not in dates_from_old_file]

        if not dates_that_are_not_overlapping:
            print('All dates are overlapping')
        else:
            # Process the new data
            new_df_processed = (
                self.prepare_raw_csv(new_df)
                .with_columns(pl.col('y').cast(pl.Float64))
                .filter(pl.col('ds').is_in(dates_that_are_not_overlapping))
            )

            with pl.StringCache():
                # Append the new data to the old data, sort and write to parquet
                (
                    pl
                    .concat([old_df, new_df_processed], how='vertical')
                    .sort(by=['unique_id', 'ds'])
                    .write_parquet(self.preprocessed_data_file_path)
                )
                
            print(f"Data for file {new_data_file_name} has been successfully merged with the preprocessed data")
