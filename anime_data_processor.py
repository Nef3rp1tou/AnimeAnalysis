import pandas as pd
"""
    A class to handle loading, cleaning, and preprocessing anime data.

    Methods:
    --------
    load_data(anime_path, ratings_path)
        Load anime and ratings datasets from CSV files.
    clean_data()
        Perform cleaning and preprocessing on the datasets.
    """

class AnimeDataProcessor:
    def __init__(self):
        self.anime_df = None
        self.ratings_df = None
        self.merged_df = None

    def load_data(self, anime_path, ratings_path):
        """Load anime and ratings datasets"""
        self.anime_df = pd.read_csv(anime_path)
        self.ratings_df = pd.read_csv(ratings_path)
        return self.anime_df, self.ratings_df

    def clean_data(self):
        """Clean and preprocess the datasets"""
        self.anime_df = self.anime_df.dropna(subset=['name', 'genre'])
        self.anime_df['genre'] = self.anime_df['genre'].fillna('Unknown')
        self.anime_df['type'] = self.anime_df['type'].fillna('Unknown')
        self.anime_df['episodes'] = pd.to_numeric(self.anime_df['episodes'], errors='coerce').fillna(0)
        self.ratings_df = self.ratings_df[self.ratings_df['rating'] != -1]
        self.merged_df = pd.merge(self.ratings_df, self.anime_df, on='anime_id', how='left')
        self.merged_df = self.merged_df.rename(columns={'rating_x': 'user_rating', 'rating_y': 'average_rating'})
        self.merged_df = self.merged_df.dropna(subset=['name', 'genre', 'user_rating', 'average_rating'])
        self.merged_df = self.merged_df.drop_duplicates(subset=['anime_id', 'name'], keep='first')
        self.remove_outliers('episodes', method='iqr', threshold=1.5)
        self.remove_outliers('members', method='iqr', threshold=1.5)
        return self.merged_df

    def remove_outliers(self, column, method='iqr', threshold=1.5):
        """
        Remove outliers from a specified column using the specified method.
        Args:
            column (str): Column name for which outliers are to be removed.
            method (str): Method for outlier detection ('iqr' or 'zscore').
            threshold (float): Threshold for detecting outliers.
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        if method == 'iqr':
            Q1 = self.merged_df[column].quantile(0.25)
            Q3 = self.merged_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            self.merged_df = self.merged_df[
                (self.merged_df[column] >= lower_bound) & (self.merged_df[column] <= upper_bound)
            ]
        elif method == 'zscore':
            from scipy.stats import zscore
            self.merged_df = self.merged_df[
                (zscore(self.merged_df[column]) > -threshold) & (zscore(self.merged_df[column]) < threshold)
            ]
        return self.merged_df
