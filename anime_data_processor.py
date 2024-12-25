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
        return self.merged_df

    def load_data(self, anime_path, ratings_path):
        try:
            self.anime_df = pd.read_csv(anime_path)
            self.ratings_df = pd.read_csv(ratings_path)
        except FileNotFoundError as e:
            print(f"Error: {e}. Please check the file paths.")
            raise
        except pd.errors.EmptyDataError:
            print("Error: One or both files are empty.")
            raise
        return self.anime_df, self.ratings_df
