from anime_data_processor import AnimeDataProcessor
from anime_eda import AnimeEDA
from anime_ml import AnimeML

import warnings
warnings.filterwarnings('ignore')

def main():
    processor = AnimeDataProcessor()
    print("Loading data...")
    anime_df, ratings_df = processor.load_data('data/anime.csv', 'data/rating.csv')

    print("Cleaning data...")
    merged_df = processor.clean_data()

    print("Generating visualizations...")
    eda = AnimeEDA(merged_df)
    eda.rating_distribution()
    eda.correlation_analysis()
    eda.episodes_vs_ratings()
    eda.seasonal_trends()
    eda.print_top_animes()

    print("Training machine learning models...")
    ml = AnimeML(merged_df)
    X, y = ml.prepare_features()

    mse, r2 = ml.train_rating_predictor()
    print("Rating Predictor Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    eda.print_summary_statistics()

if __name__ == "__main__":
    main()
