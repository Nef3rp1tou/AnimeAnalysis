# anime_eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
class AnimeEDA:
    def __init__(self, df):
        self.df = df
        plt.style.use('ggplot')

    def rating_distribution(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(self.df['user_rating'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of User Ratings')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.show()

    def correlation_analysis(self):
        """Generate a correlation heatmap for numerical features"""
        numeric_features = self.df[['episodes', 'members', 'user_rating', 'average_rating']]
        correlation_matrix = numeric_features.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def episodes_vs_ratings(self):
        """Interactive scatter plot using Plotly for episodes vs average rating."""
        # Create a Plotly scatter plot
        fig = px.scatter(
            self.df,
            x='episodes',
            y='average_rating',
            hover_data=['name'],  # Show anime names on hover
            title='Episodes vs Average Rating',
            labels={'episodes': 'Number of Episodes', 'average_rating': 'Average Rating'}
        )

        # Update marker size and transparency
        fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))

        # Show the figure
        fig.show()

    def print_summary_statistics(self):
        """Print summary statistics for numerical columns"""
        print("\nSummary Statistics:")
        print(f"- Average User Rating: {self.df['user_rating'].mean():.2f}")
        print(f"- Average Episodes: {self.df['episodes'].mean():.2f}")
        print(f"- Average Members: {self.df['members'].mean():,.0f}")

        if 'release_season' in self.df.columns:
            most_common_season = self.df['release_season'].mode()[0]
            highest_avg_rating_season = self.df.groupby('release_season')['average_rating'].mean().idxmax()
            print("\nInsights:")
            print(f"- Most Common Release Season: {most_common_season}")
            print(f"- Highest Average Rating by Season: {highest_avg_rating_season}")


    def seasonal_trends(self):
        """Analyze seasonal trends in anime ratings"""
        self.df['release_season'] = self.df['name'].str.extract('(Spring|Summer|Fall|Winter)', expand=False)
        season_avg_ratings = self.df.groupby('release_season')['average_rating'].mean()

        plt.figure(figsize=(8, 6))
        season_avg_ratings.plot(kind='bar', color='purple', edgecolor='black')
        plt.title('Average Ratings by Release Season')
        plt.xlabel('Season')
        plt.ylabel('Average Rating')
        plt.tight_layout()
        plt.show()

    def print_top_animes(self):
        """Print top 10 anime by average rating"""
        top_animes = self.df[['name', 'average_rating']].drop_duplicates(subset=['name']).sort_values(by='average_rating', ascending=False).head(10)
        print("Top 10 Anime by Average Rating:")
        print(top_animes.to_string(index=False))
