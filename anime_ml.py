# anime_ml.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

class AnimeML:
    def __init__(self, df):
        self.df = df
        self.X = None
        self.y = None
        self.model_rf = None
        self.tfidf = None
        self.cosine_sim = None

    def prepare_features(self):
        """Prepare features for ML models"""
        le = LabelEncoder()
        self.df['type_encoded'] = le.fit_transform(self.df['type'])

        # Limit dataset size to improve speed
        self.df = self.df.sample(n=5000, random_state=42)  # Use a random subset

        self.X = pd.DataFrame({
            'episodes': self.df['episodes'],
            'type_encoded': self.df['type_encoded'],
            'members': self.df['members'],
            'user_rating': self.df['user_rating']
        })
        self.y = self.df['average_rating']

        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        return self.X, self.y

    def train_rating_predictor(self):
        """Train Random Forest model for rating prediction"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Reduce number of estimators and limit depth for faster training
        self.model_rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        self.model_rf.fit(X_train, y_train)

        y_pred = self.model_rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature Importance
        feature_importances = pd.Series(self.model_rf.feature_importances_, index=self.X.columns)
        feature_importances.sort_values(ascending=False).plot(kind='bar', color='teal', edgecolor='black')
        plt.title('Feature Importance')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.show()

        return mse, r2

    def train_content_recommender(self):
        """Train content-based recommendation system"""
        self.tfidf = TfidfVectorizer(max_features=1000, token_pattern=r'[^,\s][^,]*[^,\s]')
        genre_matrix = self.tfidf.fit_transform(self.df['genre'])
        self.cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
        return self.cosine_sim
