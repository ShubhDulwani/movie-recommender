# Movie Recommendation System using Content-Based Filtering
# Machine Learning Project

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        
    def load_data(self):
        """Load movie dataset"""
        # Sample movie data
        data = {
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'Interstellar', 'The Prestige', 
                'The Silence of the Lambs', 'Se7en', 'Fight Club',
                'The Green Mile', 'The Departed', 'Gladiator'
            ],
            'genre': [
                'Drama', 'Crime', 'Action', 'Crime', 'Drama', 'Sci-Fi',
                'Sci-Fi', 'Crime', 'Sci-Fi', 'Drama', 'Thriller', 'Thriller',
                'Drama', 'Drama', 'Crime', 'Action'
            ],
            'director': [
                'Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan',
                'Quentin Tarantino', 'Robert Zemeckis', 'Christopher Nolan',
                'Wachowski Brothers', 'Martin Scorsese', 'Christopher Nolan',
                'Christopher Nolan', 'Jonathan Demme', 'David Fincher',
                'David Fincher', 'Frank Darabont', 'Martin Scorsese', 'Ridley Scott'
            ],
            'rating': [9.3, 9.2, 9.0, 8.9, 8.8, 8.8, 8.7, 8.7, 8.6, 8.5, 8.6, 8.6, 8.8, 8.6, 8.5, 8.5],
            'year': [1994, 1972, 2008, 1994, 1994, 2010, 1999, 1990, 2014, 2006, 1991, 1995, 1999, 1999, 2006, 2000],
            'tags': [
                'prison hope friendship redemption',
                'mafia family crime power',
                'superhero action thriller dark',
                'crime nonlinear dark comedy',
                'life story inspiration romance drama',
                'dreams heist mind-bending sci-fi',
                'ai reality action sci-fi',
                'mafia crime true story',
                'space time family sci-fi',
                'magic rivalry mystery drama',
                'psychological serial killer crime thriller',
                'detective dark serial killer thriller',
                'psychological twist identity drama',
                'prison death row drama emotional',
                'crime undercover mafia thriller',
                'action historical epic warrior'
            ]
        }
        
        self.movies_df = pd.DataFrame(data)
        print("✓ Dataset loaded successfully!")
        print(f"Total movies in database: {len(self.movies_df)}\n")
        
    def create_features(self):
        """Create combined features for content-based filtering"""
        # Combine genre, director, and tags into one feature
        self.movies_df['combined_features'] = (
            self.movies_df['genre'] + ' ' + 
            self.movies_df['director'] + ' ' + 
            self.movies_df['tags']
        )
        
        print("✓ Features created successfully!\n")
        
    def build_model(self):
        """Build TF-IDF model and compute similarity matrix"""
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print("✓ TF-IDF model built successfully!")
        print(f"Feature matrix shape: {self.tfidf_matrix.shape}")
        print(f"Similarity matrix shape: {self.cosine_sim.shape}\n")
        
    def get_recommendations(self, movie_title, top_n=5):
        """Get movie recommendations based on similarity"""
        try:
            # Get index of the movie
            idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            
            # Sort by similarity score
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top N similar movies (excluding the movie itself)
            sim_scores = sim_scores[1:top_n+1]
            
            # Get movie indices
            movie_indices = [i[0] for i in sim_scores]
            
            # Create recommendations dataframe
            recommendations = self.movies_df.iloc[movie_indices].copy()
            recommendations['similarity_score'] = [score[1] for score in sim_scores]
            
            return recommendations
            
        except IndexError:
            print(f"❌ Movie '{movie_title}' not found in database!")
            return None
            
    def display_recommendations(self, movie_title, recommendations):
        """Display recommendations in a formatted way"""
        print("="*80)
        print(f"🎬 RECOMMENDATIONS BASED ON: {movie_title}")
        print("="*80)
        
        if recommendations is not None and len(recommendations) > 0:
            for idx, row in recommendations.iterrows():
                print(f"\n📽️  {row['title']}")
                print(f"   Genre: {row['genre']}")
                print(f"   Director: {row['director']}")
                print(f"   Rating: ⭐ {row['rating']}/10")
                print(f"   Year: {row['year']}")
                print(f"   Similarity Score: {row['similarity_score']:.2%}")
                print("-" * 80)
        else:
            print("\n❌ No recommendations found!")
            
    def show_all_movies(self):
        """Display all available movies"""
        print("\n" + "="*80)
        print("📚 AVAILABLE MOVIES IN DATABASE")
        print("="*80)
        for idx, row in self.movies_df.iterrows():
            print(f"{idx+1}. {row['title']} ({row['year']}) - {row['genre']} - ⭐ {row['rating']}")
        print("="*80 + "\n")

def main():
    """Main function to run the recommendation system"""
    print("\n" + "="*80)
    print("🎥 MOVIE RECOMMENDATION SYSTEM - ML PROJECT")
    print("="*80 + "\n")
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load and prepare data
    print("📊 Step 1: Loading dataset...")
    recommender.load_data()
    
    print("🔧 Step 2: Creating features...")
    recommender.create_features()
    
    print("🤖 Step 3: Building ML model...")
    recommender.build_model()
    
    # Interactive recommendation loop
    while True:
        print("\n" + "="*80)
        print("OPTIONS:")
        print("1. Get recommendations for a movie")
        print("2. Show all available movies")
        print("3. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            movie_title = input("\nEnter movie title: ").strip()
            num_recommendations = input("How many recommendations? (default 5): ").strip()
            num_recommendations = int(num_recommendations) if num_recommendations else 5
            
            recommendations = recommender.get_recommendations(movie_title, num_recommendations)
            recommender.display_recommendations(movie_title, recommendations)
            
        elif choice == '2':
            recommender.show_all_movies()
            
        elif choice == '3':
            print("\n👋 Thank you for using Movie Recommendation System!")
            print("="*80 + "\n")
            break
            
        else:
            print("\n❌ Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()


