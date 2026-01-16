# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Machine Learning techniques including TF-IDF vectorization and cosine similarity.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Screenshots](#screenshots)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a **Content-Based Filtering** approach to recommend movies similar to a user's selection. The system analyzes movie features like genre, director, tags, and ratings to find and suggest similar movies.

## ✨ Features

- **Content-Based Filtering**: Recommends movies based on similarity of features
- **TF-IDF Vectorization**: Converts text features into numerical vectors
- **Cosine Similarity**: Calculates similarity between movies
- **Interactive CLI**: User-friendly command-line interface
- **Customizable Recommendations**: Choose number of recommendations
- **Similarity Scores**: Shows percentage match for each recommendation

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
  - TF-IDF Vectorizer
  - Cosine Similarity

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/ShubhDulwani/movie-recommender.git
cd movie-recommender
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python movie_recommender.py
```

## 🚀 Usage

1. Run the program:
```bash
python movie_recommender.py
```

2. Choose from the menu options:
   - **Option 1**: Get recommendations for a specific movie
   - **Option 2**: View all available movies in the database
   - **Option 3**: Exit the program

3. Enter the movie title when prompted (e.g., "Inception")

4. Specify the number of recommendations you want (default: 5)

5. View personalized recommendations with similarity scores!

### Example:
```
Enter movie title: Inception
How many recommendations? (default 5): 3

🎬 RECOMMENDATIONS BASED ON: Inception
================================================================================

📽️  Interstellar
   Genre: Sci-Fi
   Director: Christopher Nolan
   Rating: ⭐ 8.6/10
   Year: 2014
   Similarity Score: 78.45%
```

## 🧠 How It Works

### 1. **Feature Engineering**
Combines multiple movie attributes into a single feature:
- Genre
- Director
- Tags (keywords describing the movie)

### 2. **TF-IDF Vectorization**
Converts text features into numerical vectors using Term Frequency-Inverse Document Frequency

### 3. **Similarity Calculation**
Uses cosine similarity to find movies with similar feature vectors:
```
Similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

### 4. **Recommendation Generation**
Returns top N movies with highest similarity scores

## 📊 Dataset

The current dataset includes **16 popular movies** with the following attributes:
- **Title**: Movie name
- **Genre**: Category (Drama, Sci-Fi, Crime, Action, Thriller)
- **Director**: Director name
- **Rating**: IMDb rating
- **Year**: Release year
- **Tags**: Descriptive keywords

### Sample Movies:
- The Shawshank Redemption
- The Godfather
- Inception
- The Matrix
- Pulp Fiction
- And more...

## 📸 Screenshots

### Main Menu
```
🎥 MOVIE RECOMMENDATION SYSTEM - ML PROJECT
================================================================================

OPTIONS:
1. Get recommendations for a movie
2. Show all available movies
3. Exit
```

### Recommendations Output
```
🎬 RECOMMENDATIONS BASED ON: The Dark Knight
================================================================================

📽️  Gladiator
   Genre: Action
   Director: Ridley Scott
   Rating: ⭐ 8.5/10
   Year: 2000
   Similarity Score: 45.23%
```

## 🔮 Future Enhancements

- [ ] Add collaborative filtering
- [ ] Integrate with real movie APIs (TMDb, OMDb)
- [ ] Implement hybrid recommendation system
- [ ] Add user ratings and feedback
- [ ] Create web interface using Flask/Django
- [ ] Include movie posters and descriptions
- [ ] Add more movies to the dataset
- [ ] Implement user authentication
- [ ] Store user preferences and history

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shubh Dulwani**

- GitHub: [@ShubhDulwani](https://github.com/ShubhDulwani)

## 🙏 Acknowledgments

- Scikit-learn documentation
- Machine Learning community
- Movie data sources

## 📞 Contact

For any queries or suggestions, feel free to reach out!

---

⭐ **If you found this project helpful, please give it a star!** ⭐