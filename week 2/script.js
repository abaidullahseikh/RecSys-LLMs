// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Cosine similarity for equally sized vectors; missing genre information scores zero.
function cosineSimilarity(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length) {
        throw new Error('Similarity vectors must have the same length.');
    }

    let dotProduct = 0;
    let normASquared = 0;
    let normBSquared = 0;
    for (let index = 0; index < vectorA.length; index++) {
        dotProduct += vectorA[index] * vectorB[index];
        normASquared += vectorA[index] * vectorA[index];
        normBSquared += vectorB[index] * vectorB[index];
    }

    if (normASquared === 0 || normBSquared === 0) return 0;
    return dotProduct / (Math.sqrt(normASquared) * Math.sqrt(normBSquared));
}

// Shared Top 5 ranking for an item vector or an averaged profile vector.
function recommendFromVector(vector, excludedMovieIds = []) {
    if (!vector.some(value => value > 0)) return [];

    const excludedIds = new Set(excludedMovieIds);
    return movies
        .filter(movie => !excludedIds.has(movie.id) && movie.genreVector.some(value => value > 0))
        .map(movie => ({ ...movie, score: cosineSimilarity(vector, movie.genreVector) }))
        .sort((a, b) => b.score - a.score || a.id - b.id)
        .slice(0, 5);
}

// Console/test entry point; additional exclusions allow a shared comparison candidate pool.
function getItemRecommendations(movieId, excludedMovieIds = []) {
    const movie = movies.find(movie => movie.id === movieId);
    if (!movie) throw new Error('Selected movie not found in database.');
    return recommendFromVector(movie.genreVector, [movieId, ...excludedMovieIds]);
}

// Equal-weight arithmetic mean of the selected movies, before cosine normalization.
function buildProfileVector(movieIds) {
    if (movieIds.length === 0 || new Set(movieIds).size !== movieIds.length) {
        throw new Error('A profile requires at least one movie and distinct movie IDs.');
    }

    const profile = new Array(genreNames.length).fill(0);
    movieIds.forEach(id => {
        const movie = movies.find(movie => movie.id === id);
        if (!movie || !movie.genreVector.some(value => value > 0)) {
            throw new Error(`Movie ${id} has no usable known-genre vector.`);
        }
        movie.genreVector.forEach((value, index) => {
            profile[index] += value;
        });
    });
    return profile.map(value => value / movieIds.length);
}

// Three positively rated movies build the profile; the user's entire rated history is excluded.
function getProfileRecommendations(userId, likedMovieIds) {
    if (likedMovieIds.length !== 3) {
        throw new Error('Select exactly three positively rated movies for the experiment.');
    }

    const userRatings = ratings.filter(rating => rating.userId === userId);
    if (!likedMovieIds.every(id => userRatings.some(rating =>
        rating.itemId === id && (rating.rating === 4 || rating.rating === 5)
    ))) {
        throw new Error('Each profile movie must be rated 4 or 5 by the selected user.');
    }

    const profile = buildProfileVector(likedMovieIds);
    const watchedMovieIds = userRatings.map(rating => rating.itemId);
    return recommendFromVector(profile, watchedMovieIds);
}

// Main recommendation function for the existing single-movie UI.
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);
        
        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Steps 3-6: exclude the active/zero-vector movies and rank the cosine Top 5.
                const topRecommendations = getItemRecommendations(likedMovie.id);
                
                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
