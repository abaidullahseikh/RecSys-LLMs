// Global variables for storing movie and rating data
let movies = [];
let ratings = [];

// Collaborative filtering structures (populated by buildRatingMatrix)
let numUsers = 0;          // highest user id found in u.data
let numMovies = 0;         // number of parsed movies
let ratingMatrix = null;   // (numUsers + 1) x (numMovies + 1); 0 = "not rated"
let moviesById = new Map();
let ratedMovieIds = [];
let itemRaterIds = [];
let userMeans = [];
let itemMeans = [];

// Genre names as defined in the u.item file
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Reloading replaces the data rather than appending duplicate records.
        movies = [];
        ratings = [];
        ratingMatrix = null;
        numUsers = 0;
        numMovies = 0;
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        // MovieLens u.item contains Latin-1 titles; UTF-8 decoding loses accents.
        const moviesText = new TextDecoder('iso-8859-1').decode(await moviesResponse.arrayBuffer());
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Derive matrix dimensions, then build the rating matrix
        numUsers = ratings.reduce((max, r) => Math.max(max, r.userId), 0);
        numMovies = movies.length;
        buildRatingMatrix();
    } catch (error) {
        console.error('Error loading data:', error);
        throw error; // script.js owns the error state in both result cards.
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];

        // Field 5 is the "unknown" flag; the following 18 flags match genreNames.
        const genreValues = fields.slice(6, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);

        movies.push({ id, title, genres });
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines

        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);

        ratings.push({ userId, itemId, rating, timestamp });
    }
}

// Raw-ID matrix; row/column 0 are reserved. Missing cells stay 0, never imputed.
// Means use observed ratings only. Sparse ID indexes support column comparisons
// without repeatedly scanning every user. Rebuilding replaces all derived data.
function buildRatingMatrix() {
    ratingMatrix = Array.from({ length: numUsers + 1 }, () => Array(numMovies + 1).fill(0));
    moviesById = new Map(movies.map(movie => [movie.id, movie]));
    ratedMovieIds = Array.from({ length: numUsers + 1 }, () => []);
    itemRaterIds = Array.from({ length: numMovies + 1 }, () => []);
    userMeans = Array(numUsers + 1).fill(null);
    itemMeans = Array(numMovies + 1).fill(null);
    const userSums = Array(numUsers + 1).fill(0);
    const itemSums = Array(numMovies + 1).fill(0);

    for (const { userId, itemId, rating } of ratings) {
        if (!Number.isInteger(userId) || userId < 1 || userId > numUsers ||
            !Number.isInteger(itemId) || itemId < 1 || itemId > numMovies ||
            !moviesById.has(itemId) || !Number.isInteger(rating) || rating < 1 || rating > 5) {
            throw new Error('Invalid user, movie, or rating in the dataset.');
        }
        if (ratingMatrix[userId][itemId] !== 0) {
            throw new Error(`Duplicate rating for user ${userId}, movie ${itemId}.`);
        }
        ratingMatrix[userId][itemId] = rating;
        ratedMovieIds[userId].push(itemId);
        itemRaterIds[itemId].push(userId);
        userSums[userId] += rating;
        itemSums[itemId] += rating;
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        ratedMovieIds[userId].sort((a, b) => a - b);
        if (ratedMovieIds[userId].length) {
            userMeans[userId] = userSums[userId] / ratedMovieIds[userId].length;
        }
    }
    for (let movieId = 1; movieId <= numMovies; movieId++) {
        itemRaterIds[movieId].sort((a, b) => a - b);
        if (itemRaterIds[movieId].length) {
            itemMeans[movieId] = itemSums[movieId] / itemRaterIds[movieId].length;
        }
    }
}

// null means no observed baseline; it is not a fabricated zero rating.
function getItemMean(movieId) {
    return itemMeans[movieId] ?? null;
}

function getUserMean(userId) {
    return userMeans[userId] ?? null;
}
