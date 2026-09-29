// ---------------------------------------------------------------------------
// HW3 — Collaborative Filtering core
//
// ONE missing-value strategy: WEIGHTED BY NUMBER OF COMMON RATINGS.
// Raw cosine uses only jointly observed, non-zero ratings (no imputation).
// SOURCE-DERIVED: lecture p.16: final_sim = raw_sim * n / 50.
// IMPLEMENTATION DECISION (approved Step 2): cap support at 1 to reconcile
// that factor with the starter's [0,1] contract. The lecture did not give a cap.
// Prediction equations: lecture p.9 (user baseline + neighbor deviations),
// p.11 (target-item baseline + rated-item deviations), inspected in Step 1.
// Raw, positive-rating cosine is required here; the lecture's signed examples
// verify prediction algebra, NOT the source of runtime cosine similarities.
// ---------------------------------------------------------------------------
const SUPPORT_SCALE = 50;
const USER_NEIGHBOR_COUNT = 20; // IMPLEMENTATION DECISION: README's example N.
let itemSimilarityCache = new Map();
let similarityMatrixVersion = null;
let isCalculating = false;

// Initialize the application when the window loads
window.onload = async function() {
    setControlsDisabled(true);
    try {
        setResultState('Loading data…');
        await loadData();
        populateUserDropdown();
        populateMovieDropdown();
        setResultState('Ready — select a user.');
        setControlsDisabled(false);
    } catch (error) {
        console.error('Initialization error:', error);
        setResultState(`Error: ${error.message}`, true);
    }
};

// Populate the user dropdown with one option per user id found in u.data
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    for (let userId = 1; userId <= numUsers; userId++) {
        if (!ratedMovieIds[userId].length) continue;
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    }
}

// Shared arithmetic for dense rows and sparse item-column intersections.
function finishCosine(dot, squaredNormA, squaredNormB, commonCount) {
    const denominator = Math.sqrt(squaredNormA) * Math.sqrt(squaredNormB);
    let rawCosine = commonCount > 0 && denominator > 0 ? dot / denominator : 0;
    if (!Number.isFinite(rawCosine)) rawCosine = 0;
    // Only a floating-point cosine boundary guard, NOT prediction clipping.
    rawCosine = Math.max(0, Math.min(1, rawCosine));
    const supportWeight = Math.min(commonCount / SUPPORT_SCALE, 1);
    return { dot, squaredNormA, squaredNormB, commonCount, rawCosine,
        supportWeight, weightedSimilarity: rawCosine * supportWeight };
}

function getCosineDetails(a, b) {
    if (a.length !== b.length) throw new Error('Rating vectors must have equal length.');
    let dot = 0, squaredNormA = 0, squaredNormB = 0, commonCount = 0;
    for (let i = 0; i < a.length; i++) {
        if (!Number.isFinite(a[i]) || !Number.isFinite(b[i]) || a[i] < 0 || b[i] < 0) {
            throw new Error('Cosine requires finite, nonnegative rating vectors.');
        }
        if (a[i] === 0 || b[i] === 0) continue;
        dot += a[i] * b[i];
        squaredNormA += a[i] * a[i];
        squaredNormB += b[i] * b[i];
        commonCount++;
    }
    return finishCosine(dot, squaredNormA, squaredNormB, commonCount);
}

function rawCosineSimilarity(a, b) {
    return getCosineDetails(a, b).rawCosine;
}

function cosineSimilarity(a, b) {
    return getCosineDetails(a, b).weightedSimilarity;
}

function getItemCosineDetails(movieA, movieB) {
    const ratersA = itemRaterIds[movieA] || [];
    const ratersB = itemRaterIds[movieB] || [];
    const raters = ratersA.length <= ratersB.length ? ratersA : ratersB;
    let dot = 0, squaredNormA = 0, squaredNormB = 0, commonCount = 0;
    for (const userId of raters) {
        const a = ratingMatrix[userId][movieA];
        const b = ratingMatrix[userId][movieB];
        if (a === 0 || b === 0) continue;
        dot += a * b;
        squaredNormA += a * a;
        squaredNormB += b * b;
        commonCount++;
    }
    return finishCosine(dot, squaredNormA, squaredNormB, commonCount);
}

function getItemSimilarity(movieA, movieB) {
    if (similarityMatrixVersion !== ratingMatrix) {
        itemSimilarityCache.clear();
        similarityMatrixVersion = ratingMatrix;
    }
    const key = Math.min(movieA, movieB) * (numMovies + 1) + Math.max(movieA, movieB);
    if (!itemSimilarityCache.has(key)) {
        itemSimilarityCache.set(key, getItemCosineDetails(movieA, movieB).weightedSimilarity);
    }
    return itemSimilarityCache.get(key);
}

function isValidUser(userId) {
    return ratingMatrix !== null && Number.isInteger(userId) && userId >= 1 && userId <= numUsers;
}

function getUserNeighbors(activeUserId) {
    if (!isValidUser(activeUserId) || getUserMean(activeUserId) === null) return [];
    const neighbors = [];
    for (let userId = 1; userId <= numUsers; userId++) {
        if (userId === activeUserId || getUserMean(userId) === null) continue;
        const similarity = cosineSimilarity(ratingMatrix[activeUserId], ratingMatrix[userId]);
        if (similarity > 0) neighbors.push({ userId, similarity });
    }
    return neighbors.sort((a, b) => b.similarity - a.similarity || a.userId - b.userId)
        .slice(0, USER_NEIGHBOR_COUNT);
}

// SOURCE-DERIVED algebra (lecture pp.9,11): baseline + sum(sim * deviation)
// / sum(abs(sim)). Keeping signed effects here also permits separate lecture
// fixtures with negative similarity; runtime raw-rating cosine is nonnegative.
// No supporting terms => null. Predictions are deliberately NOT clipped to 1–5.
function predictFromDeviations(baseline, contributions) {
    if (!Number.isFinite(baseline)) return null;
    let numerator = 0, denominator = 0;
    for (const { similarity, rating, mean } of contributions) {
        if (!Number.isFinite(similarity) || similarity === 0 ||
            !Number.isFinite(rating) || rating <= 0 || !Number.isFinite(mean)) continue;
        numerator += similarity * (rating - mean);
        denominator += Math.abs(similarity);
    }
    if (denominator === 0) return null;
    const score = baseline + numerator / denominator;
    return Number.isFinite(score) ? score : null;
}

function predictUserBasedRating(activeUserId, movieId, neighbors = getUserNeighbors(activeUserId)) {
    if (!isValidUser(activeUserId) || !moviesById.has(movieId) || ratingMatrix[activeUserId][movieId] !== 0) return null;
    const contributions = neighbors.filter(neighbor => neighbor.userId !== activeUserId &&
        neighbor.similarity > 0 && ratingMatrix[neighbor.userId]?.[movieId] > 0)
        .map(neighbor => ({ similarity: neighbor.similarity,
            rating: ratingMatrix[neighbor.userId][movieId], mean: getUserMean(neighbor.userId) }));
    // IMPLEMENTATION DECISION: baseline means use all observed ratings of each
    // user, not just this pair's overlap. This is the lecture's user-mean role.
    return predictFromDeviations(getUserMean(activeUserId), contributions);
}

function predictItemBasedRating(activeUserId, movieId) {
    if (!isValidUser(activeUserId) || !moviesById.has(movieId) || ratingMatrix[activeUserId][movieId] !== 0 ||
        getUserMean(activeUserId) === null || getItemMean(movieId) === null) return null;
    const contributions = ratedMovieIds[activeUserId].map(ratedId => ({
        similarity: getItemSimilarity(movieId, ratedId),
        rating: ratingMatrix[activeUserId][ratedId], mean: getItemMean(ratedId)
    }));
    return predictFromDeviations(getItemMean(movieId), contributions);
}

function rankRecommendations(activeUserId, topK, predict) {
    if (!isValidUser(activeUserId) || getUserMean(activeUserId) === null ||
        !Number.isInteger(topK) || topK <= 0) return [];
    const candidates = [];
    for (const [movieId, movie] of moviesById) {
        if (ratingMatrix[activeUserId][movieId] !== 0) continue;
        const score = predict(movieId);
        if (Number.isFinite(score) && movie.title) {
            candidates.push({ movieId, title: movie.title, score });
        }
    }
    return candidates.sort((a, b) => b.score - a.score || a.movieId - b.movieId).slice(0, topK);
}

function getUserBasedRecommendations(activeUserId, topK = 5) {
    const neighbors = getUserNeighbors(activeUserId);
    return rankRecommendations(activeUserId, topK,
        movieId => predictUserBasedRating(activeUserId, movieId, neighbors));
}

function getItemBasedRecommendations(activeUserId, topK = 5) {
    return rankRecommendations(activeUserId, topK,
        movieId => predictItemBasedRating(activeUserId, movieId));
}

function setControlsDisabled(disabled) {
    document.getElementById('user-select').disabled = disabled;
    document.getElementById('recommend-btn').disabled = disabled;
    updatePredictionControls();
}

// UI only: the active user and both prediction helpers are shared with Top-5.
function populateMovieDropdown() {
    const userId = Number(document.getElementById('user-select').value);
    const select = document.getElementById('movie-select');
    select.replaceChildren();
    const unseen = isValidUser(userId) ? [...moviesById.values()]
        .filter(movie => ratingMatrix[userId][movie.id] === 0)
        .sort((a, b) => a.id - b.id) : [];
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = !isValidUser(userId) ? 'Select a user first' :
        unseen.length ? 'Select a movie' : 'No unseen movies for this user';
    select.appendChild(placeholder);
    for (const movie of unseen) {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        select.appendChild(option);
    }
    resetMoviePrediction();
    document.getElementById('prediction-status').textContent = placeholder.textContent;
    updatePredictionControls();
}

function updatePredictionControls() {
    const user = document.getElementById('user-select');
    const movie = document.getElementById('movie-select');
    movie.disabled = user.disabled || !isValidUser(Number(user.value)) || movie.options.length <= 1;
    const movieId = Number(movie.value);
    document.getElementById('predict-rating-btn').disabled = movie.disabled ||
        !moviesById.has(movieId) || ratingMatrix[Number(user.value)]?.[movieId] !== 0;
}

function resetMoviePrediction() {
    for (const id of ['user-based-prediction', 'item-based-prediction']) {
        document.getElementById(id).textContent = '—';
        document.getElementById(`${id}-note`).textContent = '';
    }
    document.getElementById('prediction-status').textContent = 'Select a movie and click Predict Rating.';
}

async function predictSelectedMovie() {
    if (isCalculating) return;
    resetMoviePrediction();
    const userId = Number(document.getElementById('user-select').value);
    const movieId = Number(document.getElementById('movie-select').value);
    const status = document.getElementById('prediction-status');
    if (!isValidUser(userId) || !moviesById.has(movieId) || ratingMatrix[userId][movieId] !== 0) {
        status.textContent = 'Please select a valid user and an unseen movie.';
        return;
    }
    isCalculating = true;
    setControlsDisabled(true);
    document.getElementById('movie-prediction').setAttribute('aria-busy', 'true');
    status.textContent = 'Calculating predictions…';
    try {
        await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
        const scores = [predictUserBasedRating(userId, movieId), predictItemBasedRating(userId, movieId)];
        ['user-based-prediction', 'item-based-prediction'].forEach((id, index) => {
            document.getElementById(id).textContent = Number.isFinite(scores[index]) ?
                scores[index].toFixed(1) : 'N/A';
            document.getElementById(`${id}-note`).textContent = Number.isFinite(scores[index]) ?
                '' : 'insufficient evidence';
        });
        status.textContent = `User ${userId} · ${moviesById.get(movieId).title}` +
            (scores.some(score => Number.isFinite(score) && (score < 1 || score > 5)) ?
                ' · Estimate outside 1–5; scores are not clipped.' : '');
    } catch (error) {
        console.error('Prediction error:', error);
        status.textContent = `Error: ${error.message}`;
    } finally {
        isCalculating = false;
        setControlsDisabled(false);
        document.getElementById('movie-prediction').setAttribute('aria-busy', 'false');
    }
}

function setResultState(message, error = false) {
    for (const id of ['user-based-result', 'item-based-result']) {
        renderList(id, [], message);
        document.getElementById(id).classList.toggle('error', error);
    }
    document.getElementById('app-status').textContent = message;
}

// Existing interaction: one user -> both Top-5 lists. Let the calculating state
// paint before synchronous CF work, and prevent overlapping requests.
async function getRecommendations() {
    if (isCalculating) return;
    const selected = document.getElementById('user-select').value;
    const userId = Number(selected);
    if (selected === '') {
        setResultState('Please select a user first.');
        return;
    }
    if (!isValidUser(userId)) {
        setResultState('Error: select a valid user.', true);
        return;
    }
    isCalculating = true;
    setControlsDisabled(true);
    document.getElementById('result-box').setAttribute('aria-busy', 'true');
    setResultState(`Calculating recommendations for User ${userId}…`);
    try {
        await new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
        const userResults = getUserBasedRecommendations(userId);
        const itemResults = getItemBasedRecommendations(userId);
        renderList('user-based-result', userResults);
        renderList('item-based-result', itemResults);
        const outsideRange = [...userResults, ...itemResults].some(item => item.score < 1 || item.score > 5);
        document.getElementById('app-status').textContent =
            `Ready — User ${userId} · ${ratedMovieIds[userId].length} rated movies.` +
            (outsideRange ? ' Some baseline/deviation estimates are outside 1–5; scores are not clipped.' : '');
    } catch (error) {
        console.error('Recommendation error:', error);
        setResultState(`Error: ${error.message}`, true);
    } finally {
        isCalculating = false;
        setControlsDisabled(false);
        document.getElementById('result-box').setAttribute('aria-busy', 'false');
    }
}

function renderList(elementId, items, message) {
    const el = document.getElementById(elementId);
    el.replaceChildren();
    if (message || !items || items.length === 0) {
        const paragraph = document.createElement('p');
        paragraph.textContent = message || 'No recommendation evidence — not enough shared ratings to score unseen movies.';
        el.appendChild(paragraph);
        return;
    }
    const list = document.createElement('ol');
    list.className = 'recommendations';
    for (const item of items) {
        if (!Number.isFinite(item.score)) throw new Error('Cannot display a nonfinite predicted rating.');
        const entry = document.createElement('li');
        entry.dataset.movieId = item.movieId;
        const title = document.createElement('span');
        title.className = 'movie-title';
        title.textContent = item.title;
        const score = document.createElement('span');
        score.className = 'predicted-rating';
        score.textContent = `Predicted rating: ${item.score.toFixed(3)}`;
        entry.append(title, score);
        list.appendChild(entry);
    }
    el.appendChild(list);
}
