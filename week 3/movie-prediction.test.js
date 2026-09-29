// Focused enhancement checks only; no Step 3 evaluation or benchmark.
// Run: bun movie-prediction.test.js
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const source = ['data.js', 'script.js'].map(file => fs.readFileSync(file, 'utf8')).join('\n');
const c = vm.createContext({ window: {}, console, TextDecoder,
    fetch: async file => new Response(fs.readFileSync(file)) });
vm.runInContext(source, c);
const run = expression => vm.runInContext(expression, c);
const plain = value => JSON.parse(JSON.stringify(value));
let passed = 0;
function test(name, action) { action(); passed++; console.log('PASS', name); }
const validation = JSON.parse(fs.readFileSync('step3-evidence/validation.json', 'utf8'));
function integrity() {
    for (const file of ['u.data', 'u.item', 'readme.md']) {
        assert.equal(createHash('sha256').update(fs.readFileSync(file)).digest('hex'),
            validation.protectedHashesAfter[file]);
    }
    for (const [file, hash] of Object.entries(validation.artifactHashes)) {
        assert.equal(createHash('sha256').update(fs.readFileSync(`step3-evidence/${file}`)).digest('hex'), hash);
    }
}
test('datasets, README and saved Step 3 artifacts unchanged before tests', integrity);
await run('loadData()');
test('invalid user and movie IDs safely return null for both helpers', () => {
    for (const bad of [undefined, null, '', '1', 0, -1, 1.5, NaN, Infinity, 99999]) {
        c.bad = bad;
        for (const method of ['User', 'Item']) {
            assert.equal(run(`predict${method}BasedRating(bad, 512)`), null);
            assert.equal(run(`predict${method}BasedRating(1, bad)`), null);
        }
    }
});
test('already-rated movie rejected by both helpers', () => {
    assert.equal(run('ratingMatrix[1][1]'), 5);
    assert.equal(run('predictUserBasedRating(1, 1)'), null);
    assert.equal(run('predictItemBasedRating(1, 1)'), null);
});
const saved = JSON.parse(fs.readFileSync('step3-evidence/recommendations.json', 'utf8'));
for (const userId of [1, 33]) {
    test(`user ${userId}: Top-5 unchanged; every single score matches and repeats exactly`, () => {
        const row = saved.rows.find(row => row.userId === userId);
        for (const [method, key] of [['User', 'user'], ['Item', 'item']]) {
            const list = plain(run(`get${method}BasedRecommendations(${userId})`));
            assert.deepEqual(list, row[key]);
            for (const item of list) {
                const score = run(`predict${method}BasedRating(${userId}, ${item.movieId})`);
                assert.ok(Number.isFinite(score)); // Neither NaN nor Infinity.
                assert.ok(Math.abs(score - item.score) <= 1e-12);
                assert.equal(run(`predict${method}BasedRating(${userId}, ${item.movieId})`), score);
            }
        }
    });
}
test('User 1 unseen movie 512 supported by both methods', () => {
    const example = plain(run(`({userId:1,movieId:512,title:moviesById.get(512).title,
        observedRating:ratingMatrix[1][512],user:predictUserBasedRating(1,512),
        item:predictItemBasedRating(1,512)})`));
    assert.equal(example.observedRating, 0);
    assert.ok(Number.isFinite(example.user) && Number.isFinite(example.item));
    assert.equal(example.user, 4.96454866430897);
    assert.equal(example.item, 4.247308314123525);
    console.log('MOVIE_EXAMPLE', JSON.stringify(example));
});
test('out-of-range prediction remains unclipped', () => {
    assert.equal(run('predictItemBasedRating(1,1653)'), 5.607784740931014);
});
test('zero-overlap fixture returns null from both methods', () => {
    run(`numUsers=2;numMovies=2;
        movies=[{id:1,title:'A'},{id:2,title:'B'}];
        ratings=[{userId:1,itemId:1,rating:5},{userId:2,itemId:2,rating:4}];
        buildRatingMatrix();`);
    assert.equal(run('predictUserBasedRating(1,2)'), null);
    assert.equal(run('predictItemBasedRating(1,2)'), null);
});
test('datasets, README and saved Step 3 artifacts unchanged after tests', integrity);
console.log(`ALL ${passed} MOVIE PREDICTION TEST GROUPS PASSED. Dropdown/UI checks run in real Chrome.`);
