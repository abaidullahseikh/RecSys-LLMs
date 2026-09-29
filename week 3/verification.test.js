// Step 2 implementation checks, not a quality evaluation or runtime benchmark.
// Run from this directory: bun verification.test.js
// Uses runtime built-ins only; fixtures never write to the MovieLens files.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');

const protectedHashes = {
    'u.data': '06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490',
    'u.item': '553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701',
    'readme.md': 'e2713ef3940dbf376d353d41af986a417f795bbf7ba748b70963401f1b981d8f'
};
const source = ['data.js', 'script.js'].map(name => fs.readFileSync(name, 'utf8')).join('\n');
function context() {
    const c = vm.createContext({ window: {}, console, TextDecoder,
        fetch: async name => new Response(fs.readFileSync(name)) });
    vm.runInContext(source, c);
    return c;
}
function run(c, expression) { return vm.runInContext(expression, c); }
function plain(value) { return JSON.parse(JSON.stringify(value)); }
function near(actual, expected, epsilon = 1e-12) {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= epsilon,
        `Expected ${actual} to equal ${expected} within ${epsilon}`);
}
let passed = 0;
async function test(name, action) {
    await action();
    passed++;
    console.log(`PASS ${name}`);
}
function checkIntegrity() {
    for (const [name, expected] of Object.entries(protectedHashes)) {
        assert.equal(createHash('sha256').update(fs.readFileSync(name)).digest('hex'), expected, name);
    }
}
function fixture(rows) {
    const c = context();
    c.fixtureRows = rows;
    run(c, `numUsers = fixtureRows.length; numMovies = fixtureRows[0].length;
        movies = Array.from({length:numMovies}, (_,i)=>({id:i+1,title:'Fixture '+(i+1),genres:[]}));
        ratings = [];
        fixtureRows.forEach((row,u)=>row.forEach((rating,i)=>{
            if(rating>0) ratings.push({userId:u+1,itemId:i+1,rating,timestamp:0});
        }));
        buildRatingMatrix();`);
    return c;
}
function validateList(c, userId, results, count = 5) {
    assert.equal(results.length, count);
    const ids = new Set();
    for (let i = 0; i < results.length; i++) {
        const item = results[i];
        assert.ok(Number.isInteger(item.movieId));
        assert.equal(run(c, `moviesById.get(${item.movieId}).title`), item.title);
        assert.ok(item.title && Number.isFinite(item.score));
        assert.equal(run(c, `ratingMatrix[${userId}][${item.movieId}]`), 0);
        assert.ok(!ids.has(item.movieId)); ids.add(item.movieId);
        if (i) assert.ok(results[i-1].score > item.score ||
            (results[i-1].score === item.score && results[i-1].movieId < item.movieId));
    }
}

(async () => {
    console.log('ENVIRONMENT', JSON.stringify({ bun: typeof Bun !== 'undefined' ? Bun.version : null,
        runtime: process.version, cwd: process.cwd() }));
    await test('protected files match Step 1 before tests', checkIntegrity);
    const c = context();
    await run(c, 'loadData()');
    await test('943 users, 1682 movies, 100000 ratings; raw-ID dimensions', () => {
        assert.deepEqual(plain(run(c, '[numUsers,numMovies,ratings.length,ratingMatrix.length,ratingMatrix[0].length]')),
            [943,1682,100000,944,1683]);
    });
    await test('all observed records, missing cells, row independence, finite 1–5 ratings', () => {
        assert.ok(run(c, `ratings.every(r=>ratingMatrix[r.userId][r.itemId]===r.rating)`));
        assert.ok(run(c, `ratingMatrix.every(row=>row.length===1683 && row.every(r=>
            Number.isFinite(r) && (r===0 || Number.isInteger(r) && r>=1 && r<=5)))`));
        assert.equal(run(c, 'ratingMatrix.flat().filter(r=>r>0).length'),100000);
        assert.deepEqual(plain(run(c, '[ratingMatrix[196][242],ratingMatrix[1][1],ratingMatrix[1][273]]')),[3,5,0]);
        assert.ok(run(c, 'ratingMatrix[1] !== ratingMatrix[2] && ratingMatrix[0].every(r=>r===0) && ratingMatrix.every(r=>r[0]===0)'));
    });
    await test('Latin-1 title and correct genre alignment', () => {
        assert.equal(run(c, 'moviesById.get(543).title'), 'Misérables, Les (1995)');
        assert.deepEqual(plain(run(c, 'moviesById.get(1).genres')), ['Animation', "Children's", 'Comedy']);
        assert.ok(run(c, 'movies.every(m=>!m.title.includes("\ufffd"))'));
    });
    await test('every cached user/item mean equals an independent observed-record mean', () => {
        assert.ok(run(c, `Array.from({length:numUsers},(_,i)=>i+1).every(u=>{
            const r=ratings.filter(r=>r.userId===u); return Math.abs(getUserMean(u)-r.reduce((s,r)=>s+r.rating,0)/r.length)<1e-12;
        })`));
        assert.ok(run(c, `movies.every(m=>{const r=ratings.filter(r=>r.itemId===m.id);
            return Math.abs(getItemMean(m.id)-r.reduce((s,r)=>s+r.rating,0)/r.length)<1e-12;
        })`));
    });
    const cosineCases = [
        ['identical',[1,2,3],[1,2,3],1,3],
        ['partial overlap',[5,0,4,1],[4,3,2,0],28/Math.sqrt(41*20),2],
        ['zero overlap',[5,0],[0,4],0,0],
        ['one overlap',[1],[5],1,1],
        ['four overlaps',[2,5,1,2],[3,4,4,4],38/Math.sqrt(34*57),4],
        ['zero denominator',[0,0],[0,0],0,0],
        ['empty',[],[],0,0]
    ];
    for (const [name,a,b,raw,n] of cosineCases) await test(`cosine ${name}`, () => {
        const result=c.getCosineDetails(a,b);
        near(result.rawCosine,raw); assert.equal(result.commonCount,n);
        near(result.supportWeight,Math.min(n/50,1));
        near(result.weightedSimilarity,raw*Math.min(n/50,1));
        assert.ok(Object.values(result).every(Number.isFinite));
        near(c.rawCosineSimilarity(a,b),raw); near(c.cosineSimilarity(a,b),result.weightedSimilarity);
        console.log('COSINE_RESULT',name,JSON.stringify(result));
    });
    for(const n of [1,3,40,50,80]) await test(`support n=${n}`,()=>{
        const result=c.getCosineDetails(Array(n).fill(3),Array(n).fill(4));
        near(result.rawCosine,1); near(result.supportWeight,Math.min(n/50,1));
        assert.ok(result.weightedSimilarity<=result.rawCosine && result.weightedSimilarity>=0 && result.weightedSimilarity<=1);
        console.log('SUPPORT_RESULT',JSON.stringify(result));
    });
    await test('invalid vector inputs are rejected, not converted to nonfinite similarities',()=>{
        for(const [a,b] of [[[NaN],[1]],[[Infinity],[1]],[[-1],[1]],[[1,2],[1]]]) {
            assert.throws(()=>c.getCosineDetails(a,b));
        }
    });
    await test('sparse item cosine equals independent dense column cosine',()=>{
        for(const [a,b] of [[1,2],[1,50],[245,258],[1681,1682]]) {
            const sparse=c.getItemCosineDetails(a,b);
            const dense=run(c,`getCosineDetails(ratingMatrix.map(row=>row[${a}]),ratingMatrix.map(row=>row[${b}]))`);
            assert.deepEqual(plain(sparse),plain(dense));
        }
        const f=fixture([[5,4],[1,2],[0,3],[4,0]]);
        const item=f.getItemCosineDetails(1,2);
        near(item.rawCosine,22/Math.sqrt(26*20)); assert.equal(item.commonCount,2);
    });
    await test('Step 1 real overlap examples and capped n>50 example',()=>{
        for(const [u,v,n,raw] of [[2,172,0,0],[1,88,1,1],[1,33,4,38/Math.sqrt(34*57)],[1,5,80,0.9326135855945642]]) {
            const d=run(c,`getCosineDetails(ratingMatrix[${u}],ratingMatrix[${v}])`);
            assert.equal(d.commonCount,n); near(d.rawCosine,raw);
            console.log('REAL_OVERLAP',JSON.stringify({users:[u,v],
                shared:n<=4 ? plain(run(c,`ratedMovieIds[${u}].filter(i=>ratingMatrix[${v}][i]>0).map(i=>({movieId:i,title:moviesById.get(i).title,ratings:[ratingMatrix[${u}][i],ratingMatrix[${v}][i]]}))`)) : '80 shared IDs omitted from console output',...d}));
        }
    });
    await test('lecture User-Based baseline/deviation arithmetic',()=>{
        const result=c.predictFromDeviations(13/3,[
            {similarity:1,rating:2,mean:10/3},
            {similarity:-1,rating:5,mean:8/3},
            {similarity:Math.sqrt(3)/2,rating:4,mean:4}
        ]);
        near(result,13/3+(-11/3)/(2+Math.sqrt(3)/2));
        assert.equal(result.toFixed(2),'3.05');
        console.log('LECTURE_USER_ILLUSTRATIVE',JSON.stringify({prediction:result,
            slideAggregate:4.33-3.667/2.866,note:'Illustrative means consistent with displayed effects, not hidden exact slide data.'}));
    });
    await test('lecture Item-Based 2.58 and signed/absolute-denominator structure',()=>{
        const slideArithmetic=3.67+(-2.764/2.543);
        assert.equal(slideArithmetic.toFixed(2),'2.58');
        const roundedInputs=c.predictFromDeviations(3.67,[
            {similarity:-.99,rating:5,mean:3.33},
            {similarity:.72,rating:3,mean:3},
            {similarity:-.84,rating:5,mean:3.67}
        ]);
        assert.equal(roundedInputs.toFixed(2),'2.58');
        near(roundedInputs,3.67+(-.99*(5-3.33)-.84*(5-3.67))/(.99+.72+.84));
        assert.equal(c.predictFromDeviations(3,[]),null);
        assert.equal(c.predictFromDeviations(3,[{similarity:0,rating:5,mean:3}]),null);
        console.log('LECTURE_ITEM',JSON.stringify({slideArithmetic,roundedInputs,
            note:'Rounded inputs cannot recover all hidden slide decimals; separate from MovieLens.'}));
    });
    await test('User-Based contributors, baseline, deviations, self exclusion, absent candidate rating',()=>{
        const f=fixture([[5,1,0,0],[4,2,5,0],[2,4,1,0],[3,3,0,0]]);
        const neighbors=[{userId:1,similarity:100},{userId:2,similarity:.8},{userId:3,similarity:.2},{userId:4,similarity:.9}];
        near(f.getUserMean(1),3); near(f.getUserMean(2),11/3); near(f.getUserMean(3),7/3);
        near(f.predictUserBasedRating(1,3,neighbors),3+(.8*(5-11/3)+.2*(1-7/3))/(.8+.2));
        assert.equal(f.predictUserBasedRating(1,4,neighbors),null);
        assert.equal(f.predictUserBasedRating(1,1,neighbors),null);
        assert.equal(f.getUserBasedRecommendations(1).length,1);
    });
    await test('Item-Based target mean and item deviations; signed formula integration',()=>{
        const f=fixture([[5,1,0,0],[4,2,5,0],[2,4,1,0],[3,3,0,0]]);
        near(f.getItemMean(1),3.5); near(f.getItemMean(2),2.5); near(f.getItemMean(3),3);
        assert.equal(f.getItemMean(4),null);
        const s1=f.getItemSimilarity(3,1),s2=f.getItemSimilarity(3,2);
        near(f.predictItemBasedRating(1,3),3+(s1*(5-3.5)+s2*(1-2.5))/(Math.abs(s1)+Math.abs(s2)));
        // Inject signed similarity only into an isolated fixture; runtime cosine
        // remains raw and nonnegative. This detects loss of sign/absolute norm.
        run(f,'getItemSimilarity = (a,b) => b===1 ? -.75 : .25');
        near(f.predictItemBasedRating(1,3),1.5);
        assert.equal(f.predictItemBasedRating(1,4),null);
        assert.equal(f.predictItemBasedRating(1,1),null);
        run(f,'getItemSimilarity = () => 0');
        assert.equal(f.predictItemBasedRating(1,3),null);
    });
    await test('deterministic movie-ID tie-break and Top-K boundary',()=>{
        const f=fixture([[5,0,0,0],[5,3,3,3]]);
        for(const fn of ['getUserBasedRecommendations','getItemBasedRecommendations']) {
            assert.deepEqual(plain(f[fn](1,2).map(r=>r.movieId)),[2,3]);
            assert.equal(f[fn](1,0).length,0);
            assert.equal(f[fn](999).length,0);
        }
    });
    await test('baseline/deviation estimates are not silently clipped',()=>{
        near(c.predictFromDeviations(5,[{similarity:.2,rating:5,mean:2}]),8);
        near(c.predictFromDeviations(1,[{similarity:.2,rating:1,mean:4}]),-2);
    });
    await test('real user neighborhoods: self excluded, useful positive similarity, max 20, deterministic ties',()=>{
        for(const u of [1,33]) {
            const neighbors=c.getUserNeighbors(u);
            assert.equal(neighbors.length,20);
            assert.ok(neighbors.every(n=>n.userId!==u && n.similarity>0 && Number.isFinite(n.similarity)));
            const independentlySorted=[];
            for(let other=1;other<=943;other++) if(other!==u) {
                const similarity=run(c,`cosineSimilarity(ratingMatrix[${u}],ratingMatrix[${other}])`);
                if(similarity>0) independentlySorted.push({userId:other,similarity});
            }
            independentlySorted.sort((a,b)=>b.similarity-a.similarity || a.userId-b.userId);
            assert.deepEqual(plain(neighbors),independentlySorted.slice(0,20));
        }
    });
    const realOutputs={};
    for(const userId of [1,33]) await test(`real user ${userId}: both Top-5 valid, distinct evidence, deterministic`,()=>{
        const user=c.getUserBasedRecommendations(userId),item=c.getItemBasedRecommendations(userId);
        validateList(c,userId,user); validateList(c,userId,item);
        assert.deepEqual(plain(user),plain(c.getUserBasedRecommendations(userId)));
        assert.deepEqual(plain(item),plain(c.getItemBasedRecommendations(userId)));
        realOutputs[userId]={ratedCount:run(c,`ratedMovieIds[${userId}].length`),user:plain(user),item:plain(item)};
        console.log('REAL_USER',JSON.stringify({userId,...realOutputs[userId]}));
        console.log('OUTSIDE_1_5',JSON.stringify({userId,user:user.filter(r=>r.score<1||r.score>5),item:item.filter(r=>r.score<1||r.score>5)}));
    });
    await test('independent real-output prediction arithmetic for both methods',()=>{
        for(const [u,output] of Object.entries(realOutputs)) {
            const userId=Number(u),matrix=run(c,'ratingMatrix'),neighbors=c.getUserNeighbors(userId);
            for(const result of output.user) {
                let effect=0,denominator=0;
                for(const n of neighbors) if(matrix[n.userId][result.movieId]>0) {
                    effect+=n.similarity*(matrix[n.userId][result.movieId]-c.getUserMean(n.userId));
                    denominator+=Math.abs(n.similarity);
                }
                near(result.score,c.getUserMean(userId)+effect/denominator);
            }
            for(const result of output.item) {
                let effect=0,denominator=0;
                for(const id of run(c,`ratedMovieIds[${userId}]`)) {
                    // Dense column reference is independent of sparse cache traversal.
                    const a=matrix.map(row=>row[result.movieId]),b=matrix.map(row=>row[id]);
                    let dot=0,aa=0,bb=0,n=0;
                    for(let k=1;k<a.length;k++) if(a[k]>0&&b[k]>0){dot+=a[k]*b[k];aa+=a[k]**2;bb+=b[k]**2;n++;}
                    const s=n?dot/Math.sqrt(aa*bb)*Math.min(n/50,1):0;
                    effect+=s*(matrix[userId][id]-c.getItemMean(id));denominator+=Math.abs(s);
                }
                near(result.score,c.getItemMean(result.movieId)+effect/denominator);
            }
        }
    });
    await test('temporary new user 944 / new item 1683 have no CF evidence',()=>{
        // Isolated context: source records are read, not modified or written.
        const cold=context(); cold.actualMovies=plain(run(c,'movies'));cold.actualRatings=plain(run(c,'ratings'));
        run(cold,`movies=[...actualMovies,{id:1683,title:'Synthetic unrated item',genres:[]}];
            ratings=actualRatings; numUsers=944;numMovies=1683;buildRatingMatrix();`);
        assert.equal(cold.getUserMean(944),null); assert.equal(cold.getItemMean(1683),null);
        assert.equal(cold.getUserNeighbors(944).length,0);
        assert.equal(cold.getUserBasedRecommendations(944).length,0);
        assert.equal(cold.getItemBasedRecommendations(944).length,0);
        assert.equal(cold.predictUserBasedRating(1,1683),null);
        assert.equal(cold.predictItemBasedRating(1,1683),null);
        assert.equal(run(cold,'getCosineDetails(ratingMatrix[944],ratingMatrix[1]).commonCount'),0);
        assert.equal(cold.getItemCosineDetails(1683,1).commonCount,0);
        console.log('COLD_START',{newUser:944,userRecommendations:[],itemRecommendations:[],newItem:1683,userPrediction:null,itemPrediction:null,commonCounts:0});
    });
    await test('item cache invalidated after matrix rebuild; no stale means',()=>{
        const f=fixture([[5,4],[1,2]]);const old=f.getItemSimilarity(1,2);
        run(f,'ratings=[{userId:1,itemId:1,rating:5},{userId:2,itemId:2,rating:2}];buildRatingMatrix()');
        assert.ok(old>0);assert.equal(f.getItemSimilarity(1,2),0);assert.equal(f.getItemMean(1),5);
    });
    await test('reload replaces arrays and keeps real outputs reproducible',async()=>{
        await run(c,'loadData()');
        assert.deepEqual(plain(run(c,'[movies.length,ratings.length,numMovies]')),[1682,100000,1682]);
        assert.deepEqual(plain(c.getItemBasedRecommendations(1)),realOutputs[1].item);
    });
    await test('protected files still match Step 1 after all tests',checkIntegrity);
    console.log(`ALL ${passed} IMPLEMENTATION TEST GROUPS PASSED. No full quality evaluation or runtime benchmark performed.`);
})().catch(error=>{console.error(error);process.exitCode=1;});
