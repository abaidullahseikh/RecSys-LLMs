// Reproducible evaluation of UNMODIFIED Week 3 production functions.
// Commands: bun step3-evaluation.js quality|overlap|recommendations|benchmark|validate
// Built-ins only. Writes evidence under step3-evidence/, never production files.
const fs = require('node:fs');
const vm = require('node:vm');
const os = require('node:os');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const path = require('node:path');
assert.equal(process.cwd(), __dirname, 'Run from the Week 3 project directory.');
const expected = {
    'u.data':'06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490',
    'u.item':'553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701',
    'readme.md':'e2713ef3940dbf376d353d41af986a417f795bbf7ba748b70963401f1b981d8f',
    'data.js':'ec0e696ec50f5207ee6d73c818ff7c66c3dbe0b96444fbca5300e229afe93103',
    'script.js':'66ff1600ac8658a1bb6d2b5d75a764ab853cceced8495c3c193421699dc49511',
    'index.html':'971676a647b3436cab6f558289a6fd3d1d75744efbd0f40a21127a66742b560c',
    'style.css':'4d46415c5d9e9336d2129701c0d5825eba926863b9ab832adf7bdc13f7b187d9'
};
const digest = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
function integrity() {
    const hashes = Object.fromEntries(Object.keys(expected).map(file=>[file,digest(file)]));
    assert.deepEqual(hashes,expected,'STOP: protected source/dataset changed.');
    return hashes;
}
const before = integrity();
const config = JSON.parse(fs.readFileSync('step3-evidence/protocol.json','utf8'));
const environment = {
    bun:Bun.version,nodeCompatibility:process.version,platform:process.platform,
    architecture:process.arch,osRelease:os.release(),cpu:os.cpus()[0].model,
    logicalCPUs:os.cpus().length,totalMemoryBytes:os.totalmem(),cwd:process.cwd()
};
const run = (c, expression) => vm.runInContext(expression,c);
const plain = value => JSON.parse(JSON.stringify(value));
const sum = values => values.reduce((a,b)=>a+b,0);
function stats(values) {
    if(!values.length) return {count:0,min:null,median:null,mean:null,max:null};
    const a=[...values].sort((a,b)=>a-b),n=a.length;
    return {count:n,min:a[0],median:n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2,mean:sum(a)/n,max:a[n-1]};
}
function near(a,b) { assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`); }
function save(name,payload) {
    const evidence = {command:`bun step3-evaluation.js ${process.argv[2]}`,
        recordedAt:new Date().toISOString(),environment,config,
        harnessSHA256:digest('step3-evaluation.js'),protocolSHA256:digest('step3-evidence/protocol.json'),
        protectedHashesBefore:before,protectedHashesAfter:integrity(),...payload};
    // Separate runs are explicit commands; no production state is persisted.
    fs.writeFileSync(path.join('step3-evidence',name+'.json'),JSON.stringify(evidence,null,2)+'\n');
    console.log('SAVED',`step3-evidence/${name}.json`);
}
async function load() {
    const c=vm.createContext({window:{},console,TextDecoder,
        fetch:async name=>new Response(fs.readFileSync(name))});
    for(const file of ['data.js','script.js']) vm.runInContext(fs.readFileSync(file,'utf8'),c,{filename:file});
    await run(c,'loadData()');
    assert.deepEqual(plain(run(c,'[numUsers,numMovies,ratings.length]')),[943,1682,100000]);
    return c;
}
function metrics(rows,key,clip=false) {
    const valid=rows.filter(r=>Number.isFinite(r[key]));
    const errors=valid.map(r=>(clip?Math.max(1,Math.min(5,r[key])):r[key])-r.actual);
    return {attempted:rows.length,valid:valid.length,missing:rows.length-valid.length,
        coveragePercent:100*valid.length/rows.length,
        mae:valid.length?sum(errors.map(Math.abs))/valid.length:null,
        rmse:valid.length?Math.sqrt(sum(errors.map(e=>e*e))/valid.length):null};
}
function range(rows,key) {
    const valid=rows.filter(r=>Number.isFinite(r[key]));
    const a=valid.map(r=>r[key]);
    const below=valid.filter(r=>r[key]<1),above=valid.filter(r=>r[key]>5);
    return {valid:valid.length,below1:below.length,inside1to5:valid.length-below.length-above.length,
        above5:above.length,outsidePercent:100*(below.length+above.length)/valid.length,
        min:Math.min(...a),max:Math.max(...a),outsideCases:[...below,...above]};
}
function inBin(n,[low,high]) {return n>=low && (high===null||n<=high);}
function binLabel([low,high]) {return high===null?`${low}+`:`${low}-${high}`;}

async function quality() {
    const c=await load();
    const records=plain(run(c,'ratings'));
    const byUser=Array.from({length:944},()=>[]),byItem=Array.from({length:1683},()=>[]);
    for(const r of records){byUser[r.userId].push(r);byItem[r.itemId].push(r);}
    // Truth stays in the host, outside the prediction context. Every training
    // rebuild is based on records excluding the target, not a cell-only edit.
    const cases=[];
    const leakage={checkedCases:0,restoredCases:0,targetAbsent:true,meansChecked:true,indexesChecked:true,cachesEmpty:true};
    for(let u=1;u<=943;u++) {
        const target=[...byUser[u]].sort((a,b)=>b.timestamp-a.timestamp||a.itemId-b.itemId)[0];
        const train=records.filter(r=>!(r.userId===u&&r.itemId===target.itemId));
        c.trainingRecords=train;
        run(c,'ratings=trainingRecords;buildRatingMatrix();itemSimilarityCache.clear();similarityMatrixVersion=null;');
        const state=run(c,'({matrix:ratingMatrix,history:ratedMovieIds,raters:itemRaterIds,cacheSize:itemSimilarityCache.size})');
        assert.equal(train.length,99999);
        assert.equal(state.matrix[u][target.itemId],0);
        assert.ok(!state.history[u].includes(target.itemId));
        assert.ok(!state.raters[target.itemId].includes(u));
        assert.equal(state.cacheSize,0);
        assert.equal(run(c,`ratings.some(r=>r.userId===${u}&&r.itemId===${target.itemId})`),false);
        const remainingUser=byUser[u].filter(r=>r.itemId!==target.itemId);
        const remainingItem=byItem[target.itemId].filter(r=>r.userId!==u);
        near(c.getUserMean(u),sum(remainingUser.map(r=>r.rating))/remainingUser.length);
        if(remainingItem.length) near(c.getItemMean(target.itemId),sum(remainingItem.map(r=>r.rating))/remainingItem.length);
        else assert.equal(c.getItemMean(target.itemId),null);
        const neighbors=c.getUserNeighbors(u);
        const user=c.predictUserBasedRating(u,target.itemId,neighbors);
        const item=c.predictItemBasedRating(u,target.itemId);
        assert.ok(user===null||Number.isFinite(user),'STOP: invalid User-Based prediction');
        assert.ok(item===null||Number.isFinite(item),'STOP: invalid Item-Based prediction');
        const userContributorCount=neighbors.filter(n=>state.matrix[n.userId][target.itemId]>0).length;
        const itemContributorCount=state.history[u].filter(i=>c.getItemSimilarity(target.itemId,i)>0).length;
        cases.push({userId:u,movieId:target.itemId,title:run(c,`moviesById.get(${target.itemId}).title`),
            timestamp:target.timestamp,actual:target.rating,trainingHistory:remainingUser.length,
            targetTrainingRaters:remainingItem.length,user,item,userContributorCount,itemContributorCount,
            userMissingReason:user===null?(remainingItem.length?'No target-rating contributor among top-20 neighbors':'Target item has no remaining ratings'):null,
            itemMissingReason:item===null?(remainingItem.length?'No usable similarity to user history':'Target item has no remaining ratings'):null});
        leakage.checkedCases++;
        // Explicit restore and rebuild before proceeding to the next case.
        c.trainingRecords=records;
        run(c,'ratings=trainingRecords;buildRatingMatrix();itemSimilarityCache.clear();similarityMatrixVersion=null;');
        assert.equal(run(c,`ratingMatrix[${u}][${target.itemId}]`),target.rating);
        assert.equal(run(c,'ratings.length'),100000);
        leakage.restoredCases++;
        if(u%100===0)console.log('QUALITY_PROGRESS',u);
    }
    const common=cases.filter(r=>Number.isFinite(r.user)&&Number.isFinite(r.item));
    const distribution=Object.fromEntries([1,2,3,4,5].map(v=>[v,cases.filter(r=>r.actual===v).length]));
    const history=config.quality.historyBins.map(bin=>{
        const rows=cases.filter(r=>inBin(r.trainingHistory,bin));
        const paired=rows.filter(r=>Number.isFinite(r.user)&&Number.isFinite(r.item));
        return {bin:binLabel(bin),cases:rows.length,user:metrics(rows,'user'),item:metrics(rows,'item'),
            common:paired.length,commonUser:metrics(paired,'user'),commonItem:metrics(paired,'item')};
    });
    const summary={sample:{users:943,heldOut:cases.length,ratingDistribution:distribution,
        originalHistory:stats(cases.map(r=>r.trainingHistory+1)),trainingHistory:stats(cases.map(r=>r.trainingHistory))},
        user:metrics(cases,'user'),item:metrics(cases,'item'),common:{count:common.length,user:metrics(common,'user'),item:metrics(common,'item')},
        userRange:range(cases,'user'),itemRange:range(cases,'item'),history,
        sensitivity:{label:config.quality.secondaryLabel,user:metrics(cases,'user',true),item:metrics(cases,'item',true),
            commonUser:metrics(common,'user',true),commonItem:metrics(common,'item',true)},
        failureCases:cases.filter(r=>r.user===null||r.item===null),leakage};
    save('quality',{summary,cases});console.log('QUALITY_SUMMARY',JSON.stringify(summary));
}

function overlapBin(n){return n===0?'0':n===1?'1':n<=4?'2-4':n<=9?'5-9':n<=19?'10-19':n<=49?'20-49':'50+';}
function newBins(){return Object.fromEntries(config.overlap.bins.map(bin=>[bin,{raw:[],weighted:[]}]));}
function addPair(bins,d){const b=bins[overlapBin(d.commonCount)];b.raw.push(d.rawCosine);b.weighted.push(d.weightedSimilarity);}
function summarizeBins(bins){return Object.entries(bins).map(([bin,b])=>({bin,pairs:b.raw.length,raw:stats(b.raw),weighted:stats(b.weighted)}));}
async function overlap(){
    const c=await load(),matrix=run(c,'ratingMatrix'),histories=run(c,'ratedMovieIds');
    const users=newBins(),items=newBins();let checked=0;
    for(let a=1;a<=943;a++)for(let b=a+1;b<=943;b++){
        // Extract only the common coordinates in ascending movie-ID order, then
        // call production cosine. Removed coordinates would be skipped there.
        const shorter=histories[a].length<=histories[b].length?histories[a]:histories[b];
        const x=[],y=[];
        for(const i of shorter)if(matrix[a][i]>0&&matrix[b][i]>0){x.push(matrix[a][i]);y.push(matrix[b][i]);}
        const d=c.getCosineDetails(x,y);addPair(users,d);
        if((a*944+b)%997===0){assert.deepEqual(plain(d),plain(c.getCosineDetails(matrix[a],matrix[b])));checked++;}
    }
    console.log('OVERLAP_USER_PAIRS_COMPLETE');
    for(let a=1;a<=1682;a++)for(let b=a+1;b<=1682;b++)addPair(items,c.getItemCosineDetails(a,b));
    const examples=[[1,88],[1,33],[1,5],[2,172]].map(([a,b])=>({users:[a,b],
        ...plain(c.getCosineDetails(matrix[a],matrix[b])),
        shared:histories[a].filter(i=>matrix[b][i]>0).map(i=>({movieId:i,title:run(c,`moviesById.get(${i}).title`),ratings:[matrix[a][i],matrix[b][i]]}))}));
    near(examples[0].weightedSimilarity,.02);near(examples[1].weightedSimilarity,.06905524926448671);
    assert.equal(examples[2].commonCount,80);assert.equal(examples[2].supportWeight,1);
    const userBins=summarizeBins(users),itemBins=summarizeBins(items);
    assert.equal(sum(userBins.map(b=>b.pairs)),444153);assert.equal(sum(itemBins.map(b=>b.pairs)),1413721);
    // Synthetic entities in the isolated context only.
    run(c,`numUsers=944;numMovies=1683;movies=[...movies,{id:1683,title:'Synthetic new item',genres:[]}];
        buildRatingMatrix();itemSimilarityCache.clear();similarityMatrixVersion=null;`);
    const cold={newUser:{id:944,mean:c.getUserMean(944),
        similarity:plain(run(c,'getCosineDetails(ratingMatrix[944],ratingMatrix[1])')),
        neighbors:plain(c.getUserNeighbors(944)),user:plain(c.getUserBasedRecommendations(944)),item:plain(c.getItemBasedRecommendations(944))},
        newItem:{id:1683,mean:c.getItemMean(1683),similarity:plain(c.getItemCosineDetails(1683,1)),
            userPrediction:c.predictUserBasedRating(1,1683),itemPrediction:c.predictItemBasedRating(1,1683)}};
    assert.equal(cold.newUser.similarity.commonCount,0);assert.equal(cold.newItem.similarity.commonCount,0);
    assert.equal(cold.newUser.user.length,0);assert.equal(cold.newUser.item.length,0);
    assert.equal(cold.newItem.userPrediction,null);assert.equal(cold.newItem.itemPrediction,null);
    const lecture={user:4.33-3.667/2.866,item:3.67+(-2.764/2.543),
        itemRoundedHelper:c.predictFromDeviations(3.67,[{similarity:-.99,rating:5,mean:3.33},
            {similarity:.72,rating:3,mean:3},{similarity:-.84,rating:5,mean:3.67}]),
        label:'Lecture fixtures, not MovieLens evaluation samples; displayed values are rounded'};
    assert.equal(lecture.user.toFixed(2),'3.05');assert.equal(lecture.item.toFixed(2),'2.58');assert.equal(lecture.itemRoundedHelper.toFixed(2),'2.58');
    save('overlap',{userBins,itemBins,examples,cold,lecture,denseRowEquivalenceChecks:checked,
        dimensions:{users:943,items:1682,ratings:100000,userPairs:444153,itemPairs:1413721,sparsityPercent:100*(1-100000/(943*1682))}});
    console.log('OVERLAP_SUMMARY',JSON.stringify({userBins,itemBins,denseRowEquivalenceChecks:checked,cold,lecture}));
}

function validateList(c,u,list){
    assert.ok(list.length<=5);const seen=new Set(),matrix=run(c,'ratingMatrix'),catalog=run(c,'moviesById');
    for(let k=0;k<list.length;k++){
        const r=list[k];assert.ok(catalog.has(r.movieId)&&catalog.get(r.movieId).title===r.title&&Number.isFinite(r.score));
        assert.equal(matrix[u][r.movieId],0);assert.ok(!seen.has(r.movieId));seen.add(r.movieId);
        if(k)assert.ok(list[k-1].score>r.score||list[k-1].score===r.score&&list[k-1].movieId<r.movieId);
    }
}
function historyUsers(c){return Array.from({length:943},(_,i)=>({userId:i+1,history:run(c,`ratedMovieIds[${i+1}].length`)}))
    .sort((a,b)=>a.history-b.history||a.userId-b.userId);}
async function recommendations(){
    const c=await load(),rows=[],sorted=historyUsers(c);
    const examples=[...new Set([1,33,...[0,471,942].map(i=>sorted[i].userId)])];
    for(let u=1;u<=943;u++){
        const user=c.getUserBasedRecommendations(u,5),item=c.getItemBasedRecommendations(u,5);
        validateList(c,u,user);validateList(c,u,item);
        const shared=user.filter(r=>item.some(s=>s.movieId===r.movieId)).map(r=>({movieId:r.movieId,
            userRank:user.findIndex(s=>s.movieId===r.movieId)+1,itemRank:item.findIndex(s=>s.movieId===r.movieId)+1}));
        rows.push({userId:u,history:run(c,`ratedMovieIds[${u}].length`),user:plain(user),item:plain(item),overlap:shared.length,shared});
        if(u%100===0)console.log('RECOMMENDATION_PROGRESS',u);
    }
    function method(key){
        const entries=rows.flatMap(r=>r[key].map(m=>({userId:r.userId,...m}))),ids=[...new Set(entries.map(r=>r.movieId))].sort((a,b)=>a-b);
        return {users:943,recommendations:entries.length,uniqueMovies:ids.length,movieIds:ids,catalogPercent:100*ids.length/1682,
            five:rows.filter(r=>r[key].length===5).length,fewerThanFive:rows.filter(r=>r[key].length<5).length,
            below1:entries.filter(r=>r.score<1).length,above5:entries.filter(r=>r.score>5).length,
            min:Math.min(...entries.map(r=>r.score)),max:Math.max(...entries.map(r=>r.score)),
            largest:entries.sort((a,b)=>b.score-a.score||a.userId-b.userId||a.movieId-b.movieId).slice(0,5)};
    }
    const user=method('user'),item=method('item');
    const summary={user,item,unionMovies:new Set([...user.movieIds,...item.movieIds]).size,
        usersWithAnyOverlap:rows.filter(r=>r.overlap>0).length,completelyDifferent:rows.filter(r=>r.overlap===0).length,
        overlapDistribution:Object.fromEntries([0,1,2,3,4,5].map(n=>[n,rows.filter(r=>r.overlap===n).length])),
        sharedMovieOccurrences:sum(rows.map(r=>r.overlap)),sharedOccurrencesAtDifferentRanks:sum(rows.map(r=>r.shared.filter(s=>s.userRank!==s.itemRank).length)),
        exampleUserIds:examples};
    save('recommendations',{summary,examples:rows.filter(r=>examples.includes(r.userId)),rows});
    console.log('RECOMMENDATION_SUMMARY',JSON.stringify(summary));
    console.log('TOP5_EXAMPLES',JSON.stringify(rows.filter(r=>examples.includes(r.userId))));
}

async function benchmark(){
    const c=await load(),sorted=historyUsers(c);
    const sample=config.benchmark.historyBins.flatMap(bin=>{
        const a=sorted.filter(u=>inBin(u.history,bin));
        return Array.from({length:5},(_,k)=>({...a[Math.floor(k*(a.length-1)/4)],bin:binLabel(bin)}));
    });
    assert.equal(new Set(sample.map(r=>r.userId)).size,20);
    const calls=[],fn={user:c.getUserBasedRecommendations,item:c.getItemBasedRecommendations};
    const clear=()=>run(c,'itemSimilarityCache.clear();similarityMatrixVersion=null;');
    console.log('BENCHMARK_SAMPLE',JSON.stringify(sample));
    for(let position=0;position<sample.length;position++){
        const {userId,history,bin}=sample[position];
        const policies=position%2?['warm-cache','cold-cache']:['cold-cache','warm-cache'];
        let reference;
        for(const policy of policies){
            clear();
            for(let w=0;w<2;w++)for(const method of ['user','item']){
                const results=fn[method](userId,5);validateList(c,userId,results);
                if(w===1){reference??={};reference[method]=plain(results);}
            }
            for(let repeat=0;repeat<5;repeat++){
                const order=(position+repeat)%2?['item','user']:['user','item'];
                for(const method of order){
                    if(policy==='cold-cache')clear();
                    const cacheBefore=run(c,'itemSimilarityCache.size');
                    if(policy==='cold-cache')assert.equal(cacheBefore,0);
                    const start=performance.now();
                    const result=fn[method](userId,5);
                    const milliseconds=performance.now()-start;
                    assert.deepEqual(plain(result),reference[method]);
                    const cacheAfter=run(c,'itemSimilarityCache.size');
                    if(policy==='warm-cache')assert.equal(cacheBefore,cacheAfter);
                    calls.push({userId,history,bin,policy,repeat:repeat+1,method,milliseconds,cacheBefore,cacheAfter});
                }
            }
        }
        console.log('BENCHMARK_USER_COMPLETE',userId);
    }
    const summary={};
    for(const policy of ['cold-cache','warm-cache']){
        summary[policy]={};
        for(const method of ['user','item'])summary[policy][method]=stats(calls.filter(r=>r.policy===policy&&r.method===method).map(r=>r.milliseconds));
    }
    const perUser=sample.map(u=>({...u,results:Object.fromEntries(['cold-cache','warm-cache'].map(policy=>[policy,
        Object.fromEntries(['user','item'].map(method=>[method,stats(calls.filter(r=>r.userId===u.userId&&r.policy===policy&&r.method===method).map(r=>r.milliseconds))]))]))}));
    save('benchmark',{sample,summary,perUser,calls});console.log('BENCHMARK_SUMMARY',JSON.stringify({environment,summary}));
}

function validate(){
    const q=JSON.parse(fs.readFileSync('step3-evidence/quality.json','utf8'));
    const r=JSON.parse(fs.readFileSync('step3-evidence/recommendations.json','utf8'));
    const o=JSON.parse(fs.readFileSync('step3-evidence/overlap.json','utf8'));
    const b=JSON.parse(fs.readFileSync('step3-evidence/benchmark.json','utf8'));
    for(const result of [q,r,o,b]){
        assert.deepEqual(result.protectedHashesBefore,expected);assert.deepEqual(result.protectedHashesAfter,expected);
        assert.equal(result.harnessSHA256,digest('step3-evaluation.js'));
    }
    assert.equal(q.cases.length,943);assert.equal(new Set(q.cases.map(r=>r.userId)).size,943);
    for(const method of ['user','item'])assert.deepEqual(metrics(q.cases,method),q.summary[method]);
    assert.equal(q.summary.leakage.checkedCases,943);assert.equal(q.summary.leakage.restoredCases,943);
    assert.equal(r.rows.length,943);assert.equal(b.calls.length,400);
    for(const policy of ['cold-cache','warm-cache'])for(const method of ['user','item']){
        const samples=b.calls.filter(r=>r.policy===policy&&r.method===method);
        assert.equal(samples.length,100);assert.ok(samples.every(r=>Number.isFinite(r.milliseconds)&&r.milliseconds>=0));
        assert.deepEqual(stats(samples.map(r=>r.milliseconds)),b.summary[policy][method]);
    }
    save('validation',{passed:true,checks:['943 unique held-out users','943 restores and leakage assertions',
        'quality metrics recomputed','943 full-data recommendation rows','400 timing observations and summaries',
        'all result files use identical harness and production hashes'],
        artifactHashes:Object.fromEntries(['quality','recommendations','overlap','benchmark'].map(name=>[name+'.json',digest('step3-evidence/'+name+'.json')]))});
    console.log('PASS all saved evidence and protected-file hashes validated');
}

(async()=>{
    const actions={quality,overlap,recommendations,benchmark,validate};
    assert.ok(actions[process.argv[2]],'Use quality|overlap|recommendations|benchmark|validate');
    console.log('CONFIG',JSON.stringify({command:process.argv.join(' '),environment,config}));
    await actions[process.argv[2]]();
    integrity();
})().catch(error=>{console.error('STOP: evaluation/check failed; no production code changed.',error);process.exitCode=1;});
