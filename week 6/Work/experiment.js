/**
 * Reproducible HW6 experiment/evidence driver. Load after the existing page:
 * await import('./experiment.js'); await HW6_EXPERIMENT.setup();
 * No application function, dataset, MF objective, or test definition is replaced.
 * Experiments call the supplied scorers, recall function, renderers and exporters.
 * Validation is completed and locked before MF training and test comparison.
 */
window.HW6_EXPERIMENT = (() => {
  const evidence = { version: 1, startedAt: new Date().toISOString(), validation: [],
    selection: null, mf: null, test: null, finalLists: [], controlled: [], failures: [] };
  let independent = null;
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  const memory = () => ({ ...tf.memory(), jsHeapUsed: performance.memory?.usedJSHeapSize ?? null });
  const mean = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : 0;
  const equal = (a,b) => JSON.stringify(a) === JSON.stringify(b);
  const sha = async buffer => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
  const modelHash = async () => {
    const model=HW6_MF_MODEL;
    if(!model)return null;
    return { users:await sha(model.userFactors.dataSync().slice().buffer),
      movies:await sha(model.itemFactors.dataSync().slice().buffer), globalMean:model.globalMean };
  };

  async function backend(name) {
    if(name==='wasm' && !tf.findBackend('wasm')) {
      await new Promise((resolve,reject)=>{
        const script=document.createElement('script');
        script.src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.js';
        script.onload=resolve;script.onerror=()=>reject(new Error('WASM backend failed to load'));
        document.head.appendChild(script);
      });
      tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
    }
    await tf.setBackend(name); await tf.ready();
    HW6_STATE.backend=tf.getBackend();
    if(MF_INFO.trainedAt===null)MF_INFO.backend=tf.getBackend();
    hw6InvalidateCaches();
    document.getElementById('tf-status').textContent='TensorFlow.js backend: '+tf.getBackend();
    return tf.getBackend();
  }

  function deriveIndependently() {
    const byUser=new Map();
    for(const r of ratings){if(!byUser.has(r.userId))byUser.set(r.userId,[]);byUser.get(r.userId).push(r);}
    const expected={train:[],val:[],test:[]},splitExcluded=[];
    for(const [userId,rows] of byUser){
      if(rows.length<5){splitExcluded.push(userId);continue;}
      const sorted=rows.slice().sort((a,b)=>a.timestamp-b.timestamp||a.movieId-b.movieId);
      const a=Math.floor(.8*rows.length),b=a+Math.floor(.1*rows.length);
      expected.train.push(...sorted.slice(0,a));expected.val.push(...sorted.slice(a,b));expected.test.push(...sorted.slice(b));
    }
    for(const split of ['train','val','test'])assert(expected[split].length===HW6_SPLIT[split].length&&expected[split].every((r,i)=>r===HW6_SPLIT[split][i]),'Chronological split mismatch: '+split);
    const users=[...new Set(expected.train.map(r=>r.userId))].sort((a,b)=>a-b);
    const candidates=[...new Set(expected.train.map(r=>r.movieId))].sort((a,b)=>a-b),candidateSet=new Set(candidates);
    const seen={train:new Map(),val:new Map(),test:new Map()},grades={train:new Map(),val:new Map(),test:new Map()};
    for(const split of ['train','val','test'])for(const r of expected[split]){
      if(!seen[split].has(r.userId)){seen[split].set(r.userId,new Set());grades[split].set(r.userId,new Map());}
      seen[split].get(r.userId).add(r.movieId);grades[split].get(r.userId).set(r.movieId,r.rating);
    }
    const positivePairs=new Set(),positiveUsers=new Set(),popularity=new Map();
    for(const r of expected.train)if(r.rating>=4){
      const key=r.userId+':'+r.movieId;if(positivePairs.has(key))continue;
      positivePairs.add(key);positiveUsers.add(r.userId);popularity.set(r.movieId,(popularity.get(r.movieId)||0)+1);
    }
    const cohorts={},exclusions={};
    for(const split of ['val','test']){
      const cohort=[],noEligible=[],noPositive=[],unseenIds=new Set();let unseenCount=0;
      for(const userId of users){
        const relevant=[];
        if(!positiveUsers.has(userId))noPositive.push(userId);
        for(const [id,rating] of grades[split].get(userId)||[]){
          if(rating<4)continue;
          if(!candidateSet.has(id)){unseenCount++;unseenIds.add(id);continue;}
          if(seen.train.get(userId)?.has(id)||(split==='test'&&seen.val.get(userId)?.has(id)))continue;
          relevant.push(id);
        }
        if(!relevant.length)noEligible.push(userId);
        else if(positiveUsers.has(userId))cohort.push(userId);
      }
      cohorts[split]=cohort;
      exclusions[split]={candidateUsers:users.length,cohortSize:cohort.length,excludedUnseen:unseenCount,
        excludedNoEligible:noEligible.length,excludedNoPositiveEdge:noPositive.length,
        unseenIds:[...unseenIds].sort((a,b)=>a-b),noEligibleUserIds:noEligible,noPositiveUserIds:noPositive,
        excludedUserUnion:users.filter(u=>!cohort.includes(u)),additionalRestrictions:0};
      assert(equal(cohort,hw6SharedCohort(split)),'Independent cohort mismatch: '+split);
      const app=hw6CohortExclusions(split);for(const key of ['cohortSize','excludedUnseen','excludedNoEligible','excludedNoPositiveEdge'])assert(app[key]===exclusions[split][key],'Exclusion mismatch '+split+' '+key);
    }
    const g=getGraph();assert(equal(users,g.userIds)&&equal(candidates,g.movieIds),'Graph universe mismatch');
    const degree=new Float64Array(g.nodeCount),edges=[];
    for(let u=0;u<users.length;u++)for(let i=0;i<candidates.length;i++){
      const positive=positivePairs.has(users[u]+':'+candidates[i]);assert(g.R[u][i]===(positive?1:0),'Incorrect positive graph edge');
      if(positive){const m=g.nUsers+i;degree[u]++;degree[m]++;edges.push([u,m]);}
    }
    assert(degree.every((d,i)=>d===g.degree[i]),'Graph degrees mismatch');
    const counts={users:byUser.size,movies:movies.length,ratings:ratings.length,
      maxUserId:Math.max(...byUser.keys()),maxMovieId:Math.max(...movies.map(m=>m.id)),
      splitCounts:Object.fromEntries(Object.entries(expected).map(([k,v])=>[k,v.length])),splitExcluded,
      candidateMovies:candidates.length,graphUsers:users.length,graphNodes:g.nodeCount,positiveEdges:positivePairs.size,
      zeroDegreeUsers:users.filter((id,i)=>degree[i]===0),zeroDegreeMovies:candidates.filter((id,i)=>degree[g.nUsers+i]===0),
      sparsity:1-ratings.length/(byUser.size*movies.length),chronologicalSplitVerified:true,allGraphEdgesVerified:true};
    return {counts,cohorts,exclusions,seen,grades,candidates,candidateSet,popularity,edges,degree};
  }

  async function setup() {
    assert(HW6_STATE.loaded,'Wait for dataset loading');
    assert(MF_INFO.trainedAt===null,'Use a fresh page for a reproducible experiment');
    if(!getGraph())hw6HandleBuildGraph();
    await backend('wasm');
    independent=deriveIndependently();
    evidence.environment={userAgent:navigator.userAgent,platform:navigator.platform,tf:tf.version.tfjs,d3:d3.version,
      propagationBackend:tf.getBackend(),wasmBackendVersion:'4.22.0',wasmSimd:tf.env().get('WASM_HAS_SIMD_SUPPORT'),
      hardwareConcurrency:navigator.hardwareConcurrency,jsHeapLimit:performance.memory?.jsHeapSizeLimit??null};
    evidence.dataset=independent.counts;evidence.cohorts=independent.cohorts;evidence.exclusions=independent.exclusions;
    evidence.sourceHashes={};
    for(const name of ['graph.js','script.js','data.js','index.html','style.css','readme.md','u.data','u.item','vendor/d3.min.js']){
      const response=await fetch(name);assert(response.ok,'Fetch failed '+name);evidence.sourceHashes[name]=await sha(await response.arrayBuffer());
    }
    const itemText=await(await fetch('u.item')).text();
    evidence.dataset.metadataReplacementCharacters=(itemText.match(/\ufffd/g)||[]).length;
    evidence.originalTests=runSelfChecks();assert(evidence.originalTests.failed===0&&evidence.originalTests.pending===0,'Original checks not green');
    evidence.fixedUsers=independent.cohorts.test.slice(0,2);
    evidence.userSelection={chosenAt:new Date().toISOString(),rule:'Two smallest common-test-cohort IDs, fixed before validation results or final test metrics; different training-history sizes; same pair as integration verification.',usesTestPerformance:false};
    evidence.setupMemory=memory();
    return {environment:evidence.environment,dataset:evidence.dataset,cohortSizes:{val:independent.cohorts.val.length,test:independent.cohorts.test.length},fixedUsers:evidence.fixedUsers,tests:evidence.originalTests};
  }

  function propagationDiagnostic(propagation,seed,alpha) {
    const p=propagation.scores,g=getGraph(),next=new Float64Array(g.nodeCount);let d=0,sum=0,min=Infinity;
    for(let i=0;i<p.length;i++){sum+=p[i];min=Math.min(min,p[i]);if(independent.degree[i]===0)d+=p[i];}
    for(const [u,m] of independent.edges){next[m]+=alpha*p[u]/independent.degree[u];next[u]+=alpha*p[m]/independent.degree[m];}
    const restart=1-alpha+alpha*d;
    if(seed===null)for(let i=0;i<next.length;i++)next[i]+=restart/next.length;
    else next[seed]+=restart;
    let stationaryL1=0;for(let i=0;i<p.length;i++)stationaryL1+=Math.abs(next[i]-p[i]);
    return {iterations:propagation.iterations,residual:propagation.residual,converged:propagation.converged,
      probabilitySum:sum,minProbability:min,stationaryL1,seedNode:seed,
      rawUserAtSeed:seed===null?null:g.userIds[seed],effectiveAlpha:alpha};
  }

  function auditLists(method,split,alpha,report=null) {
    const started=performance.now(),cohort=independent.cohorts[split],perUser=[];
    const reportByUser=report?new Map(report.perUser.map(r=>[r.userId,r])):null;
    if(report)assert(equal(report.cohort,cohort),'Method changed cohort');
    let globalDiagnostic=null,minCandidateScore=Infinity,maxCandidateScore=-Infinity;
    for(const userId of cohort){
      const excluded=new Set(independent.seen.train.get(userId));
      if(split==='test')for(const id of independent.seen.val.get(userId)||[])excluded.add(id);
      const relevant=[...(independent.grades[split].get(userId)||[])].filter(([id,r])=>r>=4&&independent.candidateSet.has(id)&&!excluded.has(id)).map(([id])=>id).sort((a,b)=>a-b);
      const vector=hw6ScoreVector(method,userId,alpha,split),recs=recommendTopK(method,userId,10,split);
      assert(equal(vector.movieIds,independent.candidates),'Candidate pool mismatch');
      const expected=[];
      for(let i=0;i<vector.movieIds.length;i++){
        const id=vector.movieIds[i],score=vector.scores[i];assert(Number.isFinite(score),'Invalid candidate score');assert(method==='mf'||score>=0,'Negative graph/popularity score');
        minCandidateScore=Math.min(minCandidateScore,score);maxCandidateScore=Math.max(maxCandidateScore,score);
        if(!excluded.has(id))expected.push({id,score});
      }
      expected.sort((a,b)=>b.score-a.score||a.id-b.id);
      assert(equal(recs.map(r=>r.movieId),expected.slice(0,10).map(r=>r.id)),'Incorrect ranking/filtering for '+method+' '+userId);
      assert(recs.length<=10,'Too many recommendations');
      for(let i=0;i<recs.length;i++){
        const r=recs[i],movie=hw6MovieById(r.movieId);
        assert(r.score===expected[i].score,'Score mismatch');
        assert(r.popularity===(independent.popularity.get(r.movieId)||0),'Popularity mismatch');
        assert(r.title===movie.title&&equal(r.genres,movie.genres),'Metadata mismatch');
      }
      const hitIds=recs.filter(r=>relevant.includes(r.movieId)).map(r=>r.movieId);
      if(report){const row=reportByUser.get(userId);assert(row.relevant===relevant.length&&row.hit===hitIds.length,'Per-user recall mismatch');}
      let propagation=null;
      if(method==='ppr'){
        const node=getGraph().nodeOfUser(userId);assert(getGraph().userIds[node]===userId,'Wrong seed mapping');
        propagation=propagationDiagnostic(vector.propagation,node,alpha);
      }else if(method==='pagerank'){
        if(!globalDiagnostic)globalDiagnostic=propagationDiagnostic(vector.propagation,null,alpha);
        propagation=globalDiagnostic;
      }
      if(propagation)assert(propagation.effectiveAlpha===alpha,'Wrong effective alpha');
      perUser.push({userId,relevant:relevant.length,relevantIds:relevant,hit:hitIds.length,hitIds,
        recall:hitIds.length/relevant.length,recommendations:recs.map(r=>({movieId:r.movieId,score:r.score,popularity:r.popularity})),
        meanPopularity:mean(recs.map(r=>r.popularity)),zeroScores:recs.filter(r=>r.score===0).length,propagation});
    }
    const recall=mean(perUser.map(r=>r.recall));if(report)assert(Math.abs(recall-report.recall)<1e-14,'Macro recall mismatch');
    const p=perUser.map(r=>r.propagation).filter(Boolean);
    const summary={method,split,alpha:['ppr','pagerank'].includes(method)?alpha:null,recall,cohortSize:cohort.length,
      ...independent.exclusions[split],meanPopularity:mean(perUser.map(r=>r.meanPopularity)),minCandidateScore,maxCandidateScore,
      nonconverged:p.filter(r=>!r.converged).length,
      minIterations:p.length?Math.min(...p.map(r=>r.iterations)):null,maxIterations:p.length?Math.max(...p.map(r=>r.iterations)):null,
      meanIterations:p.length?mean(p.map(r=>r.iterations)):null,maxResidual:p.length?Math.max(...p.map(r=>r.residual)):null,
      maxMassError:p.length?Math.max(...p.map(r=>Math.abs(r.probabilitySum-1))):null,
      maxStationaryL1:p.length?Math.max(...p.map(r=>r.stationaryL1)):null,
      shortLists:perUser.filter(r=>r.recommendations.length<10).length,zeroScoreLists:perUser.filter(r=>r.zeroScores>0).length,
      auditRuntimeMs:performance.now()-started,backend:tf.getBackend(),allRankingAndFilteringChecksPassed:true};
    return {summary,cohort:cohort.slice(),perUser};
  }

  function validationAlpha(alpha) {
    assert(independent,'Run setup first');assert(!evidence.selection&&!evidence.test,'Validation already locked');
    assert(tf.getBackend()==='wasm','Switch to WASM before the full validation sweep');
    assert(ALPHA_GRID.includes(alpha)&&!evidence.validation.some(r=>r.summary.alpha===alpha),'Invalid/repeated alpha');
    assert(alpha===ALPHA_GRID[evidence.validation.length],'Run ascending prescribed grid');
    hw6SetAlpha(alpha);
    const before=memory(),startedAt=new Date().toISOString(),start=performance.now();
    const report=recallAtK('ppr',10,'val'),recallRuntimeMs=performance.now()-start;
    const audit=auditLists('ppr','val',alpha,report);
    Object.assign(audit.summary,{startedAt,completedAt:new Date().toISOString(),recallRuntimeMs,totalRuntimeMs:performance.now()-start,memoryBefore:before,memoryAfter:memory(),complete:true});
    evidence.validation.push(audit);
    HW6_STATE.alphaSweepRows=evidence.validation.map(r=>({alpha:r.summary.alpha,recall:r.summary.recall,cohortSize:r.summary.cohortSize,error:null}));
    hw6RenderAlphaSweep({rows:HW6_STATE.alphaSweepRows,best:null,note:'Selection pending completion of full grid.'});
    return audit;
  }

  function selectAlpha() {
    assert(equal(evidence.validation.map(r=>r.summary.alpha),ALPHA_GRID),'Complete all four validation alphas first');
    assert(evidence.validation.every(r=>r.summary.complete),'Partial validation');
    assert(!evidence.test,'Test must not precede selection');
    const sorted=evidence.validation.map(r=>r.summary).slice().sort((a,b)=>b.recall-a.recall||a.alpha-b.alpha),alpha=sorted[0].alpha;
    hw6SetAlpha(alpha);
    const g=getGraph(),userId=evidence.fixedUsers[0];
    hw6RunAndRecordConvergence('pagerank',g,alpha);hw6RunAndRecordConvergence('ppr',g,alpha,userId);
    assert(HW6_STATE.convergence.pagerank.alpha===alpha&&HW6_STATE.convergence.ppr.alpha===alpha&&HW6_STATE.convergence.ppr.userId===userId&&hw6ReadAlpha()===alpha,'Selection not applied');
    evidence.selection={at:new Date().toISOString(),alpha,objective:'maximum validation macro Recall@10; exact ties choose smaller alpha',
      validationScores:evidence.validation.map(r=>({alpha:r.summary.alpha,recall:r.summary.recall})),
      testNotYetRun:evidence.test===null,sliderAlpha:hw6ReadAlpha(),effectiveGlobalAlpha:HW6_STATE.convergence.pagerank.alpha,effectivePprAlpha:HW6_STATE.convergence.ppr.alpha,
      representativeRawUser:userId,representativeNode:g.nodeOfUser(userId)};
    hw6RenderAlphaSweep({rows:HW6_STATE.alphaSweepRows,best:alpha,note:''});hw6RenderConvergence();
    return evidence.selection;
  }

  async function benchmarkMf() {
    assert(!HW6_MF_MODEL,'Benchmark before actual MF training');
    const rows=[];
    for(const name of ['cpu','webgl']) {
      await backend(name);
      const tensors=tf.tidy(()=>({
        u:tf.variable(tf.fill([numUsers+1,MF_FACTORS],.01)),
        i:tf.variable(tf.fill([HW6_MAX_MOVIE_ID+1,MF_FACTORS],.01)),
        ui:tf.tensor1d(Array.from({length:512},(_,i)=>i%numUsers+1),'int32'),
        ii:tf.tensor1d(Array.from({length:512},(_,i)=>i%HW6_MAX_MOVIE_ID+1),'int32'),
        y:tf.fill([512],4),
      }));
      const optimizer=tf.train.adam(MF_CONFIG.lr),times=[];
      try {
        for(let step=0;step<3;step++) {
          const start=performance.now();
          const loss=optimizer.minimize(()=>tf.tidy(()=>{
            const pred=tf.gather(tensors.u,tensors.ui).mul(tf.gather(tensors.i,tensors.ii)).sum(1).add(3.5);
            return tf.losses.meanSquaredError(tensors.y,pred).add(tensors.u.square().sum().add(tensors.i.square().sum()).mul(MF_CONFIG.reg));
          }),true,[tensors.u,tensors.i]);
          loss.dataSync();loss.dispose();times.push(performance.now()-start);
        }
      } finally { optimizer.dispose();tf.dispose(tensors); }
      rows.push({backend:name,batchMs:times,estimatedTrainingMinutes:mean(times.slice(1))*Math.ceil(HW6_SPLIT.train.length/MF_CONFIG.batchSize)*MF_CONFIG.epochs/60000});
    }
    evidence.mfBackendBenchmark={rows,synthetic:true,purpose:'runtime feasibility only; no model selection and no validation/test outcomes',memory:memory()};
    return evidence.mfBackendBenchmark;
  }

  async function trainMf(backendName='cpu') {
    assert(evidence.selection&&!evidence.test,'Select alpha before MF and test');assert(!evidence.mf,'MF already measured');
    // WASM lacks UnsortedSegmentSum, used by the supplied gather gradient.
    // CPU/WebGL are standard TF.js backends; the instructor model is unmodified.
    assert(['cpu','webgl'].includes(backendName),'MF requires a backend supporting gather gradients');
    await backend(backendName);
    const before=memory(),startedAt=new Date().toISOString(),start=performance.now();
    await hw6EnsureMfForNeed();
    assert(HW6_MF_MODEL&&MF_INFO.trainedAt&&MF_INFO.history.length===MF_CONFIG.epochs,'MF training incomplete');
    const best=MF_INFO.history.slice().sort((a,b)=>a.valRMSE-b.valRMSE||a.epoch-b.epoch)[0];
    assert(best.epoch===MF_INFO.checkpoint,'MF checkpoint selection mismatch');
    const restoredValidationRMSE=hw6MfRmse(HW6_SPLIT.val);
    assert(Math.abs(restoredValidationRMSE-MF_INFO.validationRMSE)<1e-7,'MF checkpoint not restored');
    const norm=t=>{let sum=0,maxAbs=0;for(const x of t.dataSync()){sum+=x*x;maxAbs=Math.max(maxAbs,Math.abs(x));}return {l2Norm:Math.sqrt(sum),maxAbs};};
    evidence.mf={startedAt,completedAt:new Date().toISOString(),runtimeMs:performance.now()-start,backend:tf.getBackend(),
      info:JSON.parse(JSON.stringify(MF_INFO)),modelType:'dot product plus training global mean, no user/item biases',optimizer:'Adam',
      objective:'batch MSE + 0.05 * (sum(all user factors squared) + sum(all movie factors squared))',
      initialization:'Mulberry32 seed 20261013; Box-Muller normal * 0.05; seeded shuffled batches',
      trainedOn:'training records only',restoredValidationRMSE,globalMean:HW6_MF_MODEL.globalMean,
      userFactorNorm:norm(HW6_MF_MODEL.userFactors),movieFactorNorm:norm(HW6_MF_MODEL.itemFactors),
      frozenModelHash:await modelHash(),memoryBefore:before,memoryAfter:memory()};
    return evidence.mf;
  }

  async function testComparison() {
    assert(evidence.selection&&evidence.mf&&!evidence.test,'Requires selected alpha and frozen MF; do not repeat test selection');
    assert(equal(await modelHash(),evidence.mf.frozenModelHash),'MF changed before test');
    await backend('wasm');hw6SetAlpha(evidence.selection.alpha);
    const alpha=hw6ReadAlpha(),g=getGraph(),start=performance.now(),before=memory(),startedAt=new Date().toISOString();
    assert(alpha===evidence.selection.alpha,'Wrong test alpha');
    hw6RunAndRecordConvergence('pagerank',g,alpha);hw6RunAndRecordConvergence('ppr',g,alpha,evidence.fixedUsers[0]);
    const rows=await hw6RunComparison('test'),comparisonRuntimeMs=performance.now()-start;
    assert(rows.length===4&&rows.every(r=>!r.error&&Number.isFinite(r.recall)),'Four-method comparison incomplete');
    const audits=[];
    for(const row of rows){
      const audit=auditLists(row.method,'test',alpha);
      assert(Math.abs(row.recall-audit.summary.recall)<1e-14,'Runner versus independent recall mismatch');
      assert(row.cohortSize===audit.cohort.length,'Cohort size mismatch');
      assert(Math.abs(row.meanPopularity-audit.summary.meanPopularity)<1e-10,'Mean popularity mismatch');
      if(['ppr','pagerank'].includes(row.method))assert(row.alpha===alpha,'Runner alpha mismatch');
      audits.push(audit);
    }
    assert(audits.every(r=>equal(r.cohort,audits[0].cohort)),'Methods used different users');
    const finalHash=await modelHash();assert(equal(finalHash,evidence.mf.frozenModelHash),'MF changed during test');
    evidence.test={startedAt,completedAt:new Date().toISOString(),alpha,comparisonRuntimeMs,totalRuntimeMs:performance.now()-start,
      rows,audits,modelHashAfterTest:finalHash,memoryBefore:before,memoryAfter:memory(),complete:true};
    hw6RenderComparison(rows);hw6RenderConvergence();
    return {...evidence.test,summary:{alpha,rows,comparisonRuntimeMs,totalRuntimeMs:evidence.test.totalRuntimeMs,
      audits:audits.map(r=>r.summary),modelHashUnchanged:true,memoryBefore:before,memoryAfter:evidence.test.memoryAfter}};
  }

  function collectUsers() {
    assert(evidence.test?.complete,'Complete final test first');
    const g=getGraph(),selected=evidence.selection.alpha;
    const list=(setting,method,userId,alpha)=>{
      if(alpha!==null)hw6SetAlpha(alpha);
      const recs=recommendTopK(method,userId,10,'test');
      const propagation=method==='ppr'?propagationDiagnostic(hw6RunPPR(g,userId,alpha),g.nodeOfUser(userId),alpha):
        method==='pagerank'?propagationDiagnostic(hw6RunPageRank(g,alpha),null,alpha):null;
      hw6RecordRecommendationRows(userId,method,alpha,recs);
      return {setting,method,userId,node:g.nodeOfUser(userId),alpha,recommendations:recs,
        meanPopularity:mean(recs.map(r=>r.popularity)),zeroScores:recs.filter(r=>r.score===0).length,propagation};
    };
    evidence.finalLists=[];
    for(const method of ['popularity','pagerank','ppr','mf'])for(const userId of evidence.fixedUsers)
      evidence.finalLists.push(list('final_selected',method,userId,['ppr','pagerank'].includes(method)?selected:null));
    const [a,b]=evidence.fixedUsers;
    evidence.controlled=[list('baseline_A_085','ppr',a,.85),list('seed_only_B_085','ppr',b,.85),list('alpha_only_A_050','ppr',a,.5)];
    const overlap=(x,y)=>x.recommendations.filter(r=>y.recommendations.some(s=>s.movieId===r.movieId)).map(r=>r.movieId);
    evidence.overlap=Object.fromEntries(['popularity','pagerank','ppr','mf'].map(method=>{
      const lists=evidence.finalLists.filter(r=>r.method===method);return [method,{commonMovieIds:overlap(...lists),count:overlap(...lists).length}];
    }));
    evidence.controlledOverlap={seedVersusBaseline:overlap(evidence.controlled[0],evidence.controlled[1]),alphaVersusBaseline:overlap(evidence.controlled[0],evidence.controlled[2])};
    evidence.histories=evidence.fixedUsers.map(userId=>{
      const rows=HW6_SPLIT.train.filter(r=>r.userId===userId).map(r=>({...r,title:hw6Title(r.movieId),genres:hw6Genres(r.movieId)}));
      const genreCounts={};for(const r of rows)if(r.rating>=4)for(const genre of r.genres)genreCounts[genre]=(genreCounts[genre]||0)+1;
      const test=evidence.test.audits[0].perUser.find(r=>r.userId===userId);
      return {userId,node:g.nodeOfUser(userId),trainingRatings:rows.length,positiveTrainingRatings:rows.filter(r=>r.rating>=4).length,
        positiveGenreCounts:genreCounts,eligibleRelevantTestCount:test.relevant,eligibleRelevantTestIds:test.relevantIds,
        examples:rows.filter(r=>r.rating>=4).slice(0,5).concat(rows.filter(r=>r.rating<=2).slice(0,3)),fullTrainingHistory:rows};
    });
    hw6SetAlpha(selected);
    return {fixedUsers:evidence.fixedUsers,histories:evidence.histories,finalLists:evidence.finalLists,controlled:evidence.controlled,
      overlap:evidence.overlap,controlledOverlap:evidence.controlledOverlap};
  }

  function bundle() {
    assert(evidence.test?.complete&&evidence.finalLists.length===8&&evidence.controlled.length===3,'Experiments incomplete');
    return {evidence,originalExporterOutputs:{verification:serializeVerification(),results:serializeResults(),recommendations:serializeRecommendations()},
      summary:{selectedAlpha:evidence.selection.alpha,mfCheckpoint:evidence.mf.info.checkpoint,
        validation:evidence.validation.map(r=>r.summary),test:evidence.test.audits.map(r=>r.summary),overlap:evidence.overlap}};
  }
  return {evidence,setup,backend,validationAlpha,selectAlpha,benchmarkMf,trainMf,testComparison,collectUsers,bundle};
})();
