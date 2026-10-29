/** Ordered Step 4B experiment. Uses its own model and new evidence paths only. */
window.HW5_MF_REPRODUCTION_EXPERIMENT = (() => {
  const core=HW5_MF_REPRODUCTION;
  const limitation='The original HW5 selected epoch-19 checkpoint weights were unavailable. A new model was trained using the corrected HW5 MF implementation and selected by validation RMSE. This reproduction is not asserted to be numerically identical to the original HW5 model.';
  const checks=[];
  const assert=(condition,message)=>{if(!condition)throw new Error(message);};
  const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
  const memory=()=>({...tf.memory()});
  const digest=async buffer=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),x=>x.toString(16).padStart(2,'0')).join('');
  const modelHash=async w=>({users:await digest(Float32Array.from(w.userFactors).buffer),movies:await digest(Float32Array.from(w.movieFactors).buffer),globalMean:w.globalMean});
  async function save(name,value){
    const response=await fetch('/__reproduction/'+name,{method:'POST',body:JSON.stringify(value)});
    if(!response.ok)throw new Error('Evidence save failed: '+name+' '+await response.text());
    return response.json();
  }
  async function read(path){const r=await fetch(path,{cache:'no-store'});assert(r.ok,'Cannot load '+path);return r.json();}
  async function check(name,fn){
    try {const detail=await fn();checks.push({name,status:'PASS',detail});}
    catch(error){checks.push({name,status:'FAIL',detail:String(error.stack||error)});throw error;}
  }
  const stats=values=>{const avg=mean(values);return {count:values.length,min:Math.min(...values),max:Math.max(...values),mean:avg,
    std:Math.sqrt(mean(values.map(x=>(x-avg)**2))),l2Norm:Math.sqrt(values.reduce((s,x)=>s+x*x,0)),distinct:new Set(values).size};};

  /** Reconstruct chronological order and eligibility without application cohort helpers. */
  function dataAudit(old) {
    const byUser=new Map();for(const r of ratings){if(!byUser.has(r.userId))byUser.set(r.userId,[]);byUser.get(r.userId).push(r);}
    const splits={train:[],val:[],test:[]},splitExcluded=[];
    for(const [u,rs] of byUser){
      if(rs.length<5){splitExcluded.push(u);continue;}
      const sorted=rs.slice().sort((a,b)=>a.timestamp-b.timestamp||a.movieId-b.movieId);
      const a=Math.floor(.8*rs.length),b=a+Math.floor(.1*rs.length);
      splits.train.push(...sorted.slice(0,a));splits.val.push(...sorted.slice(a,b));splits.test.push(...sorted.slice(b));
    }
    for(const name of ['train','val','test'])assert(splits[name].length===HW6_SPLIT[name].length&&splits[name].every((r,i)=>r===HW6_SPLIT[name][i]),'Split order differs: '+name);
    const candidates=[...new Set(splits.train.map(r=>r.movieId))].sort((a,b)=>a-b),candidateSet=new Set(candidates);
    const users=[...new Set(splits.train.map(r=>r.userId))].sort((a,b)=>a-b);
    const seen={train:new Map(),val:new Map(),test:new Map()},grades={train:new Map(),val:new Map(),test:new Map()};
    for(const name of ['train','val','test'])for(const r of splits[name]){
      if(!seen[name].has(r.userId)){seen[name].set(r.userId,new Set());grades[name].set(r.userId,new Map());}
      seen[name].get(r.userId).add(r.movieId);grades[name].get(r.userId).set(r.movieId,r.rating);
    }
    const pairs=new Set(),positiveUsers=new Set(),popularity=new Map();
    for(const r of splits.train)if(r.rating>=4&&!pairs.has(r.userId+':'+r.movieId)){
      pairs.add(r.userId+':'+r.movieId);positiveUsers.add(r.userId);popularity.set(r.movieId,(popularity.get(r.movieId)||0)+1);
    }
    const cohort={},relevant={},exclusions={};
    const excluded=(name,u)=>new Set([...(seen.train.get(u)||[]),...(name==='test'?(seen.val.get(u)||[]):[])]);
    for(const name of ['val','test']){
      relevant[name]=new Map();const noEligible=[],noPositive=[],unseen=new Set();let unseenCount=0;
      for(const u of users){
        const ban=excluded(name,u),eligible=[];
        for(const [i,r] of grades[name].get(u)||[])if(r>=4){
          if(!candidateSet.has(i)){unseen.add(i);unseenCount++;}
          else if(!ban.has(i))eligible.push(i);
        }
        relevant[name].set(u,eligible.sort((a,b)=>a-b));
        if(!eligible.length)noEligible.push(u);if(!positiveUsers.has(u))noPositive.push(u);
      }
      cohort[name]=users.filter(u=>positiveUsers.has(u)&&relevant[name].get(u).length);
      assert(equal(cohort[name],old.cohorts[name]),'Cohort changed: stop before comparing methods ('+name+')');
      exclusions[name]={candidateUsers:users.length,cohortSize:cohort[name].length,excludedNoEligible:noEligible.length,
        excludedNoPositiveEdge:noPositive.length,excludedUnseen:unseenCount,unseenIds:[...unseen].sort((a,b)=>a-b),
        excludedUserIds:users.filter(u=>!cohort[name].includes(u)),noEligibleUserIds:noEligible,noPositiveUserIds:noPositive};
    }
    assert(equal(candidates,HW6_STATE.candidateIds),'Candidate IDs differ');
    const counts={users:users.length,movies:movies.length,ratings:ratings.length,
      splits:Object.fromEntries(Object.entries(splits).map(([k,v])=>[k,v.length])),splitExcluded,
      candidates:candidates.length,positiveTrainingEdges:pairs.size};
    assert(counts.users===943&&counts.movies===1682&&counts.ratings===100000&&counts.candidates===1611&&pairs.size===45602,'Unexpected dataset counts');
    return {splits,candidates,candidateSet,users,seen,grades,popularity,positiveUsers,cohort,relevant,excluded,exclusions,counts};
  }

  async function prechecks() {
    const start=memory();
    await check('HW5 clipped predictions, raw ranking scores, unknown-ID fallback',()=>{
      core.restoreModel({userShape:[2,2],movieShape:[3,2],userFactors:[0,0,1,2],movieFactors:[0,0,3,4,-3,-4],globalMean:3});
      const scores=core.scoreAll(1);
      assert(core.predictRating(1,1)===5&&core.predictRating(1,2)===1&&core.predictRating(999,1)===3,'Prediction contract failed');
      assert(scores[1]===14&&scores[2]===-8,'Raw recommendation score was clipped/converted');
      core.disposeModel();return {clippedHigh:5,clippedLow:1,unknownFallback:3,rawHigh:14,rawLow:-8};
    });
    await check('Checkpoint tie chooses earliest epoch without mutating input',()=>{
      const rows=[{epoch:4,valRMSE:.3},{epoch:1,valRMSE:.3},{epoch:2,valRMSE:.4}],snapshot=JSON.stringify(rows);
      assert(core.selectBestCheckpoint(rows).epoch===1&&JSON.stringify(rows)===snapshot,'Tie selection failed');
      return {selectedEpoch:1};
    });
    await check('HW6 filtering, exact ties, and finite negative MF scores',()=>{
      const scores=[0,3,3,8,-2,0,1],candidates=[6,4,5,2,3,1];
      const list=core.rankScores(scores,candidates,new Set([3,6]),10);
      assert(equal(list.map(r=>r.movieId),[1,2,5,4]),'Filtering/tie fixture failed');return list;
    });
    await check('Actual training updates match independent batch-row-L2 Adam calculation and replay',async()=>{
      const initial={userShape:[3,2],movieShape:[3,2],userFactors:[2,3,.4,-.2,3,4],movieFactors:[5,6,.3,.1,-.1,.5],globalMean:3};
      const rows=[{userId:1,movieId:1,rating:3.4},{userId:1,movieId:2,rating:2.5}];
      core.restoreModel(initial);const initialRounded=core.exportModel();
      const options={epochs:2,lr:.01,reg:.05,batchSize:2};let actual;
      const optimizerProbe=tf.train.adam(options.lr),adam=optimizerProbe.getConfig();optimizerProbe.dispose();
      await core.trainMF(rows,rows,options,async()=>{actual=core.exportModel();});
      const U=initialRounded.userFactors.slice(),I=initialRounded.movieFactors.slice();
      const mU=U.map(()=>0),vU=U.map(()=>0),mI=I.map(()=>0),vI=I.map(()=>0);
      let initialLoss;
      for(let step=1;step<=2;step++){
        const gu=U.map(()=>0),gi=I.map(()=>0);let mse=0,l2=0;
        for(const row of rows){
          const u=row.userId*2,i=row.movieId*2,error=3+U[u]*I[i]+U[u+1]*I[i+1]-row.rating;
          mse+=error*error/2;
          for(let f=0;f<2;f++){
            l2+=.05*(U[u+f]**2+I[i+f]**2)/2;
            gu[u+f]+=error*I[i+f]+.05*U[u+f];gi[i+f]+=error*U[u+f]+.05*I[i+f];
          }
        }
        if(step===1)initialLoss={mse,batchMeanL2:l2,total:mse+l2};
        for(const [values,gradient,m,v] of [[U,gu,mU,vU],[I,gi,mI,vI]])for(let i=0;i<values.length;i++){
          m[i]=.9*m[i]+.1*gradient[i];v[i]=.999*v[i]+.001*gradient[i]**2;
          values[i]-=.01*(m[i]/(1-.9**step))/(Math.sqrt(v[i]/(1-.999**step))+adam.epsilon);
        }
      }
      const maxError=Math.max(...U.map((x,i)=>Math.abs(x-actual.userFactors[i])),...I.map((x,i)=>Math.abs(x-actual.movieFactors[i])));
      assert(maxError<2e-5,'Adam/objective mismatch: '+maxError);
      assert(equal(actual.userFactors.slice(0,2),initialRounded.userFactors.slice(0,2))&&equal(actual.userFactors.slice(4),initialRounded.userFactors.slice(4))&&equal(actual.movieFactors.slice(0,2),initialRounded.movieFactors.slice(0,2)),'Unselected rows were penalized');
      core.restoreModel(initialRounded);let replay;
      await core.trainMF(rows,rows,options,async()=>{replay=core.exportModel();});
      assert(equal(actual,replay),'Same initial factors/order did not replay exactly');
      core.disposeModel();return {initialLoss,adam,maxAbsoluteUpdateError:maxError,untouchedRowsUnchanged:true,twoEpochReplayExact:true};
    });
    await check('Checkpoint restoration returns an earlier epoch after validation worsens',async()=>{
      core.restoreModel({userShape:[2,2],movieShape:[2,2],userFactors:[0,0,1,0],movieFactors:[0,0,1,0],globalMean:0});
      const train=[{userId:1,movieId:1,rating:5}],val=[{userId:1,movieId:1,rating:1}],snapshots=[];
      const history=await core.trainMF(train,val,{epochs:3,lr:.1,reg:0,batchSize:1},async()=>{snapshots.push(core.exportModel());});
      const best=core.selectBestCheckpoint(history);
      assert(best.epoch===1&&history[2].valRMSE>best.valRMSE,'Fixture did not exercise an earlier checkpoint');
      assert(equal(core.exportModel(),snapshots[best.epoch-1]),'Earlier factors were not restored');
      core.disposeModel();return {history,selectedEpoch:best.epoch,exactFactorRestoration:true};
    });
    const end=memory();assert(start.numTensors===end.numTensors&&start.numDataBuffers===end.numDataBuffers,'Prechecks leaked tensors');
    return {checks:checks.slice(),memoryBefore:start,memoryAfter:end};
  }

  function predictionProbe() {
    return HW6_SPLIT.train.slice(0,20).concat(HW6_SPLIT.val.slice(0,20)).map(r=>({userId:r.userId,movieId:r.movieId,prediction:core.predictRating(r.userId,r.movieId)}));
  }

  function independentRMSE(weights,records) {
    let error=0;const F=weights.userShape[1];
    for(const r of records){let dot=0;for(let f=0;f<F;f++)dot+=weights.userFactors[r.userId*F+f]*weights.movieFactors[r.movieId*F+f];
      const pred=Math.min(5,Math.max(1,weights.globalMean+dot));error+=(pred-r.rating)**2;}
    return Math.sqrt(error/records.length);
  }

  /** Evaluate only the frozen reproduction; old graph-method outputs are reused. */
  async function evaluateSplit(name,data,old) {
    const before=memory(),start=performance.now(),perUser=[],cohort=data.cohort[name];
    const movieMap=new Map(movies.map(m=>[m.id,m]));let checkedScores=0,minScore=Infinity,maxScore=-Infinity,maxIndependentError=0;
    const weights=core.exportModel();
    for(const userId of cohort){
      const scores=core.scoreAll(userId);
      for(const id of data.candidates){assert(Number.isFinite(scores[id]),'Cannot score required candidate '+userId+':'+id);checkedScores++;minScore=Math.min(minScore,scores[id]);maxScore=Math.max(maxScore,scores[id]);}
      const excluded=data.excluded(name,userId),ranked=core.rankScores(scores,data.candidates,excluded,10);
      assert(ranked.length===Math.min(10,data.candidates.filter(id=>!excluded.has(id)).length),'Incorrect list length');
      const relevant=data.relevant[name].get(userId),hitIds=ranked.filter(r=>relevant.includes(r.movieId)).map(r=>r.movieId);
      const recommendations=ranked.map(r=>({...r,title:movieMap.get(r.movieId).title,genres:movieMap.get(r.movieId).genres,popularity:data.popularity.get(r.movieId)||0}));
      for(const r of recommendations)assert(data.candidateSet.has(r.movieId)&&!excluded.has(r.movieId),'Candidate/exclusion violation');
      if(old.fixedUsers.includes(userId))for(const id of data.candidates){
        let dot=0;for(let f=0;f<16;f++)dot+=weights.userFactors[userId*16+f]*weights.movieFactors[id*16+f];
        maxIndependentError=Math.max(maxIndependentError,Math.abs(weights.globalMean+dot-scores[id]));
      }
      perUser.push({userId,relevant:relevant.length,relevantIds:relevant,hit:hitIds.length,hitIds,recall:hitIds.length/relevant.length,
        recommendations,meanPopularity:mean(recommendations.map(r=>r.popularity)),zeroScores:recommendations.filter(r=>r.score===0).length,
        scoreStats:stats(data.candidates.map(i=>scores[i]))});
      if(perUser.length%64===0)await tf.nextFrame();
    }
    assert(maxIndependentError<2e-5,'Independent dot product mismatch');
    const after=memory();assert(before.numTensors===after.numTensors&&before.numDataBuffers===after.numDataBuffers,'Evaluation leaked tensors');
    const summary={label:core.label,method:'mf_hw5_reproduction',split:name,recall:mean(perUser.map(r=>r.recall)),
      cohortSize:cohort.length,candidateCount:data.candidates.length,meanPopularity:mean(perUser.map(r=>r.meanPopularity)),
      ...data.exclusions[name],shortLists:perUser.filter(r=>r.recommendations.length<10).length,
      zeroScoreLists:perUser.filter(r=>r.zeroScores>0).length,checkedCandidateScores:checkedScores,minScore,maxScore,
      maxIndependentScoreError:maxIndependentError,cohortIdenticalToExisting:equal(cohort,old.cohorts[name]),
      runtimeMs:performance.now()-start,backend:tf.getBackend(),memoryBefore:before,memoryAfter:after};
    return {summary,cohort,perUser};
  }

  async function run() {
    assert(HW6_STATE.loaded,'Dataset not loaded');
    assert(tf.version.tfjs==='4.22.0','Unexpected TensorFlow.js version');
    await tf.setBackend('webgl');await tf.ready();
    HW6_STATE.backend=tf.getBackend();if(!getGraph())hw6HandleBuildGraph();
    assert(HW6_MF_MODEL===null&&MF_INFO.trainedAt===null,'Use a fresh page; supplied reference must remain untrained');
    const protectedNames=['buildGraph','rowNormalize','pageRank','personalizedPageRank','movieScores','recommendTopK','recallAtK','runSelfChecks','hw6TrainMf'];
    const globalsBefore=Object.fromEntries(protectedNames.map(n=>[n,window[n].toString()]));
    const originalBefore=runSelfChecks();assert(originalBefore.failed===0&&originalBefore.pending===0,'Original prechecks failed');
    const oldBundle=await read('evidence/bundle.json'),old=oldBundle.evidence;
    assert(old.selection.alpha===.5&&old.originalTests.passed===12&&old.test.complete,'Prior evidence incomplete');
    const sourceHashes={};
    for(const [name,hash] of Object.entries(old.sourceHashes)){
      const response=await fetch(name,{cache:'no-store'});sourceHashes[name]=await digest(await response.arrayBuffer());assert(sourceHashes[name]===hash,'Prior evidence source mismatch '+name);
    }
    const implementationHashes={};for(const name of ['hw5_mf_reproduction.js','hw5_mf_reproduction_experiment.js'])implementationHashes[name]=await digest(await(await fetch(name,{cache:'no-store'})).arrayBuffer());
    const data=dataAudit(old);
    HW5_REPRO_JOB.stage='verification';
    const pre=await prechecks();await save('pretraining_verification.json',{...pre,originalBefore});
    await save('dataset_audit_verified.json',{counts:data.counts,cohorts:data.cohort,exclusions:data.exclusions,candidateIds:data.candidates,
      sourceHashes,implementationHashes,trainingOnly:true,chronologicalSplitAndOrderMatch:true,sharedUserIdsMatchExisting:true,
      fixedUsers:old.fixedUsers,pprAlpha:old.selection.alpha});
    const optimizerProbe=tf.train.adam(core.config.lr),optimizerConfiguration=optimizerProbe.getConfig();optimizerProbe.dispose();
    const beforeInit=memory();core.initializeFactors(data.splits.train,16);const initial=core.exportModel();
    assert(equal(initial.userShape,[944,16])&&equal(initial.movieShape,[1683,16]),'Wrong factor shapes/indexing');
    const initStats={users:stats(initial.userFactors),movies:stats(initial.movieFactors)};
    for(const s of Object.values(initStats))assert(Math.abs(s.mean)<.005&&s.std>.045&&s.std<.055,'Unexpected initialization');
    assert(initial.globalMean===data.splits.train.reduce((s,r)=>s+r.rating,0)/data.splits.train.length,'Training mean mismatch');
    const initialArtifact=await save('initial_weights.json',initial),initialHash=await modelHash(initial);
    await save('initialization.json',{policy:'Original unseeded tf.randomNormal(shape,0,0.05)',seed:null,
      initialArtifact,initialHash,statistics:initStats,memoryBefore:beforeInit,memoryAfter:memory(),
      reproducibility:'Actual starting arrays archived for same-backend replay; independent full retraining and cross-device bitwise identity are not claimed.'});
    const epochHashes=[],startedAt=new Date().toISOString(),start=performance.now();
    const history=await core.trainMF(data.splits.train,data.splits.val,core.config,async row=>{
      assert(Number.isFinite(row.trainRMSE)&&Number.isFinite(row.valRMSE),'Nonfinite training metric');
      const fingerprint=await modelHash(core.exportModel());epochHashes.push({epoch:row.epoch,...fingerprint});
      await save('epoch_'+String(row.epoch).padStart(2,'0')+'.json',{...row,at:new Date().toISOString(),fingerprint});
      HW5_REPRO_JOB.stage='training';HW5_REPRO_JOB.epoch=row.epoch;HW5_REPRO_JOB.validationRMSE=row.valRMSE;
      await tf.nextFrame();
    });
    const trainingRuntimeMs=performance.now()-start,best=core.selectBestCheckpoint(history),selectedWeights=core.exportModel();
    assert(history.length===20,'Incomplete training');
    const selectedHash=await modelHash(selectedWeights),bestHash=epochHashes[best.epoch-1];
    assert(selectedHash.users===bestHash.users&&selectedHash.movies===bestHash.movies,'Best epoch factors not restored');
    const restoredVal=core.computeRMSEFor(data.splits.val);assert(restoredVal===best.valRMSE,'Restored checkpoint metric mismatch');
    const modelArtifact=await save('model_weights.json',selectedWeights);
    await save('training_history.json',{label:core.label,config:core.config,history,epochHashes,selected:best,startedAt,completedAt:new Date().toISOString(),trainingRuntimeMs});
    const rawBefore=Object.fromEntries(old.fixedUsers.map(u=>[u,core.scoreAll(u)])),predBefore=predictionProbe();
    core.disposeModel();const savedWeights=await read(modelArtifact.path);core.restoreModel(savedWeights);
    assert(equal(core.exportModel(),selectedWeights),'Saved/restored factors changed');
    assert(equal(predBefore,predictionProbe()),'Saved/restored clipped predictions changed');
    for(const u of old.fixedUsers)assert(equal(rawBefore[u],core.scoreAll(u)),'Saved/restored raw candidate predictions changed');
    assert(core.computeRMSEFor(data.splits.val)===best.valRMSE,'Disk restoration validation mismatch');
    await save('restoration_verification.json',{exactFactorEquality:true,float32Hashes:selectedHash,
      clippedPredictionPairs:predBefore.length,rawPredictionPairs:old.fixedUsers.length*selectedWeights.movieShape[0],
      exactPredictionEquality:true,validationRMSE:best.valRMSE,restoredFrom:modelArtifact});
    const independentVal=independentRMSE(selectedWeights,data.splits.val);
    assert(Math.abs(independentVal-best.valRMSE)<2e-5,'Independent validation RMSE differs');
    const gl=tf.backend().gpgpu?.gl,debug=gl?.getExtension('WEBGL_debug_renderer_info');
    const metadata={label:core.label,limitation,originalCheckpointRecovered:false,config:core.config,
      selectedEpoch:best.epoch,validationRMSE:best.valRMSE,restoredValidationRMSE:restoredVal,independentValidationRMSE:independentVal,
      trainingRMSE:best.trainRMSE,trainingHistory:history,trainingRuntimeMs,startedAt,selectedAt:new Date().toISOString(),
      tfVersion:tf.version.tfjs,dependency:window.HW5_REPRO_DEPENDENCY,backend:tf.getBackend(),userAgent:navigator.userAgent,
      gpuRenderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null,
      indexing:{user:'raw userId row; row zero allocated but never a candidate',movie:'raw movieId row; metadata max ID + 1 rows',userShape:selectedWeights.userShape,movieShape:selectedWeights.movieShape},
      globalMean:selectedWeights.globalMean,ratingPrediction:'clamp(trainingMean + dot(userFactors, movieFactors),1,5)',
      rankingScore:'raw trainingMean + dot, identical to corrected HW5 recommendTopK; not clipped and not a probability',
      optimizer:'tf.train.adam(0.01), default beta1/beta2/epsilon',optimizerConfiguration,regularization:'0.05 * mean_batch(||u||^2 + ||i||^2)',
      initialization:{type:'tf.randomNormal',mean:0,std:.05,seed:null,archivedInitialArtifact:initialArtifact},
      batchOrder:'original per-user split concatenation; ordered contiguous batches; no shuffle',
      sourceHashes,implementationHashes,modelArtifact,modelFloat32Hashes:selectedHash,initialHash,
      factorStats:{users:stats(selectedWeights.userFactors),movies:stats(selectedWeights.movieFactors)},
      memoryBeforeInitialization:beforeInit,memoryAfterTraining:memory(),testNotUsedForSelection:true};
    await save('model_metadata.json',metadata);
    HW5_REPRO_JOB.stage='evaluation';
    const validation=await evaluateSplit('val',data,old);await save('validation_evaluation.json',validation);
    const test=await evaluateSplit('test',data,old);await save('test_evaluation.json',test);
    const testRMSE=core.computeRMSEFor(data.splits.test),independentTest=independentRMSE(selectedWeights,data.splits.test);
    assert(Math.abs(testRMSE-independentTest)<2e-5,'Independent test RMSE differs');
    assert(equal(await modelHash(core.exportModel()),selectedHash),'Model changed during test');
    const lists=old.fixedUsers.map(u=>({userId:u,method:'mf_hw5_reproduction',label:core.label,selectedEpoch:best.epoch,
      ...test.perUser.find(r=>r.userId===u)}));
    const commonIds=lists[0].recommendations.filter(r=>lists[1].recommendations.some(s=>s.movieId===r.movieId)).map(r=>r.movieId);
    await save('two_users.json',{users:old.fixedUsers,lists,overlap:{commonMovieIds:commonIds,count:commonIds.length},
      historicalSource:'evidence/bundle.json',historicalLists:old.finalLists,controlledPprUnchanged:old.controlled,
      histories:old.histories,existingOverlap:old.overlap});
    const originalAfter=runSelfChecks();assert(originalAfter.failed===0&&originalAfter.pending===0,'Original HW6 postchecks failed');
    assert(protectedNames.every(n=>window[n].toString()===globalsBefore[n]),'An existing function was replaced');
    assert(HW6_MF_MODEL===null&&MF_INFO.trainedAt===null,'Supplied MF reference was altered/trained');
    assert(memory().numTensors===beforeInit.numTensors+2,'Expected exactly two retained factor tensors');
    await save('verification.json',{label:core.label,checks,originalBefore,originalAfter,
      copiedAlgorithmSourceVerified:'provenance.json',initializationVerified:true,selectedEpoch:best.epoch,
      bestEpochFingerprintMatchesRestoredModel:true,savedRestorationVerified:true,
      candidateScoreability:{val:validation.summary.checkedCandidateScores,test:test.summary.checkedCandidateScores},
      sharedCohortIdentical:true,ascendingIdTiesVerified:true,allFilteringChecksPassed:true,
      noTestBasedSelection:true,noOriginalGlobalsReplaced:true,suppliedReferenceUntouched:true,
      memoryBefore:beforeInit,memoryAfter:memory(),expectedPersistentTensors:2,
      shortDeterministicReplayVerified:true,fullIndependentTrainingReplayRun:false,
      originalHw5ModelRecovered:false,limitation});
    const summary={label:core.label,limitation,selectedEpoch:best.epoch,validationRMSE:best.valRMSE,testRMSE,
      independentTestRMSE:independentTest,validationRecall:validation.summary.recall,testRecall:test.summary.recall,
      sharedTestUsers:test.cohort.length,candidateCount:data.candidates.length,meanPopularity:test.summary.meanPopularity,
      overlap:{commonMovieIds:commonIds,count:commonIds.length},trainingRuntimeMs,
      originalSelfChecks:{passed:originalAfter.passed,failed:originalAfter.failed,pending:originalAfter.pending}};
    await save('summary.json',summary);HW5_REPRO_JOB.stage='complete';return summary;
  }
  return {run};
})();
