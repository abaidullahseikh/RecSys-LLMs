/**
 * Corrected HW5 MF reproduction, isolated from all HW6 implementations.
 * Raw IDs index factor rows (including unused row zero); recommendations never
 * include padding. Initialization, objective, Adam, batch order, clipped RMSE,
 * checkpoint ties and restoration are copied from corrected Week 5/script.js.
 * Only trainMF's async epoch notification is new; it does not alter updates.
 * The HW6 adapter ranks the ORIGINAL raw mean-plus-dot recommendation scores,
 * with training-only candidates and explicit ascending-ID ties. Rating
 * predictions and RMSE remain clipped to [1,5], exactly as in HW5.
 */
window.HW5_MF_REPRODUCTION = (() => {
  const NUM_FACTORS = 16;
  let model = null;
  let trainingHistory = [];
  const label = 'Corrected HW5 MF reproduction';
  const config = Object.freeze({factors:16,epochs:20,lr:0.01,reg:0.05,batchSize:64});

  function isTfModel(candidate) {
    return (
      !!candidate &&
      !!candidate.userFactors &&
      typeof candidate.userFactors.dataSync === "function" &&
      !!candidate.itemFactors &&
      typeof candidate.itemFactors.dataSync === "function"
    );
  }

  function clamp(value, lo, hi) {
    return Math.min(hi, Math.max(lo, value));
  }

  function predictRating(userId, movieId) {
    if (model === null) return 0;
    if (
      !Number.isInteger(userId) || userId < 0 || userId >= model.userFactors.shape[0] ||
      !Number.isInteger(movieId) || movieId < 0 || movieId >= model.itemFactors.shape[0]
    ) return model.globalMean;

    return tf.tidy(() => {
      const f = model.userFactors.shape[1];
      const u = model.userFactors.slice([userId, 0], [1, f]);
      const i = model.itemFactors.slice([movieId, 0], [1, f]);
      return clamp(model.globalMean + u.mul(i).sum().dataSync()[0], 1, 5);
    });
  }

  function initializeFactors(train, numFactors = NUM_FACTORS) {
    if (model) tf.dispose([model.userFactors, model.itemFactors]);
    model = null;
    const globalMean = train.length
      ? train.reduce((sum, r) => sum + r.rating, 0) / train.length
      : 0;
    const maxMovieId = movies.reduce((max, movie) => Math.max(max, movie.id), 0);
    model = tf.tidy(() => ({
      userFactors: tf.variable(tf.randomNormal([numUsers + 1, numFactors], 0, 0.05)),
      itemFactors: tf.variable(tf.randomNormal([maxMovieId + 1, numFactors], 0, 0.05)),
      globalMean,
    }));
    return model;
  }

  function computeRMSEFor(split) {
    if (!split || split.length === 0) return 0;

    if (isTfModel(model)) {
      return tf.tidy(() => {
        const n = split.length;
        const uRows = model.userFactors.shape[0];
        const iRows = model.itemFactors.shape[0];
        const userIds = new Int32Array(n);
        const movieIds = new Int32Array(n);
        const actual = new Float32Array(n);
        const valid = new Float32Array(n);

        for (let k = 0; k < n; k++) {
          const r = split[k];
          userIds[k] = r.userId;
          movieIds[k] = r.movieId;
          actual[k] = r.rating;
          const uOk = r.userId >= 0 && r.userId < uRows;
          const iOk = r.movieId >= 0 && r.movieId < iRows;
          valid[k] = uOk && iOk ? 1 : 0;
        }

        const uIdx = tf.clipByValue(tf.tensor1d(userIds, "int32"), 0, Math.max(0, uRows - 1));
        const iIdx = tf.clipByValue(tf.tensor1d(movieIds, "int32"), 0, Math.max(0, iRows - 1));
        const u = tf.gather(model.userFactors, uIdx);
        const i = tf.gather(model.itemFactors, iIdx);
        const dot = tf.sum(tf.mul(u, i), 1);
        const mask = tf.tensor1d(valid, "float32");
        const pred = tf.clipByValue(dot.mul(mask).add(model.globalMean), 1, 5);
        const diff = pred.sub(tf.tensor1d(actual, "float32"));
        return Math.sqrt(tf.mean(diff.square()).dataSync()[0]);
      });
    }

    let sum = 0;
    try {
      for (const r of split) {
        const diff = predictRating(r.userId, r.movieId) - r.rating;
        sum += diff * diff;
      }
    } catch (error) {
      return Number.NaN;
    }
    return Math.sqrt(sum / split.length);
  }

  /** HW5 trainMF, with an awaited observer after each completed epoch. */
  async function trainMF(train, val, opts, onEpoch = async () => {}) {
    trainingHistory = [];
    if (train.length === 0) return trainingHistory;

    const { epochs, lr, reg } = opts;
    const batchSize = Math.max(1, Math.floor(opts.batchSize));
    const optimizer = tf.train.adam(lr);
    const userIds = Int32Array.from(train, (r) => r.userId);
    const itemIds = Int32Array.from(train, (r) => r.movieId);
    const targets = Float32Array.from(train, (r) => r.rating);
    let bestRow = null;
    let bestUsers = null;
    let bestItems = null;

    try {
      for (let epoch = 1; epoch <= epochs; epoch++) {
        for (let start = 0; start < train.length; start += batchSize) {
          const end = Math.min(start + batchSize, train.length);
          tf.tidy(() => {
            const userIdx = tf.tensor1d(userIds.subarray(start, end), "int32");
            const itemIdx = tf.tensor1d(itemIds.subarray(start, end), "int32");
            const actual = tf.tensor1d(targets.subarray(start, end));
            // returnCost=false returns no loss tensor; tidy releases intermediates.
            optimizer.minimize(() => {
              const u = tf.gather(model.userFactors, userIdx);
              const i = tf.gather(model.itemFactors, itemIdx);
              const pred = tf.sum(u.mul(i), 1).add(model.globalMean);
              const mse = tf.losses.meanSquaredError(actual, pred);
              // Average squared row norms over this batch, matching the MSE scale.
              // Penalize participating rows, not every factor at every batch update.
              const l2 = u.square().sum(1).add(i.square().sum(1)).mean().mul(reg);
              return mse.add(l2);
            }, false, [model.userFactors, model.itemFactors]);
          });
        }

        const row = {
          epoch,
          trainRMSE: computeRMSEFor(train),
          valRMSE: computeRMSEFor(val),
        };
        trainingHistory.push(row);
        // Epochs are ascending: strict improvement preserves the earliest tie.
        if (bestRow === null || row.valRMSE < bestRow.valRMSE) {
          tf.dispose([bestUsers, bestItems]);
          bestUsers = model.userFactors.clone();
          bestItems = model.itemFactors.clone();
          bestRow = row;
        }
        await onEpoch(row);
      }

      if (bestRow) {
        model.userFactors.assign(bestUsers);
        model.itemFactors.assign(bestItems);
      }
      return trainingHistory;
    } finally {
      tf.dispose([bestUsers, bestItems]);
      optimizer.dispose();
    }
  }

  function selectBestCheckpoint(history) {
    if (!history || history.length === 0) return null;
    let best = history[0];
    for (const row of history) {
      if (
        row.valRMSE < best.valRMSE ||
        (row.valRMSE === best.valRMSE && row.epoch < best.epoch)
      ) best = row;
    }
    return best;
  }

  /** All raw movie-ID scores, using HW5 recommendTopK's exact score expression. */
  function scoreAll(userId) {
    if (!model || !Number.isInteger(userId) || userId < 0 || userId >= model.userFactors.shape[0]) {
      throw new Error('MF model unavailable or user ID outside factor rows');
    }
    return tf.tidy(() => {
      const rows = model.itemFactors.shape[0];
      const userRow = model.userFactors.slice([userId, 0], [1, model.userFactors.shape[1]]);
      const scores = model.itemFactors.matMul(userRow, false, true)
        .reshape([rows]).add(model.globalMean);
      return Array.from(scores.dataSync());
    });
  }

  /** HW6 retrieval adapter; preserves raw MF scores, applies HW6 pool/exclusions. */
  function rankScores(scores, candidateIds, excluded, K=10) {
    return candidateIds.filter(id=>!excluded.has(id) && Number.isFinite(scores[id]))
      .map(movieId=>({movieId,score:scores[movieId]}))
      .sort((a,b)=>b.score-a.score||a.movieId-b.movieId).slice(0,K);
  }

  /** Export complete float32 values as JSON numbers without rounding. */
  function exportModel() {
    if (!model) throw new Error('No model to export');
    return {label,format:'float32 values in JSON, raw-ID row-major matrices',
      userShape:model.userFactors.shape.slice(),movieShape:model.itemFactors.shape.slice(),
      userFactors:Array.from(model.userFactors.dataSync()),movieFactors:Array.from(model.itemFactors.dataSync()),
      globalMean:model.globalMean};
  }

  /** Restore allocated factors, validating shape/value consistency before replacing. */
  function restoreModel(saved) {
    for (const [key,shapeKey] of [['userFactors','userShape'],['movieFactors','movieShape']]) {
      const shape=saved[shapeKey],values=saved[key];
      if (!Array.isArray(shape)||shape.length!==2||!shape.every(x=>Number.isInteger(x)&&x>0)||
          !Array.isArray(values)||values.length!==shape[0]*shape[1]||!values.every(Number.isFinite)) {
        throw new Error('Invalid saved factors: '+key);
      }
    }
    if (saved.userShape[1]!==saved.movieShape[1]||!Number.isFinite(saved.globalMean))throw new Error('Invalid saved model dimensions/mean');
    disposeModel();
    model=tf.tidy(()=>({userFactors:tf.variable(tf.tensor2d(saved.userFactors,saved.userShape)),
      itemFactors:tf.variable(tf.tensor2d(saved.movieFactors,saved.movieShape)),globalMean:saved.globalMean}));
    return model;
  }

  function disposeModel() {
    if(model)tf.dispose([model.userFactors,model.itemFactors]);
    model=null;
  }

  return {label,config,initializeFactors,trainMF,selectBestCheckpoint,predictRating,computeRMSEFor,
    scoreAll,rankScores,exportModel,restoreModel,disposeModel,get history(){return trainingHistory.slice();}};
})();
