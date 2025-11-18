// main.js - Dots Classifier (client-side) using TensorFlow.js

const qs = (id) => document.getElementById(id);

// DOM elements
const startBtn = qs('startBtn');
const pauseBtn = qs('pauseBtn');
const resetBtn = qs('resetBtn');
const statusEl = qs('status');
const previewCanvas = qs('previewCanvas');
const previewCtx = previewCanvas.getContext('2d');

// chart setup
const accCtx = qs('accChart').getContext('2d');
const chart = new Chart(accCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {label: 'Train Acc (batch)', borderColor: 'blue', data: [], tension:0.2},
      {label: 'Val Acc (batch)', borderColor: 'orange', data: [], tension:0.2}
    ]
  },
  options: {animation:false, scales:{y:{min:0,max:1}}}
});

// RNG with seed
function seededRNG(seed){
  let s = seed >>> 0;
  return function(){
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  }
}

function getOpts(){
  return {
    overlap: qs('overlap').value,
    dotMin: Number(qs('dotMin').value),
    dotMax: Number(qs('dotMax').value),
    batchSize: Number(qs('batchSize').value),
    epochs: Number(qs('epochs').value),
    imageSize: Number(qs('imageSize').value),
    radiusMin: Number(qs('radiusMin').value),
    radiusMax: Number(qs('radiusMax').value),
    seed: Number(qs('seed').value) || 1,
    stepsPerEpoch: Number(qs('stepsPerEpoch').value),
    valSteps: Number(qs('valSteps').value),
    updateEvery: Number(qs('updateEvery').value),
  };
}

function makeBlankImage(size){
  const arr = new Uint8ClampedArray(size*size);
  for(let i=0;i<arr.length;i++) arr[i]=0;
  return arr;
}

function drawCircleOnArray(arr, size, cx, cy, r){
  const r2 = r*r;
  for(let y=Math.max(0,Math.floor(cy-r)); y<=Math.min(size-1, Math.ceil(cy+r)); y++){
    for(let x=Math.max(0,Math.floor(cx-r)); x<=Math.min(size-1, Math.ceil(cx+r)); x++){
      const dx = x-cx, dy=y-cy;
      if(dx*dx+dy*dy <= r2) arr[y*size+x]=255;
    }
  }
}

function generateSingleImage(opts, rng){
  const size = opts.imageSize;
  const img = makeBlankImage(size);
  const count = Math.floor(rng()*(opts.dotMax - opts.dotMin + 1)) + opts.dotMin;
  const dots = [];
  for(let i=0;i<count;i++){
    let r = opts.radiusMin + rng()*(opts.radiusMax-opts.radiusMin);
    if(r<1) r=1;
    // pick random center, ensure fully inside
    let cx = r + rng()*(size - 2*r);
    let cy = r + rng()*(size - 2*r);
    if(opts.overlap === 'prevent'){
      // try to find non-overlapping placement
      let tries=0;
      while(tries<50){
        let ok=true;
        for(const d of dots){
          const dx=d.cx-cx, dy=d.cy-cy; const minDist = d.r + r + 1;
          if(dx*dx+dy*dy < minDist*minDist){ ok=false; break; }
        }
        if(ok) break;
        cx = r + rng()*(size - 2*r);
        cy = r + rng()*(size - 2*r);
        tries++;
      }
    }
    dots.push({cx,cy,r});
    drawCircleOnArray(img, size, cx, cy, r);
  }
  return {img, label: count};
}

function generateBatch(opts, rng){
  const images = [];
  const labels = [];
  for(let b=0;b<opts.batchSize;b++){
    const {img, label} = generateSingleImage(opts, rng);
    images.push(img);
    labels.push(label);
  }
  // convert to Float32Array normalized [0,1]
  const xs = new Float32Array(opts.batchSize*opts.imageSize*opts.imageSize);
  for(let i=0;i<images.length;i++){
    const img = images[i];
    for(let j=0;j<img.length;j++) xs[i*img.length + j] = img[j]/255.0;
  }
  // labels: classes 1..9 -> indices 0..8
  const numClasses = 9;
  const ys = new Float32Array(opts.batchSize * numClasses);
  for(let i=0;i<labels.length;i++){
    const c = Math.min(numClasses-1, Math.max(1, labels[i])) - 1;
    ys[i*numClasses + c] = 1;
  }
  return {xs, ys, labels};
}

function drawPreviewFromArray(raw, size, canvas){
  const ctx = canvas.getContext('2d');
  const scale = canvas.width / size;
  const imageData = ctx.createImageData(size, size);
  for(let i=0;i<raw.length;i++){
    const v = Math.round(raw[i]*255);
    imageData.data[i*4+0]=v; imageData.data[i*4+1]=v; imageData.data[i*4+2]=v; imageData.data[i*4+3]=255;
  }
  // draw small canvas then scale up to keep pixels crisp
  const tmp = document.createElement('canvas'); tmp.width=size; tmp.height=size; tmp.getContext('2d').putImageData(imageData,0,0);
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#222'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

function createModel(imageSize){
  const model = tf.sequential();
  model.add(tf.layers.conv2d({inputShape:[imageSize,imageSize,1], filters:16, kernelSize:3, activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize:2}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units:64, activation:'relu'}));
  model.add(tf.layers.dense({units:9, activation:'softmax'}));
  return model;
}

let stopRequested = false;

async function trainLoop(){
  const opts = getOpts();
  stopRequested = false;
  statusEl.textContent = 'Preparing model...';
  tf.engine().startScope();
  const model = createModel(opts.imageSize);
  const optimizer = tf.train.adam(0.001);
  model.compile({optimizer, loss:'categoricalCrossentropy', metrics:['accuracy']});

  const rngBase = seededRNG(opts.seed);

  const totalBatches = opts.epochs * opts.stepsPerEpoch;
  let globalBatch = 0;

  // training loop per epoch and per step
  for(let e=0;e<opts.epochs; e++){
    if(stopRequested) break;
    for(let s=0; s<opts.stepsPerEpoch; s++){
      if(stopRequested) break;
      // generate training batch
      const rng = rngBase; // deterministic stream
      const {xs, ys, labels} = generateBatch(opts, rng);
      const xsTensor = tf.tensor4d(xs, [opts.batchSize, opts.imageSize, opts.imageSize, 1]);
      const ysTensor = tf.tensor2d(ys, [opts.batchSize, 9]);
      const history = await model.trainOnBatch(xsTensor, ysTensor);
      // history = [loss, acc]
      const batchAcc = Array.isArray(history) ? history[1] : 0;

      // validation step (single batch) optionally
      let valAcc = null;
      if(opts.valSteps>0){
        // run a small validation batch
        const {xs: vxs, ys: vys} = generateBatch({...opts, batchSize: Math.min(opts.batchSize, 32)}, rng);
        const vxt = tf.tensor4d(vxs, [vxs.length/(opts.imageSize*opts.imageSize), opts.imageSize, opts.imageSize, 1]);
        const vyt = tf.tensor2d(vys, [vys.length/9, 9]);
        const evalRes = await model.evaluate(vxt, vyt, {batchSize: Math.min(opts.batchSize, 32)});
        const accTensor = Array.isArray(evalRes) ? evalRes[1] : evalRes;
        valAcc = (await accTensor.data())[0];
        vxt.dispose(); vyt.dispose();
      }

      // update preview (first image of last generated batch)
      const firstRaw = xs.slice(0, opts.imageSize*opts.imageSize);
      drawPreviewFromArray(firstRaw, opts.imageSize, previewCanvas);

      // update batch stats section
      qs('backend').textContent = tf.getBackend();
      qs('progressLabel').textContent = `Preview label: ${labels && labels.length ? labels[0] : '-'}`;
      qs('batchLoss').textContent = Array.isArray(history) ? history[0].toFixed(4) : '-';
      qs('batchAcc').textContent = Array.isArray(history) ? history[1].toFixed(4) : '-';

      // prediction (show predicted class + confidence for first sample)
      try{
        await tf.nextFrame();
        const pred = await tf.tidy(()=>{
          const input = tf.tensor4d(firstRaw, [1, opts.imageSize, opts.imageSize, 1]);
          const logits = model.predict(input);
          return logits.array();
        });
        if(pred && pred[0]){
          const probs = pred[0];
          let maxIdx = 0; let maxP = probs[0];
          for(let i=1;i<probs.length;i++) if(probs[i]>maxP){ maxP=probs[i]; maxIdx=i; }
          const predictedClass = maxIdx + 1;
          const confidence = (maxP*100).toFixed(1);
          const trueLabel = labels && labels.length? labels[0] : 'N/A';
          qs('previewInfo').textContent = `Predicted: ${predictedClass} (${confidence}%) — True: ${trueLabel}`;
        }
      }catch(err){
        // ignore prediction errors
      }

      // push to chart arrays
      chart.data.labels.push(globalBatch.toString());
      chart.data.datasets[0].data.push(batchAcc);
      chart.data.datasets[1].data.push(valAcc===null?null:valAcc);
      if(globalBatch % opts.updateEvery === 0) chart.update('none');

      globalBatch++;

      xsTensor.dispose(); ysTensor.dispose();
      await tf.nextFrame();
    }
  }

  statusEl.textContent = stopRequested ? 'Paused' : 'Done';
  tf.engine().endScope();
}

startBtn.addEventListener('click', ()=>{
  chart.data.labels = []; chart.data.datasets[0].data=[]; chart.data.datasets[1].data=[]; chart.update();
  trainLoop().catch(err=>{console.error(err); statusEl.textContent='Error: '+err.message});
});

pauseBtn.addEventListener('click', ()=>{ stopRequested = true; statusEl.textContent='Pause requested'; });

resetBtn.addEventListener('click', ()=>{ location.reload(); });
