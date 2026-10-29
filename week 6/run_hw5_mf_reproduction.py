"""Separate, non-overwriting Step 4B runner. Run with python3 -B.

prepare: record protected hashes and verify HW5 source provenance.
start: launch an isolated headless Chrome and loopback HTTP/evidence server.
run: execute the browser experiment; progress is saved per epoch.
status: read job status without restarting training.
finalize: validate and export already completed evidence, without training.
"""
import argparse
import functools
import hashlib
import http.server
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
HW5=ROOT.parent/'Week 5'
OUT=ROOT/'evidence/hw5_mf_reproduction'
TEMP=Path('/tmp/opencode/hw5-mf-reproduction')
PORT=8767
DEBUG=9334
NEW_NAMES={'hw5_mf_reproduction.js','hw5_mf_reproduction_experiment.js',
 'run_hw5_mf_reproduction.py','export_hw5_mf_reproduction.py','README_hw5_mf_reproduction.md',
 'results_hw5_mf_reproduction.csv','recommendations_hw5_mf_reproduction.csv',
 'results_four_method_corrected_mf.csv'}
SESSION6=Path('/home/dastan/Downloads/hw6-step-1-starter-audit-and-baseline-verification.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    data=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    destination=OUT/name
    if destination.exists():
        if destination.read_bytes()!=data:
            raise FileExistsError('Refusing to overwrite '+str(destination))
    else:
        with destination.open('xb') as stream:stream.write(data)
    return {'path':str(destination.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def extract(source,name):
    match=re.search(r'^(?:async )?function '+name+r'\([^\n]*\) \{.*?^\}',source,re.M|re.S)
    if not match:
        # Local namespace functions are indented two spaces in the compatibility file.
        source='\n'.join(line[2:] if line.startswith('  ') else line for line in source.splitlines())
        match=re.search(r'^(?:async )?function '+name+r'\([^\n]*\) \{.*?^\}',source,re.M|re.S)
    if not match:raise ValueError('Cannot locate '+name)
    return match.group()


def prepare():
    if (OUT/'protected_before.json').exists():
        raise FileExistsError('This evidence directory is already prepared; do not reset or overwrite it.')
    protected={}
    for directory in [HW5,ROOT]:
        for path in directory.rglob('*'):
            if not path.is_file():continue
            if directory==ROOT and (OUT in path.parents or path.relative_to(ROOT).as_posix() in NEW_NAMES):continue
            protected[str(path)]=sha(path)
    if SESSION6.exists():protected[str(SESSION6)]=sha(SESSION6)
    save('protected_before.json',protected)
    original=(HW5/'script.js').read_text();compat=(ROOT/'hw5_mf_reproduction.js').read_text()
    assert sha(HW5/'script.js')=='1475668aec8c1b008b8b9c6c0b3302f16e9c7bca326e32ea5a0cd7a11e8d51e1'
    parity={}
    for name in ['isTfModel','clamp','predictRating','initializeFactors','computeRMSEFor','selectBestCheckpoint','trainMF']:
        a=extract(original,name);b=extract(compat,name)
        if name=='trainMF':
            b=b.replace('async function trainMF(train, val, opts, onEpoch = async () => {})','function trainMF(train, val, opts)').replace('await onEpoch(row);','')
        same=re.sub(r'\s+','',a)==re.sub(r'\s+','',b)
        assert same,'HW5 function divergence: '+name
        parity[name]={'matches_ignoring_indentation':same,'original_sha256':hashlib.sha256(a.encode()).hexdigest(),
          'adaptation':'async epoch observer only' if name=='trainMF' else 'none'}
    for name in ['data.js','u.data','u.item']:assert (HW5/name).read_bytes()==(ROOT/name).read_bytes()
    session=json.loads((HW5/'seikh_abaidullah_a05_session.json').read_text())
    message=next(m for m in session['messages'] if m['info']['id']=='msg_13252f69f001qBB6U9IqaJu5AS')
    output=next(p['state']['output'] for p in message['parts'] if p.get('type')=='tool' and 'EXTRA {' in p.get('state',{}).get('output',''))
    historical=json.JSONDecoder().raw_decode(output.split('EXTRA ',1)[1])[0]
    assert historical['selected']['epoch']==19 and historical['defaults']=={'epochs':20,'lr':.01,'reg':.05,'batchSize':64}
    save('provenance.json',{'label':'Corrected HW5 MF reproduction','function_parity':parity,
      'hw5_script_sha256':sha(HW5/'script.js'),'hw5_session_sha256':sha(HW5/'seikh_abaidullah_a05_session.json'),
      'hw6_session_sha256':sha(SESSION6),'historical_source_message':message['info']['id'],
      'historical_hw5':{k:historical[k] for k in ['defaults','selected','activeValidationRMSE','testRMSE','recall','factorNorms']},
      'dataset_hashes':{name:sha(ROOT/name) for name in ['u.data','u.item','data.js']},
      'original_weights_recovered':False,'host':platform.platform(),
      'compatibility_changes':['Namespace isolation','Async epoch progress observer; unchanged updates and batch order',
        'HW6 training-only candidates, split filtering and explicit movie-ID ties',
        'Serialization/restoration plus initialization archival'],
      'score_definition':'Original HW5 raw mean-plus-dot recommendation score; rating prediction and RMSE clipped to [1,5]',
      'initialization_policy':'Preserve unseeded tf.randomNormal; archive the actual initialization for replay; no claim of original HW5 weights'})
    print('Source parity, session configuration and identical dataset verified; protected baseline saved.',flush=True)


def targets():
    with urllib.request.urlopen(f'http://127.0.0.1:{DEBUG}/json',timeout=5) as response:return json.load(response)


def evaluate(expression):
    from websockets.sync.client import connect
    target=next(t for t in targets() if t['type']=='page' and f':{PORT}/' in t['url'])
    with connect(target['webSocketDebuggerUrl'],max_size=100_000_000) as ws:
        ws.send(json.dumps({'id':1,'method':'Runtime.evaluate','params':{'expression':expression,'returnByValue':True,'awaitPromise':True}}))
        while True:
            response=json.loads(ws.recv(timeout=1200))
            if response.get('id')!=1:continue
            if 'error' in response or 'exceptionDetails' in response.get('result',{}):raise RuntimeError(json.dumps(response))
            return response['result']['result'].get('value')


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store');super().end_headers()

    def do_POST(self):
        name=self.path.removeprefix('/__reproduction/')
        if not self.path.startswith('/__reproduction/') or not re.fullmatch(r'[A-Za-z0-9_.-]+\.json',name):
            self.send_error(404);return
        try:
            value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            result=save(name,value)
        except (ValueError,FileExistsError) as error:
            self.send_error(409,str(error));return
        body=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)


def serve():
    server=http.server.ThreadingHTTPServer(('127.0.0.1',PORT),functools.partial(Handler,directory=str(ROOT)))
    chrome=subprocess.Popen(['google-chrome','--headless=new','--no-first-run','--no-default-browser-check',
      '--disable-background-timer-throttling','--disable-renderer-backgrounding','--disable-backgrounding-occluded-windows',
      '--enable-unsafe-swiftshader',f'--user-data-dir={TEMP/"profile"}',f'--remote-debugging-port={DEBUG}',
      f'http://127.0.0.1:{PORT}/index.html'],stdout=subprocess.DEVNULL,stderr=(TEMP/'chrome.log').open('a'))
    try:server.serve_forever()
    finally:chrome.terminate();server.server_close()


def start():
    TEMP.mkdir(parents=True,exist_ok=True)
    try:
        if targets():print('Dedicated browser already running.');return
    except OSError:pass
    subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'serve'],cwd=ROOT,start_new_session=True,
      stdin=subprocess.DEVNULL,stdout=(TEMP/'server.log').open('a'),stderr=subprocess.STDOUT)
    for _ in range(100):
        time.sleep(.2)
        try:
            if targets():print('Dedicated Step 4B Chrome started.');return
        except OSError:pass
    raise RuntimeError('Chrome did not start')


def run():
    if (OUT/'initial_weights.json').exists():raise FileExistsError('A model run already exists; refusing retraining/overwrite')
    result=evaluate('''(async()=>{
      for(let i=0;i<120;i++){if(typeof HW6_STATE!=='undefined')break;await new Promise(r=>setTimeout(r,500));}
      let dependency='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',fallback=false;
      if(typeof tf==='undefined') {
        dependency='https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js';fallback=true;
        await new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=dependency;s.onload=resolve;s.onerror=reject;document.head.appendChild(s);});
        await hw6Bootstrap();
      }
      if(document.querySelector('script[src="https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js"]')) {
        dependency='https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js';fallback=true;
        if(!HW6_STATE.loaded&&ratings.length===0&&movies.length===0)await hw6Bootstrap();
      }
      for(let i=0;i<120&&!HW6_STATE.loaded;i++)await new Promise(r=>setTimeout(r,500));
      const response=await fetch(dependency),bytes=await response.arrayBuffer();
      window.HW5_REPRO_DEPENDENCY={url:dependency,fallbackMirrorUsed:fallback,version:tf.version.tfjs,
        sha256:Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('')};
      if(window.HW5_REPRO_JOB&&!HW5_REPRO_JOB.error)throw Error('A reproduction job already exists');
      if(window.HW5_REPRO_JOB?.error&&window.HW5_MF_REPRODUCTION)HW5_MF_REPRODUCTION.disposeModel();
      await import('./hw5_mf_reproduction.js?v='+Date.now());await import('./hw5_mf_reproduction_experiment.js?v='+Date.now());
      window.HW5_REPRO_JOB={stage:'starting',done:false,error:null};
      setTimeout(()=>HW5_MF_REPRODUCTION_EXPERIMENT.run().then(result=>{HW5_REPRO_JOB.done=true;HW5_REPRO_JOB.result=result;},
        async error=>{HW5_REPRO_JOB.done=true;HW5_REPRO_JOB.error=String(error.stack||error);
          await fetch('/__reproduction/failure_'+Date.now()+'.json',{method:'POST',body:JSON.stringify(HW5_REPRO_JOB)});}),0);
      return HW5_REPRO_JOB;
    })()''')
    print(json.dumps(result),flush=True)
    last=None
    while True:
        time.sleep(10);status=evaluate('HW5_REPRO_JOB')
        if status!=last:print(json.dumps(status,ensure_ascii=False),flush=True);last=status
        if status['done']:
            if status['error']:raise RuntimeError(status['error'])
            from export_hw5_mf_reproduction import finalize
            print(json.dumps(finalize(ROOT),indent=2),flush=True);break


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','start','serve','run','status','finalize'])
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='start':start()
    elif args.action=='serve':serve()
    elif args.action=='run':run()
    elif args.action=='status':print(json.dumps(evaluate('HW5_REPRO_JOB'),indent=2))
    else:
        from export_hw5_mf_reproduction import finalize
        print(json.dumps(finalize(ROOT),indent=2))
