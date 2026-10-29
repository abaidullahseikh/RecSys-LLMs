"""Local Chrome/CDP runner for experiment.js (Python 3 + websockets).

python3 run_experiment.py start
python3 run_experiment.py eval 'HW6_EXPERIMENT.setup()' --out evidence/setup.json
Only a separate, temporary Chrome profile is controlled. Stage outputs are
saved in the project; browser/CDP settings and logs remain under /tmp/opencode.
"""
import argparse
import functools
import http.server
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
TEMP = Path('/tmp/opencode')
PORT = 8766
DEBUG = 9333


def targets():
    with urllib.request.urlopen(f'http://127.0.0.1:{DEBUG}/json', timeout=5) as response:
        return json.load(response)


def evaluate(expression):
    from websockets.sync.client import connect
    page = next(t for t in targets() if t['type'] == 'page' and f':{PORT}/' in t['url'])
    with connect(page['webSocketDebuggerUrl'], max_size=100_000_000) as ws:
        ws.send(json.dumps({'id': 1, 'method': 'Runtime.evaluate', 'params': {
            'expression': expression, 'awaitPromise': True, 'returnByValue': True}}))
        while True:
            response = json.loads(ws.recv(timeout=1200))
            if response.get('id') != 1:
                continue
            result = response.get('result', {})
            if 'error' in response or 'exceptionDetails' in result:
                raise RuntimeError(json.dumps(response, indent=2))
            return result.get('result', {}).get('value')


def reset():
    from websockets.sync.client import connect
    page = next(t for t in targets() if t['type']=='page' and f':{PORT}/' in t['url'])
    with connect(page['webSocketDebuggerUrl'], max_size=100_000_000) as ws:
        ws.send(json.dumps({'id':1,'method':'Page.reload','params':{'ignoreCache':True}}))
        while json.loads(ws.recv(timeout=30)).get('id')!=1:
            pass
    print('Reloaded the isolated experiment page. Wait for data loading, then run setup.')


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

def serve():
    server = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), functools.partial(Handler, directory=str(ROOT)))
    chrome = subprocess.Popen([
        'google-chrome', '--headless=new', '--no-first-run', '--no-default-browser-check',
        '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows', '--enable-unsafe-swiftshader',
        f'--user-data-dir={TEMP / "hw6-chrome-profile"}', f'--remote-debugging-port={DEBUG}',
        f'http://127.0.0.1:{PORT}/index.html'], stdout=subprocess.DEVNULL,
        stderr=(TEMP / 'chrome.log').open('a'))
    try:
        server.serve_forever()
    finally:
        chrome.terminate()
        server.server_close()


def save(name, value):
    path = ROOT / 'evidence' / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def pipeline():
    """Run after setup; persist each completed stage, polling training by epoch."""
    evaluate('HW6_EXPERIMENT.backend("wasm")')
    for alpha, label in [(0.5, '050'), (0.7, '070'), (0.85, '085'), (0.95, '095')]:
        audit = evaluate(f'HW6_EXPERIMENT.validationAlpha({alpha})')
        save(f'validation-{label}.json', audit)
        save('experiment-progress.json', evaluate('HW6_EXPERIMENT.evidence'))
        print(json.dumps({'stage': 'validation', 'alpha': alpha, 'recall': audit['summary']['recall'],
                          'nonconverged': audit['summary']['nonconverged']}), flush=True)
    selection = evaluate('HW6_EXPERIMENT.selectAlpha()')
    save('selection.json', selection)
    print(json.dumps({'stage': 'selection', **selection}), flush=True)
    evaluate('''(()=>{
      window.MF_JOB={done:false,error:null};
      setTimeout(()=>{
        HW6_EXPERIMENT.trainMf('webgl').then(result=>{MF_JOB.done=true;MF_JOB.result=result;},
          error=>{MF_JOB.done=true;MF_JOB.error=String(error.stack||error);});
      },0);
      return {started:true,backend:'webgl'};
    })()''')
    last_epoch = -1
    while True:
        time.sleep(10)
        progress = evaluate('({job:MF_JOB, info:MF_INFO,status:document.getElementById("status").textContent})')
        save('mf-progress.json', progress)
        epoch = len(progress['info']['history'])
        if epoch != last_epoch:
            print(json.dumps({'stage': 'mf', 'epoch': epoch, 'status': progress['status']}), flush=True)
            last_epoch = epoch
        if progress['job']['done']:
            if progress['job']['error']:
                raise RuntimeError(progress['job']['error'])
            save('mf.json', progress['job']['result'])
            save('mf-checkpoint.json', evaluate('({info:MF_INFO,userShape:HW6_MF_MODEL.userFactors.shape,movieShape:HW6_MF_MODEL.itemFactors.shape,userFactors:Array.from(HW6_MF_MODEL.userFactors.dataSync()),movieFactors:Array.from(HW6_MF_MODEL.itemFactors.dataSync()),globalMean:HW6_MF_MODEL.globalMean})'))
            save('experiment-progress.json', evaluate('HW6_EXPERIMENT.evidence'))
            break
    result = evaluate('HW6_EXPERIMENT.testComparison()')
    save('test.json', result)
    print(json.dumps({'stage': 'test', 'rows': result['rows']}), flush=True)
    users = evaluate('HW6_EXPERIMENT.collectUsers()')
    save('users.json', users)
    bundle = evaluate('HW6_EXPERIMENT.bundle()')
    save('bundle.json', bundle)
    from export_evidence import export_and_validate
    print(json.dumps(export_and_validate(ROOT), indent=2), flush=True)


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'serve', 'reset', 'eval', 'pipeline'])
    parser.add_argument('expression', nargs='?')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.action == 'serve':
        serve()
    elif args.action == 'reset':
        reset()
    elif args.action == 'start':
        try:
            print(json.dumps(targets(), indent=2))
            return
        except OSError:
            pass
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'serve'],
                         cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=(TEMP / 'server.log').open('a'), stderr=subprocess.STDOUT)
        for _ in range(100):
            time.sleep(.2)
            try:
                if targets():
                    print('Started isolated Chrome; application: http://127.0.0.1:8766/index.html')
                    return
            except OSError:
                pass
        raise RuntimeError('Chrome did not start; inspect /tmp/opencode/server.log')
    elif args.action == 'pipeline':
        pipeline()
    else:
        value = evaluate(args.expression)
        if args.out:
            output = Path(args.out)
            if not output.is_absolute():
                output = ROOT / output
            output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(value.get('summary', value) if isinstance(value, dict) else value,
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
