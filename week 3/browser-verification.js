// Real-browser Step 2 checks using Chrome DevTools Protocol and runtime built-ins.
// Run from this directory: bun browser-verification.js
// No external libraries. Temporary Chrome profile stays inside this directory
// and is removed on completion; the server binds only to localhost.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

const chromePath = process.env.CHROME_PATH || '/usr/bin/google-chrome';
const allowed = new Set(['index.html','style.css','data.js','script.js','u.data','u.item']);
const requests = [];
let failFile = null;
const server = Bun.serve({ hostname:'127.0.0.1',port:0,fetch(request) {
    const name = new URL(request.url).pathname.slice(1) || 'index.html';
    const status = allowed.has(name) && name !== failFile ? 200 : 404;
    requests.push({name,status});
    if(status!==200) return new Response('Not found',{status});
    return new Response(Bun.file(name), {headers:{'Cache-Control':'no-store'}});
}});
const profile = fs.mkdtempSync(path.join(process.cwd(),'.step2-chrome-'));
const chrome = Bun.spawn([chromePath,'--headless=new','--no-sandbox','--disable-gpu',
    '--disable-background-networking','--no-first-run','--no-default-browser-check',
    '--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],
    {stdout:'ignore',stderr:'pipe',env:{...process.env,XDG_CACHE_HOME:profile}});
let socket;
let counter=0;
const pending=new Map(),events=[];
function send(method,params={},sessionId) {
    return new Promise((resolve,reject)=>{
        const id=++counter;
        const timeout=setTimeout(()=>{pending.delete(id);reject(new Error(`CDP timeout: ${method}`));},60000);
        pending.set(id,{resolve:value=>{clearTimeout(timeout);resolve(value);},reject:error=>{clearTimeout(timeout);reject(error);}});
        socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));
    });
}
async function waitFor(action,message,limit=60000) {
    const start=Date.now();
    while(Date.now()-start<limit) {
        if(await action()) return;
        await Bun.sleep(50);
    }
    throw new Error(`Timed out: ${message}`);
}
async function evaluate(sessionId,expression) {
    const response=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);
    if(response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
    return response.result.value;
}
async function page() {
    const {targetId}=await send('Target.createTarget',{url:'about:blank'});
    const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
    await send('Runtime.enable',{},sessionId);
    await send('Log.enable',{},sessionId);
    await send('Page.enable',{},sessionId);
    await send('Emulation.setDeviceMetricsOverride',{width:1280,height:900,deviceScaleFactor:1,mobile:false},sessionId);
    await send('Page.navigate',{url:`http://127.0.0.1:${server.port}/`},sessionId);
    return {targetId,sessionId};
}

(async()=>{
    let stderr='';
    const endpoint=new Promise(async(resolve,reject)=>{
        const timeout=setTimeout(()=>reject(new Error('Chrome debugging endpoint unavailable')),15000);
        try {
            for await(const chunk of chrome.stderr) {
                stderr+=new TextDecoder().decode(chunk);
                const match=stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
                if(match){clearTimeout(timeout);resolve(match[1]);break;}
            }
        } catch(error){clearTimeout(timeout);reject(error);}
    });
    socket=new WebSocket(await endpoint);
    await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
    socket.onmessage=message=>{
        const packet=JSON.parse(message.data);
        if(packet.id) {
            const waiter=pending.get(packet.id);if(!waiter)return;
            pending.delete(packet.id);
            if(packet.error)waiter.reject(new Error(JSON.stringify(packet.error)));else waiter.resolve(packet.result);
        } else events.push(packet);
    };
    console.log('BROWSER_VERSION',JSON.stringify(await send('Browser.getVersion')));
    const main=await page();const s=main.sessionId;
    await waitFor(()=>evaluate(s,`document.getElementById('app-status')?.textContent.startsWith('Ready')`),'ready');
    const lectureLayout=await evaluate(s,`(() => {
        const user=document.querySelector('#user-select').getBoundingClientRect();
        const movie=document.querySelector('#movie-select').getBoundingClientRect();
        return {top5Collapsed:!document.querySelector('#top5-section').open,
            sideBySide:user.top===movie.top && user.right<movie.left,
            divider:getComputedStyle(document.querySelector('.prediction-column + .prediction-column')).borderLeftWidth,
            valueSize:getComputedStyle(document.querySelector('.prediction-value')).fontSize};
    })()`);
    assert.equal(lectureLayout.top5Collapsed,true);assert.equal(lectureLayout.sideBySide,true);
    assert.equal(lectureLayout.divider,'1px');assert.equal(lectureLayout.valueSize,'48px');
    console.log('BROWSER_LECTURE_LAYOUT',JSON.stringify(lectureLayout));
    await evaluate(s,`document.querySelector('#top5-section').open=true`);
    const initial=await evaluate(s,`({users:document.querySelector('#user-select').options.length,
        matrix:[numUsers,numMovies,ratings.length],selects:document.querySelectorAll('select').length,
        headings:[...document.querySelectorAll('.result-column h2')].map(e=>e.textContent),
        status:document.querySelector('#app-status').textContent,
        gridColumns:getComputedStyle(document.querySelector('#result-box')).gridTemplateColumns})`);
    assert.equal(initial.users,944);assert.equal(initial.selects,2);
    assert.equal(await evaluate(s,`document.querySelector('#movie-select').disabled &&
        document.querySelector('#predict-rating-btn').disabled`),true);
    assert.deepEqual(initial.matrix,[943,1682,100000]);
    assert.deepEqual(initial.headings,['User-Based CF','Item-Based CF']);
    assert.equal(initial.gridColumns.split(' ').length,2);
    assert.ok(requests.some(r=>r.name==='u.item'&&r.status===200));
    assert.ok(requests.some(r=>r.name==='u.data'&&r.status===200));
    console.log('BROWSER_INITIAL',JSON.stringify(initial));
    await evaluate(s,`document.querySelector('#recommend-btn').click()`);
    assert.equal(await evaluate(s,`document.querySelector('#user-based-result').textContent`),'Please select a user first.');
    assert.equal(await evaluate(s,`document.querySelector('#item-based-result').textContent`),'Please select a user first.');
    console.log('BROWSER_NO_SELECTION','PASS both cards');
    for(const userId of [1,33]) {
        const immediate=await evaluate(s,`document.querySelector('#user-select').value='${userId}';
            document.querySelector('#user-select').dispatchEvent(new Event('change',{bubbles:true}));
            document.querySelector('#recommend-btn').click();
            ({status:document.querySelector('#app-status').textContent,disabled:document.querySelector('#recommend-btn').disabled})`);
        assert.match(immediate.status,/Calculating/);assert.equal(immediate.disabled,true);
        await waitFor(()=>evaluate(s,`!isCalculating && document.querySelector('#app-status').textContent.startsWith('Ready — User ${userId} ')`),`user ${userId}`);
        const result=await evaluate(s,`({status:document.querySelector('#app-status').textContent,
            lists:['user-based-result','item-based-result'].map(id=>[...document.querySelectorAll('#'+id+' li')].map(e=>({
                movieId:Number(e.dataset.movieId),title:e.querySelector('.movie-title').textContent,
                score:e.querySelector('.predicted-rating').textContent,
                previouslyRated:ratingMatrix[${userId}][Number(e.dataset.movieId)]!==0}))),
            todoVisible:document.body.innerText.includes('TODO'),
            busy:document.querySelector('#result-box').getAttribute('aria-busy')})`);
        assert.equal(result.todoVisible,false);assert.equal(result.busy,'false');
        for(const list of result.lists) {
            assert.equal(list.length,5);
            assert.equal(new Set(list.map(r=>r.movieId)).size,5);
            assert.ok(list.every(r=>r.title&&!r.previouslyRated&&/^Predicted rating: -?\d+\.\d{3}$/.test(r.score)));
        }
        console.log('BROWSER_USER',JSON.stringify({userId,...result}));
        const selector=await evaluate(s,`({ids:[...document.querySelector('#movie-select').options]
            .filter(o=>o.value).map(o=>Number(o.value)),
            expected:[...moviesById.keys()].filter(id=>ratingMatrix[${userId}][id]===0).sort((a,b)=>a-b),
            cleared:document.querySelector('#movie-select').value==='',
            disabled:document.querySelector('#predict-rating-btn').disabled,
            output:document.querySelector('#user-based-prediction').textContent})`);
        assert.deepEqual(selector.ids,selector.expected);
        assert.equal(selector.ids.length,1682-(userId===1?272:24));
        assert.equal(selector.cleared,true);assert.equal(selector.disabled,true);
        assert.equal(selector.output,'—');
        const before=await evaluate(s,`document.querySelector('#result-box').innerHTML`);
        const target=userId===1?512:519;
        await evaluate(s,`document.querySelector('#movie-select').value='${target}';
            document.querySelector('#movie-select').dispatchEvent(new Event('change'));
            document.querySelector('#predict-rating-btn').click()`);
        await waitFor(()=>evaluate(s,'!isCalculating'),'movie prediction');
        const prediction=await evaluate(s,`({values:['user-based-prediction','item-based-prediction']
            .map(id=>document.getElementById(id).textContent),
            expected:[predictUserBasedRating(${userId},${target}),predictItemBasedRating(${userId},${target})]
                .map(score=>Number.isFinite(score)?score.toFixed(1):'N/A'),
            rated:ratingMatrix[${userId}][${target}],busy:document.querySelector('#movie-prediction').getAttribute('aria-busy')})`);
        assert.deepEqual(prediction.values,prediction.expected);
        assert.ok(prediction.values.every(value=>/^-?\d+\.\d$/.test(value)));
        if(userId===1) assert.deepEqual(prediction.values,['5.0','4.2']);
        assert.equal(prediction.rated,0);assert.equal(prediction.busy,'false');
        assert.equal(await evaluate(s,`document.querySelector('#result-box').innerHTML`),before);
        console.log('BROWSER_MOVIE',JSON.stringify({userId,movieId:target,unseenOptions:selector.ids.length,...prediction}));
    }
    // No unseen movies: temporary row-only UI fixture, restored immediately.
    assert.equal(await evaluate(s,`(() => {
        const row=ratingMatrix[33];ratingMatrix[33]=row.map(()=>5);
        populateMovieDropdown();
        const ok=document.querySelector('#movie-select').disabled &&
            document.querySelector('#predict-rating-btn').disabled &&
            document.querySelector('#prediction-status').textContent.includes('No unseen movies');
        ratingMatrix[33]=row;populateMovieDropdown();return ok;
    })()`),true);
    console.log('BROWSER_MOVIE_EMPTY','PASS no unseen movies');
    // Isolated temporary in-browser fixture checks the no-evidence UI state.
    await evaluate(s,`numUsers=944;buildRatingMatrix();
        const option=document.createElement('option');option.value='944';option.textContent='Synthetic new user';
        document.querySelector('#user-select').appendChild(option);document.querySelector('#user-select').value='944';
        document.querySelector('#user-select').dispatchEvent(new Event('change'));
        document.querySelector('#recommend-btn').click()`);
    await waitFor(()=>evaluate(s,'!isCalculating'),'synthetic user');
    const empty=await evaluate(s,`['user-based-result','item-based-result'].map(id=>document.getElementById(id).textContent)`);
    assert.ok(empty.every(text=>text.startsWith('No recommendation evidence')));
    console.log('BROWSER_NO_EVIDENCE',JSON.stringify(empty));
    await evaluate(s,`document.querySelector('#movie-select').value='512';
        document.querySelector('#movie-select').dispatchEvent(new Event('change'));
        document.querySelector('#predict-rating-btn').click()`);
    await waitFor(()=>evaluate(s,'!isCalculating'),'no movie evidence');
    assert.deepEqual(await evaluate(s,`['user-based-prediction','item-based-prediction'].map(id=>document.getElementById(id).textContent)`),
        ['N/A','N/A']);
    assert.deepEqual(await evaluate(s,`['user-based-prediction-note','item-based-prediction-note'].map(id=>document.getElementById(id).textContent)`),
        ['insufficient evidence','insufficient evidence']);
    console.log('BROWSER_MOVIE_NO_EVIDENCE','PASS both methods');
    await evaluate(s,`document.querySelector('#user-select').value='';populateMovieDropdown()`);
    assert.equal(await evaluate(s,`document.querySelector('#movie-select').disabled &&
        document.querySelector('#movie-select').options.length===1 && document.querySelector('#predict-rating-btn').disabled`),true);
    const successfulPageErrors=events.filter(e=>e.sessionId===s &&
        (e.method==='Runtime.exceptionThrown'||e.method==='Runtime.consoleAPICalled'&&e.params.type==='error'||
        e.method==='Log.entryAdded'&&e.params.entry.level==='error'));
    assert.deepEqual(successfulPageErrors,[]);
    console.log('BROWSER_NORMAL_CONSOLE_ERRORS',0);
    // Explicit failure path is separate from the successful application's console.
    failFile='u.data';const errorPage=await page();
    await waitFor(()=>evaluate(errorPage.sessionId,`document.querySelector('#app-status')?.textContent.startsWith('Error:')`),'load error');
    const failure=await evaluate(errorPage.sessionId,`({cards:['user-based-result','item-based-result'].map(id=>document.getElementById(id).textContent),
        disabled:document.querySelector('#recommend-btn').disabled})`);
    assert.ok(failure.cards.every(text=>text.includes('Failed to load rating data: 404')));
    assert.equal(failure.disabled,true);
    assert.equal(await evaluate(errorPage.sessionId,`document.querySelector('#movie-select').disabled &&
        document.querySelector('#predict-rating-btn').disabled`),true);
    console.log('BROWSER_INTENTIONAL_404',JSON.stringify(failure));
    console.log('BROWSER_REQUESTS',JSON.stringify(requests));
    console.log('PASS real Chrome: lecture layout, one-decimal predictions, secondary Top-5 workflow, unseen movie filtering, user changes, unchanged Top-5 cards, empty states, N/A with insufficient evidence, loading failure, console checks.');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
    if(socket?.readyState===WebSocket.OPEN) {
        try{await send('Browser.close');}catch{}
        socket.close();
    }
    chrome.kill();await chrome.exited;
    server.stop(true);
    for(const waiter of pending.values())waiter.reject(new Error('Browser session closed'));
    pending.clear();
    fs.rmSync(profile,{recursive:true,force:true});
});
