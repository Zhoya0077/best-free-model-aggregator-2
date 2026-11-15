import json, os, socket, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KIT = Path('/home/hatch/workspace/makefun/free-model-aggregator-02/kit/free-model-aggregator-02-kit')
TRIAL = KIT / 'practice/trial-0/input'
PORTS = {}
registered = {}
gateway_calls = []
prompt_count = 6

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass
    def reply(self, status, obj, headers=None):
        raw = json.dumps(obj, separators=(',', ':')).encode()
        self.send_response(status); self.send_header('content-type','application/json'); self.send_header('content-length',str(len(raw)))
        for k,v in (headers or {}).items(): self.send_header(k,v)
        self.end_headers(); self.wfile.write(raw)
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers.get('content-length','0'))))
        if self.path == '/register':
            registered.update(body); return self.reply(200, {'ok':True,'requests':prompt_count})
        if self.path in ('/chat/completions','/v1/chat/completions'):
            gateway_calls.append(body)
            model=body['model']
            if len(gateway_calls)==1:
                return self.reply(429, {'error':{'code':'rate_limit','type':'rate_limit_error','message':'programmed refusal'}})
            # The mock completion is a deterministic echo of the unique token contained in the user prompt.
            text=' '.join(str(m.get('content','')) for m in body.get('messages',[]))
            import re
            token=re.search(r'([A-Z]{4,}-[0-9]{4})', text)
            answer=token.group(1) if token else 'token-unavailable'
            if body.get('response_format',{}).get('type')=='json_object': answer=json.dumps({'token':answer},separators=(',',':'))
            if body.get('stream') is True:
                frames=[{'id':f'mock-{len(gateway_calls)}','model':model.removeprefix('openrouter/'),'choices':[{'index':0,'delta':{'role':'assistant','content':answer},'finish_reason':None}]},{'id':f'mock-{len(gateway_calls)}','model':model.removeprefix('openrouter/'),'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]},{'id':f'mock-{len(gateway_calls)}','model':model.removeprefix('openrouter/'),'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}}]
                raw=''.join('data: '+json.dumps(f,separators=(',',':'))+'\n\n' for f in frames)+'data: [DONE]\n\n'
                data=raw.encode(); self.send_response(200); self.send_header('content-type','text/event-stream'); self.send_header('content-length',str(len(data))); self.end_headers(); self.wfile.write(data)
            else:
                self.reply(200, {'id':f'mock-{len(gateway_calls)}','object':'chat.completion','model':model.removeprefix('openrouter/'),'choices':[{'index':0,'message':{'role':'assistant','content':answer},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}})
    def do_GET(self):
        if self.path=='/work': return self.reply(200, {'done':len(client_results)>=prompt_count,'pending':max(0,prompt_count-len(client_results))})
        self.reply(404,{})

client_results=[]
mock=ThreadingHTTPServer(('127.0.0.1',0),Handler); threading.Thread(target=mock.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{mock.server_address[1]}'

env=os.environ.copy(); env.update({'MAKEFUN_ENV_URL':base,'MAKEFUN_PROMPT_FILE':str(TRIAL/'prompt.txt'),'OPENAI_BASE_URL':base,'OPENAI_API_KEY':'makefun-trial-mock'})
proc=subprocess.Popen([sys.executable,str(ROOT/'main.py')],cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
try:
    deadline=time.time()+10
    while not registered and time.time()<deadline: time.sleep(.02)
    assert registered, 'server failed to register'
    port=registered['port']; assert port != 8080
    client=f'http://127.0.0.1:{port}'
    # Both authentication headers select the corresponding public model-list wire.
    with urllib.request.urlopen(urllib.request.Request(client+'/v1/models',headers={'authorization':'Bearer client'})) as response:
        openai_list=json.loads(response.read())
    with urllib.request.urlopen(urllib.request.Request(client+'/v1/models',headers={'x-api-key':'client'})) as response:
        anthropic_list=json.loads(response.read())
    catalog_ids=[m['id'] for m in registered['catalog']['providers'][0]['models']]
    assert [m['id'] for m in openai_list['data']]==catalog_ids
    assert [m['id'] for m in anthropic_list['data']]==catalog_ids
    assert all(m['object']=='model' for m in openai_list['data']) and anthropic_list['has_more'] is False
    workload=json.loads((TRIAL/'env/workload.json').read_text())
    for row in workload['manifest']:
        body=workload['bodies'][row['id']]
        path='/v1/chat/completions' if row['api']=='openai' else '/v1/messages'
        req=urllib.request.Request(client+path,data=json.dumps(body).encode(),headers={'content-type':'application/json','authorization':'Bearer client-test','x-api-key':'client-test'} if row['api']=='anthropic' else {'content-type':'application/json','authorization':'Bearer client-test'},method='POST')
        try:
            response=urllib.request.urlopen(req,timeout=8)
        except urllib.error.HTTPError as e:
            print('ERROR',row['id'],e.code,e.read()); print('SERVER STDERR',proc.stderr.read()); raise
        with response as resp:
            payload=resp.read().decode()
            if row['stream']:
                assert resp.headers.get('content-type','').startswith('text/event-stream')
            expected=row['token']; assert expected in payload,(row['id'],payload)
            if row['stream'] and row['api']=='openai':
                chunks=[json.loads(line[6:]) for line in payload.splitlines() if line.startswith('data: {')]
                assert chunks and all(c.get('model') in catalog_ids for c in chunks)
                assert payload.rstrip().endswith('data: [DONE]')
            if row['stream'] and row['api']=='anthropic':
                events=[line[7:] for line in payload.splitlines() if line.startswith('event: ')]
                assert events[0]=='message_start' and events[-2:]==['message_delta','message_stop'],events
                assert events.index('content_block_start')<events.index('content_block_delta')<events.index('content_block_stop')
                start_data=json.loads(payload.split('event: message_start\n')[1].split('\n\n')[0].removeprefix('data: '))
                assert start_data['message']['model'] in catalog_ids
            if row['id']=='p01': assert resp.headers.get('x-makefun-upstream-failures'), 'missing refusal trail'
            client_results.append((row['id'],'PASS',resp.status,resp.headers.get('x-makefun-upstream-failures','')))
    deadline=time.time()+4
    while proc.poll() is None and time.time()<deadline: time.sleep(.02)
    assert proc.poll()==0, f'server did not exit on done: {proc.poll()}'
    assert len(gateway_calls)==7, len(gateway_calls)
    assert registered['catalog']['providers'][0]['models']
    print('registration: PASS; port:',port,'catalog models:',len(registered['catalog']['providers'][0]['models']))
    for item in client_results: print(*item,sep=' | ')
    print('gateway calls:',len(gateway_calls),'(one 429 followed by successful failover)')
finally:
    if proc.poll() is None: proc.terminate(); proc.wait(timeout=2)
    mock.shutdown(); mock.server_close()
