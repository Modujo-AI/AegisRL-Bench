import json,random,re,time,argparse
from pathlib import Path
from datetime import datetime,timezone
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
p=argparse.ArgumentParser();p.add_argument('--model',default='./models/base');p.add_argument('--offset',type=int,default=0);p.add_argument('--count',type=int,default=2048);p.add_argument('--n',type=int,default=2);p.add_argument('--name',default='rollout');a=p.parse_args()
seed=20261108
settings=dict(seed=seed,model=a.model,temperature=0.7,top_p=.95,n=a.n,max_tokens=512,prompts=a.count,offset=a.offset)
Path(f'logs/{a.name}_settings.json').write_text(json.dumps(settings,indent=2))
rows=[json.loads(l) for l in open('./data/train.jsonl')]
ids=json.loads(Path('logs/split.json').read_text())['train'][a.offset:a.offset+a.count]
tok=AutoTokenizer.from_pretrained('./models/base')
llm=LLM(model=a.model,dtype='bfloat16',gpu_memory_utilization=.75,max_model_len=1024,max_num_seqs=128,seed=seed,enable_prefix_caching=True)
sampling=SamplingParams(n=a.n,temperature=.7,top_p=.95,max_tokens=512,seed=seed)
def answer(t):
 m=re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',t)
 if not m: return None
 try: return float(m[-1].replace(',',''))
 except: return None
out=open(f'logs/{a.name}s.jsonl','w',buffering=1); met=open(f'logs/{a.name}_metrics.jsonl','w',buffering=1)
start=time.time(); good=0; total=0
for k in range(0,len(ids),128):
 subset=ids[k:k+128]
 prompts=[tok.apply_chat_template([{'role':'user','content':rows[i]['question']+'\nPlease reason step by step, and put your final answer in the form #### <number>.'}],tokenize=False,add_generation_prompt=True) for i in subset]
 results=llm.generate(prompts,sampling,use_tqdm=False)
 for i,result in zip(subset,results):
  for o in result.outputs:
   reward=int(answer(o.text)==answer(rows[i]['answer']) and answer(o.text) is not None)
   record=dict(index=i,question=rows[i]['question'],answer=o.text,reward=reward,finish_reason=o.finish_reason,tokens=len(o.token_ids))
   out.write(json.dumps(record)+'\n'); total+=1; good+=reward
 metric=dict(timestamp=datetime.now(timezone.utc).isoformat(),step=k//128+1,generated=total,reward=good/total,elapsed_s=time.time()-start,settings=settings)
 met.write(json.dumps(metric)+'\n'); print(json.dumps(metric),flush=True)

out.close();met.close()
llm.llm_engine.engine_core.shutdown()
