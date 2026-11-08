import os,json,random,re,time
from datetime import datetime,timezone
from decimal import Decimal
from retry_io import retry
os.environ['VLLM_CACHE_ROOT']='./cache/vllm'
os.environ['VLLM_CONFIG_ROOT']='./cache/vllm_config'
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
SEED=20261108
cfg=dict(seed=SEED,model='./output/v4',questions=1536,n=2,temperature=.8,top_p=.95,max_tokens=512,max_model_len=1024)
os.makedirs('logs/generation2',exist_ok=True);os.makedirs('cache/datasets',exist_ok=True)
def write(path,text,mode='a'):
 with open(path,mode) as f:f.write(text);f.flush()
retry(lambda:write('logs/generation2/settings.json',json.dumps(cfg,indent=2),'w'))
rows=[json.loads(l) for l in open('data/train.jsonl')];ids=list(range(len(rows)));random.Random(SEED).shuffle(ids);ids=ids[1152:1152+cfg['questions']]
tok=AutoTokenizer.from_pretrained('./models/base')
llm=LLM(model='./output/v4',dtype='bfloat16',max_model_len=1024,gpu_memory_utilization=.82,enforce_eager=True,seed=SEED,enable_prefix_caching=True,disable_log_stats=True)
params=SamplingParams(n=2,temperature=.8,top_p=.95,max_tokens=512,seed=SEED)
def number(s):
 matches=re.findall(r'####\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)',s)
 if not matches:return None
 try:return Decimal(matches[-1].replace(',',''))
 except:return None
kept=0;correct=0;total=0
for start in range(0,len(ids),64):
 batch=ids[start:start+64]
 prompts=[tok.apply_chat_template([{'role':'user','content':rows[i]['question']+'\nPlease reason step by step, and put your final answer after ####.'}],tokenize=False,add_generation_prompt=True) for i in batch]
 outputs=llm.generate(prompts,params,use_tqdm=False);records=[];accepted=[]
 for idx,out in zip(batch,outputs):
  gold=number(rows[idx]['answer']);solutions=[]
  for o in out.outputs:
   reward=int(number(o.text)==gold and gold is not None);total+=1;correct+=reward
   records.append(dict(timestamp=datetime.now(timezone.utc).isoformat(),source_id=idx,question=rows[idx]['question'],response=o.text,reward=reward,finish_reason=o.finish_reason,hyperparameters=cfg))
   if reward and o.finish_reason=='stop':solutions.append(o.text)
  for sol in list(dict.fromkeys(solutions))[:2]:
   accepted.append(dict(question=rows[idx]['question']+'\nPlease reason step by step, and put your final answer after ####.',answer=sol,source_id=idx));kept+=1
 retry(lambda:write('logs/generation2/rollouts.jsonl',''.join(json.dumps(r)+'\n' for r in records)))
 retry(lambda:write('cache/datasets/rft2.jsonl',''.join(json.dumps(r)+'\n' for r in accepted)))
 metric=dict(timestamp=datetime.now(timezone.utc).isoformat(),step=start//64+1,questions_done=start+len(batch),reward=correct/total,correct=correct,total=total,kept=kept,hyperparameters=cfg)
 retry(lambda:write('logs/generation2/metrics.jsonl',json.dumps(metric)+'\n'));print(json.dumps(metric),flush=True)
print('Generation complete',kept,flush=True)
llm.llm_engine.engine_core.shutdown()
