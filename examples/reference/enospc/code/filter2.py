import json,re,random
from decimal import Decimal
from datetime import datetime,timezone
from retry_io import retry
rows=[json.loads(l) for l in open('data/train.jsonl')]
seen=set();counts={};accepted=[];audits=[];correct=0;total=0
for line in open('logs/generation2/rollouts.jsonl'):
 try:r=json.loads(line)
 except ValueError:continue
 key=(r['source_id'],r['response'])
 if key in seen:continue
 seen.add(key);total+=1
 gold=rows[r['source_id']]['answer'].split('####')[-1].strip().replace(',','')
 nums=list(re.finditer(r'[-+]?\d[\d,]*(?:\.\d+)?',r['response']))
 pred=nums[-1].group().replace(',','') if nums else None
 try:reward=int(Decimal(pred)==Decimal(gold) and len(r['response'])-nums[-1].end()<120 and r['finish_reason']=='stop')
 except:reward=0
 correct+=reward
 audits.append(dict(timestamp=datetime.now(timezone.utc).isoformat(),source_id=r['source_id'],prediction=pred,gold=gold,reward=reward,original_format_reward=r['reward']))
 idx=r['source_id']
 if reward and counts.get(idx,0)<2:
  accepted.append(dict(question=r['question'],answer=re.sub(r'\n####[^\n]*\s*$','',r['response'])+'\n#### '+gold,source_id=idx));counts[idx]=counts.get(idx,0)+1
random.Random(20261108).shuffle(accepted)
def save(path,records):
 with open(path,'w') as f:
  for r in records:f.write(json.dumps(r)+'\n')
retry(lambda:save('cache/datasets/rft2_checked.jsonl',accepted))
retry(lambda:save('logs/generation2/reward_recheck.jsonl',audits))
metric=dict(timestamp=datetime.now(timezone.utc).isoformat(),seed=20261108,total=total,correct=correct,reward=correct/total,kept=len(accepted),source_questions=len(counts),filter='last numeric answer within final 120 chars; completed response; exact Decimal match')
retry(lambda:save('logs/generation2/recheck_summary.jsonl',[metric]));print(json.dumps(metric))
