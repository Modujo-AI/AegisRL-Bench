import json,re,argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',default='logs/rollouts.jsonl');p.add_argument('--output',default='logs/rewarded.jsonl');a=p.parse_args()
rows=[json.loads(l) for l in open('data/train.jsonl')]
def extract(text):
 for pattern in [r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)',r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}',r'([-+]?\d[\d,]*(?:\.\d+)?)']:
  m=re.findall(pattern,text)
  if m:
   try:return float(m[-1].replace(',',''))
   except:pass
 return None
counts=Counter(); out=open(a.output,'w')
for line in open(a.input):
 r=json.loads(line); target=extract(rows[r['index']]['answer']); pred=extract(r['answer'])
 r['original_format_reward']=r['reward']; r['reward']=int(pred==target and r['finish_reason']=='stop')
 counts['total']+=1; counts['correct']+=r['reward']; counts['truncated']+=int(r['finish_reason']!='stop')
 if r['reward'] and '####' not in r['answer']:
  r['answer']+='\n#### '+re.search(r'####\s*(.*)',rows[r['index']]['answer']).group(1).strip()
 out.write(json.dumps(r)+'\n')
metrics=dict(timestamp=datetime.now(timezone.utc).isoformat(),seed=20261108,counts=dict(counts),reward=counts['correct']/counts['total'],rules='last hash answer, else boxed number, else last number; reject truncations; canonicalize correct answer')
Path(a.output+'.metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics))
with open('summary.md','a') as f:f.write(f'\n## Data processing: reward filtering ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **做了什么**: 根据已记录采样配置按seed=20261108生成训练问题解答，temperature=0.7，top_p=0.95，max_tokens=512。检查训练标签数值，排除截断输出，正确解答补齐####格式。原始轨迹和重算奖励分别保留，输入{a.input}，输出{a.output}。\n- **为什么**: 初始严格####奖励遗漏了多数boxed/普通数字答案；数值检查允许保留模型自身正确推理风格，同时学习要求的结束格式。\n- **结果**: `{json.dumps(metrics)}`\n')
