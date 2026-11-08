import argparse,json,random,time,math,re,traceback,shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
from peft import LoraConfig,get_peft_model

def main():
 p=argparse.ArgumentParser();p.add_argument('--dpo',action='store_true');p.add_argument('--name',default='v4');p.add_argument('--steps',type=int,default=60);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--beta',type=float,default=.1);a=p.parse_args()
 cfg=dict(training_type='DPO',seed=20261108,split_seed=20261108,steps=a.steps,lr=a.lr,beta=a.beta,batch_size=2,gradient_accumulation=8,effective_pair_batch=16,max_length=768,lora_rank=32,lora_alpha=64,lora_dropout=0.0,weight_decay=.01,reference='frozen base model',data='logs/rewarded.jsonl')
 Path(f'logs/{a.name}_settings.json').write_text(json.dumps(cfg,indent=2));shutil.copyfile(__file__,f'logs/{a.name}_dpo.py');shutil.copyfile('code/train.py',f'logs/{a.name}_train.py')
 met=open(f'logs/{a.name}_metrics.jsonl','a',buffering=1)
 def log(**x):
  x.update(timestamp=datetime.now(timezone.utc).isoformat(),settings=cfg);met.write(json.dumps(x)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='settings'}),flush=True)
 def summary(ok,detail):
  with open('summary.md','a') as f:f.write(f'\n## Attempt {a.name} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: {"✅ 成功" if ok else "❌ 失败"}\n- **Score**: 尚未提交 | Improvement: N/A | Best: N/A\n- **训练类型**: DPO\n- **超参数**: `{json.dumps(cfg)}`\n- **做了什么**: 对同一道训练问题的正确与错误模型采样构建偏好对；双方均补齐####答案格式并排除截断样本。以冻结基础模型为参考，训练LoRA偏好损失并保存合并完整模型。{detail}\n- **为什么**: 首次SFT降低准确率，弱化增量有收益；DPO直接比较模型自身正确与错误推理，参考模型约束有助于保留原有能力，避免只拟合标准解答风格。\n- **问题/进步**: {detail}\n- **关键代码**: `loss = -logsigmoid(beta * ((chosen_logp-rejected_logp)-(ref_chosen_logp-ref_rejected_logp))).mean()`\n- **下一步建议**: GPU释放后提交，根据官方准确率选择最佳模型。\n')
 try:
  set_seed(20261108);torch.set_num_threads(8)
  tok=AutoTokenizer.from_pretrained('./models/base');tok.pad_token=tok.eos_token
  groups=defaultdict(list)
  for l in open('logs/rewarded.jsonl'):
   r=json.loads(l)
   if r['finish_reason']=='stop':groups[r['index']].append(r)
  def encode(r):
   answer=r['answer']
   if '####' not in answer:
    ns=re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',answer)
    if ns:answer+='\n#### '+ns[-1].replace(',','')
   prompt=tok.apply_chat_template([{'role':'user','content':r['question']+'\nPlease reason step by step, and put your final answer in the form #### <number>.'}],tokenize=False,add_generation_prompt=True)
   pi=tok.encode(prompt,add_special_tokens=False);ai=tok.encode(answer+tok.eos_token,add_special_tokens=False)
   return pi+ai,[-100]*len(pi)+ai
  pairs=[]
  for group in groups.values():
   good=[r for r in group if r['reward']];bad=[r for r in group if not r['reward']]
   if good and bad:
    c,r=encode(good[0]),encode(bad[0])
    if max(len(c[0]),len(r[0]))<=768:pairs.append((c,r))
  log(event='data',pairs=len(pairs))
  model=AutoModelForCausalLM.from_pretrained('./models/base',dtype=torch.bfloat16,attn_implementation='sdpa').cuda();model.config.use_cache=False
  model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
  def pack(ex):
   n=(max(len(x[0]) for x in ex)+7)//8*8
   return dict(input_ids=torch.tensor([x[0]+[tok.pad_token_id]*(n-len(x[0])) for x in ex],device='cuda'),attention_mask=torch.tensor([[1]*len(x[0])+[0]*(n-len(x[0])) for x in ex],device='cuda')),torch.tensor([x[1]+[-100]*(n-len(x[0])) for x in ex],device='cuda')
  def lp(inputs,labels):
   logits=model(**inputs).logits[:,:-1,:];target=labels[:,1:]
   token_losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),target.reshape(-1),ignore_index=-100,reduction='none')
   return -token_losses.reshape(target.shape).sum(-1)
  # Cache frozen-reference log probabilities once, for exactly the same tokenization.
  refs=[]; model.eval()
  with torch.no_grad(),model.disable_adapter():
   for i in range(0,len(pairs),2):
    ex=[item for pair in pairs[i:i+2] for item in pair];inputs,labels=pack(ex);v=lp(inputs,labels).reshape(-1,2).cpu().tolist();refs.extend(v)
  log(event='reference_cached',pairs=len(refs)); model.train()
  rng=random.Random(20261108);order=list(range(len(pairs)));rng.shuffle(order);cursor=0;start=time.time()
  for step in range(1,a.steps+1):
   opt.zero_grad(set_to_none=True);loss_sum=margin_sum=acc_sum=0
   lr=a.lr*min(step/6,1)*(0.2+0.8*.5*(1+math.cos(math.pi*step/a.steps)))
   for g in opt.param_groups:g['lr']=lr
   for micro in range(8):
    if cursor+2>len(order):rng.shuffle(order);cursor=0
    ix=order[cursor:cursor+2];cursor+=2
    inputs,labels=pack([item for i in ix for item in pairs[i]])
    policy=lp(inputs,labels).reshape(-1,2);ref=torch.tensor([refs[i] for i in ix],device='cuda')
    rewards=a.beta*(policy-ref);margin=rewards[:,0]-rewards[:,1]
    loss=-F.logsigmoid(margin).mean();(loss/8).backward()
    loss_sum+=loss.item()/8;margin_sum+=margin.mean().item()/8;acc_sum+=(margin>0).float().mean().item()/8
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   log(event='train',step=step,loss=loss_sum,reward_margin=margin_sum,preference_accuracy=acc_sum,grad_norm=float(norm),lr=lr,elapsed_s=time.time()-start)
  model.save_pretrained(f'output/{a.name}_adapter');tok.save_pretrained(f'output/{a.name}_adapter');model=model.merge_and_unload();model.config.use_cache=True;model.save_pretrained(f'output/{a.name}');tok.save_pretrained(f'output/{a.name}')
  path=Path(f'output/{a.name}/tokenizer_config.json');d=json.loads(path.read_text());d.pop('extra_special_tokens',None);path.write_text(json.dumps(d,indent=2))
  log(event='saved',step=a.steps);summary(True,f'完成{a.steps}步，{len(pairs)}个偏好对。')
 except Exception:
  summary(False,traceback.format_exc());raise
if __name__=='__main__':main()
