# DPO mode: default lr=2e-5, steps=60, beta=0.1, batch=2, accum=8, seed=20261108.
import sys
if '--dpo' in sys.argv:
 from dpo import main
 main()
 sys.exit(0)
import os, json, random, re, time, argparse, math, traceback, shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, PeftModel

p=argparse.ArgumentParser()
p.add_argument('--name', default='v1'); p.add_argument('--steps',type=int,default=225)
p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--resume',default=None)
p.add_argument('--data',default=None); p.add_argument('--max-length',type=int,default=512)
p.add_argument('--batch',type=int,default=8); p.add_argument('--accum',type=int,default=4)
a=p.parse_args()
settings=dict(training_type='RFT' if a.data else 'SFT',seed=20261108,split_seed=20261108,lr=a.lr,steps=a.steps,batch_size=a.batch,gradient_accumulation=a.accum,effective_batch=a.batch*a.accum,max_length=a.max_length,lora_rank=32,lora_alpha=64,lora_dropout=0.05,weight_decay=0.01,warmup_steps=15,resume=a.resume,data=a.data,answer_only_loss=True,remove_calculator_annotations=True)
Path('logs').mkdir(exist_ok=True)
shutil.copyfile(__file__,f'logs/{a.name}_train.py')
json.dump(settings,open(f'logs/{a.name}_settings.json','w'),indent=2)
metrics=open(f'logs/{a.name}_metrics.jsonl','a',buffering=1)
def log(**x):
 x.update(timestamp=datetime.now(timezone.utc).isoformat(),settings=settings); metrics.write(json.dumps(x)+'\n'); print(json.dumps({k:v for k,v in x.items() if k!='settings'}),flush=True)
def summary(ok,detail):
 with open('summary.md','a') as f:
  f.write(f'\n## Attempt {a.name} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: {"✅ 成功" if ok else "❌ 失败"}\n- **Score**: 尚未提交 | Improvement: N/A | Best: N/A\n- **训练类型**: {settings.get("training_type")} (LoRA)\n- **超参数**: `{json.dumps(settings)}`\n- **做了什么**: 用训练集的逐步解答（若data非空，则筛选模型自身最终答案正确的采样轨迹）进行仅回答部分的交叉熵训练；移除 <<计算器标记>>，保留文字推理与 #### 数字答案；随机保留256条验证数据并记录索引。保存适配器和合并后的完整模型。{detail}\n- **为什么**: 一小时预算内优先使用高质量已知正确的推理轨迹，避免在线采样成本；LoRA限制参数变化以保留原模型数学能力，验证损失用于监测泛化。\n- **问题/进步**: {detail}\n- **关键代码**: `labels = [-100]*len(prompt_ids) + answer_ids; loss.backward(); model.merge_and_unload()`\n- **下一步建议**: 释放GPU后提交完整模型，根据准确率与剩余时间调整训练。\n')
try:
 set_seed(20261108); torch.set_num_threads(8)
 tok=AutoTokenizer.from_pretrained('./models/base'); tok.pad_token=tok.eos_token
 rows=[json.loads(l) for l in open('data/train.jsonl')]
 ids=list(range(len(rows))); random.Random(20261108).shuffle(ids)
 train_ids,val_ids=ids[256:],ids[:256]
 json.dump(dict(seed=20261108,train=train_ids,validation=val_ids),open('logs/split.json','w'))
 def encode_row(r):
  prompt=tok.apply_chat_template([{'role':'user','content':r['question']+'\nPlease reason step by step, and put your final answer in the form #### <number>.'}],tokenize=False,add_generation_prompt=True)
  pi=tok.encode(prompt,add_special_tokens=False); ans=tok.encode(re.sub(r'<<.*?>>','',r['answer'])+tok.eos_token,add_special_tokens=False)
  return (pi+ans,[-100]*len(pi)+ans)
 train=[encode_row(rows[i]) for i in train_ids]; val=[encode_row(rows[i]) for i in val_ids]
 if a.data:
  generated=[json.loads(l) for l in open(a.data)]
  train=[encode_row(r) for r in generated if r['reward']==1]
 train=[r for r in train if len(r[0])<=a.max_length]; val=[r for r in val if len(r[0])<=a.max_length]
 log(event='data',train_examples=len(train),validation_examples=len(val))
 model=AutoModelForCausalLM.from_pretrained('./models/base',dtype=torch.bfloat16,attn_implementation='sdpa').cuda(); model.config.use_cache=False
 if a.resume: model=PeftModel.from_pretrained(model,a.resume,is_trainable=True)
 else: model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
 model.print_trainable_parameters()
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
 def batch(ex):
  n=(max(len(r[0]) for r in ex)+7)//8*8
  return dict(input_ids=torch.tensor([r[0]+[tok.pad_token_id]*(n-len(r[0])) for r in ex],device='cuda'),labels=torch.tensor([r[1]+[-100]*(n-len(r[1])) for r in ex],device='cuda'),attention_mask=torch.tensor([[1]*len(r[0])+[0]*(n-len(r[0])) for r in ex],device='cuda'))
 def validate(step):
  model.eval(); losses=[]
  with torch.no_grad():
   for i in range(0,min(128,len(val)),a.batch): losses.append(model(**batch(val[i:i+a.batch])).loss.item())
  log(event='validation',step=step,loss=sum(losses)/len(losses)); model.train()
 validate(0)
 order=list(range(len(train))); rng=random.Random(20261108); rng.shuffle(order); cursor=0
 start=time.time(); model.train()
 for step in range(1,a.steps+1):
  opt.zero_grad(set_to_none=True); total=0
  lr=a.lr*min(step/15,1)*(0.2+0.8*.5*(1+math.cos(math.pi*max(0,step-15)/max(1,a.steps-15))))
  for group in opt.param_groups: group['lr']=lr
  for micro in range(a.accum):
   if cursor+a.batch>len(order): rng.shuffle(order); cursor=0
   ex=[train[j] for j in order[cursor:cursor+a.batch]]; cursor+=a.batch
   loss=model(**batch(ex)).loss; total+=loss.item()/a.accum; (loss/a.accum).backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  log(event='train',step=step,loss=total,lr=lr,grad_norm=float(norm),elapsed_s=time.time()-start)
  if step%75==0: validate(step)
 validate(a.steps)
 model.save_pretrained(f'output/{a.name}_adapter'); tok.save_pretrained(f'output/{a.name}_adapter')
 model=model.merge_and_unload(); model.config.use_cache=True
 model.save_pretrained(f'output/{a.name}',safe_serialization=True); tok.save_pretrained(f'output/{a.name}')
 path=Path(f'output/{a.name}/tokenizer_config.json'); config=json.loads(path.read_text()); config.pop('extra_special_tokens',None); path.write_text(json.dumps(config,indent=2))
 log(event='saved',step=a.steps,path=f'output/{a.name}'); summary(True,f'完成{a.steps}步训练；指标见 logs/{a.name}_metrics.jsonl。')
except Exception:
 error=traceback.format_exc(); summary(False,error); raise
