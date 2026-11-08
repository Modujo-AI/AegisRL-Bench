import os, json, time, random, re, argparse, traceback, math
from retry_io import retry
from datetime import datetime, timezone
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
p=argparse.ArgumentParser(); p.add_argument('--run',default='v1'); p.add_argument('--steps',type=int,default=225); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seconds',type=int,default=1100); p.add_argument('--resume'); a=p.parse_args()
cfg=dict(training_type='SFT',seed=20261108,lr=a.lr,max_steps=a.steps,batch_size=8,gradient_accumulation=4,effective_batch_size=32,max_length=512,lora_rank=32,lora_alpha=64,lora_dropout=0.0,weight_decay=0.01,warmup_steps=15,validation_size=128,seconds=a.seconds,resume=a.resume,run=a.run)
os.makedirs('logs/'+a.run,exist_ok=True)
json.dump(cfg,open('logs/'+a.run+'/settings.json','w'),indent=2)
metrics_path='logs/'+a.run+'/metrics.jsonl'
def metric_write(text):
 with open(metrics_path,'a') as f: f.write(text); f.flush()
def log(**kw):
 row=dict(timestamp=datetime.now(timezone.utc).isoformat(),hyperparameters=cfg,**kw); retry(lambda: metric_write(json.dumps(row)+'\n')); print(json.dumps({k:v for k,v in row.items() if k!='hyperparameters'}),flush=True)
def summary(ok,detail):
 with open('summary.md','a') as f:
  if f.tell()==0:f.write('# 运行总结\n')
  f.write(f'\n## Attempt {a.run} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: '+('✅ 成功' if ok else '❌ 失败')+'\n- **Score**: 尚未提交 | Improvement: N/A | Best: N/A\n- **训练类型**: SFT (LoRA)\n- **超参数**: '+json.dumps(cfg,ensure_ascii=False)+'\n- **做了什么**: 自编 code/train.py；仅使用公开训练集，固定种子划分128条验证数据，去除计算器标记但保留推理和 #### 答案；使用聊天模板，屏蔽提示词损失。训练所有注意力和MLP线性层的LoRA，保存适配器并合并为完整模型。每一步记录时间、损失、学习率、梯度范数和超参数。\n- **为什么**: 一小时预算下，监督训练能直接学习可靠数学解题过程和最终答案格式；LoRA降低优化成本，验证损失用于检查泛化；长度分桶减少填充计算。\n- **问题/进步**: '+detail+'\n- **关键代码**: `labels=[-100]*len(prompt_ids)+answer_ids; model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,...))`\n- **下一步建议**: 提交完整模型，依据分数和剩余时间调整训练强度。\n')
try:
 set_seed(cfg['seed']); torch.set_num_threads(8)
 tok=AutoTokenizer.from_pretrained('./models/base'); tok.pad_token=tok.eos_token
 rows=[json.loads(x) for x in open('data/train.jsonl')]; ids=list(range(len(rows))); random.Random(cfg['seed']).shuffle(ids)
 json.dump({'seed':cfg['seed'],'validation':ids[:128],'train':ids[128:]},open('logs/'+a.run+'/split.json','w'))
 def encode(i):
  r=rows[i]; q=tok.apply_chat_template([{'role':'user','content':r['question']}],tokenize=True,add_generation_prompt=True)
  ans=re.sub(r'<<.*?>>','',r['answer']); ans=tok.encode(ans+tok.eos_token,add_special_tokens=False)
  return ((q+ans)[:512],([-100]*len(q)+ans)[:512])
 train=[encode(i) for i in ids[128:]]; valid=[encode(i) for i in ids[:128]]
 def collate(xs):
  n=max(len(x[0]) for x in xs); n=math.ceil(n/8)*8
  return {'input_ids':torch.tensor([x[0]+[tok.pad_token_id]*(n-len(x[0])) for x in xs],device='cuda'),'attention_mask':torch.tensor([[1]*len(x[0])+[0]*(n-len(x[0])) for x in xs],device='cuda'),'labels':torch.tensor([x[1]+[-100]*(n-len(x[0])) for x in xs],device='cuda')}
 model=AutoModelForCausalLM.from_pretrained('./models/base',torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda'); model.config.use_cache=False
 if a.resume:model=PeftModel.from_pretrained(model,a.resume,is_trainable=True)
 else:model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
 model.print_trainable_parameters()
 optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr,weight_decay=.01)
 def validate(step):
  model.eval(); losses=[]
  with torch.no_grad():
   for i in range(0,len(valid),8):losses.append(model(**collate(valid[i:i+8])).loss.item())
  log(step=step,validation_loss=sum(losses)/len(losses)); model.train()
 validate(0); start=time.time(); order=[]; epoch=0
 def neworder():
  global epoch
  rng=random.Random(cfg['seed']+epoch); shuffled=list(range(len(train))); rng.shuffle(shuffled); result=[]
  for i in range(0,len(shuffled),256):
   chunk=sorted(shuffled[i:i+256],key=lambda j:len(train[j][0])); batches=[chunk[j:j+8] for j in range(0,len(chunk),8)]; rng.shuffle(batches); result.extend(batches)
  epoch+=1; return result
 for step in range(1,a.steps+1):
  optimizer.zero_grad(set_to_none=True); total=0
  lr=a.lr*min(1,step/15)*(.25+.75*.5*(1+math.cos(math.pi*step/a.steps)))
  for group in optimizer.param_groups:group['lr']=lr
  for micro in range(4):
   if not order:order=neworder()
   batch=collate([train[j] for j in order.pop()]); loss=model(**batch).loss; total+=loss.item()/4; (loss/4).backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0).item(); optimizer.step()
  log(step=step,loss=total,learning_rate=lr,grad_norm=norm,elapsed_seconds=time.time()-start,epoch=epoch)
  if step%75==0:validate(step)
  if time.time()-start>a.seconds:break
 validate(step)
 retry(lambda: model.save_pretrained('output/'+a.run+'-adapter')); model=model.merge_and_unload(); model.config.use_cache=True
 retry(lambda: model.save_pretrained('output/'+a.run,safe_serialization=True)); retry(lambda: tok.save_pretrained('output/'+a.run))
 path='output/'+a.run+'/tokenizer_config.json'; tc=json.load(open(path)); tc.pop('extra_special_tokens',None); json.dump(tc,open(path,'w'),indent=2)
 log(step=step,event='saved',model_path='output/'+a.run)
 retry(lambda: summary(True,f'完成 {step} 步，训练耗时 {time.time()-start:.1f} 秒；指标详见 logs/{a.run}/metrics.jsonl。'))
except BaseException:
 detail=traceback.format_exc(); summary(False,detail); raise
