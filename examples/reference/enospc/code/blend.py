import os,json,argparse,torch
from datetime import datetime,timezone
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import load_file
from retry_io import retry
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--left',default='v4');p.add_argument('--right',default='v5');p.add_argument('--alpha',type=float,required=True);a=p.parse_args()
torch.set_num_threads(8);os.makedirs('logs/'+a.run,exist_ok=True)
cfg=json.load(open('logs/'+a.right+'/settings.json'));cfg.update(export_type='linear combination of trained full models',left=a.left,right=a.right,alpha=a.alpha,run=a.run,device='cpu')
def write(path,text):
 with open(path,'w') as f:f.write(text)
retry(lambda:write('logs/'+a.run+'/settings.json',json.dumps(cfg,indent=2)))
model=AutoModelForCausalLM.from_pretrained('output/'+a.left,torch_dtype=torch.bfloat16,device_map='cpu')
right=load_file('output/'+a.right+'/model.safetensors',device='cpu')
changed=0
with torch.no_grad():
 for name,t in model.state_dict().items():
  if name in right:
   t.copy_(torch.lerp(t.float(),right[name].float(),a.alpha).to(t.dtype));changed+=1
retry(lambda:model.save_pretrained('output/'+a.run,safe_serialization=True))
tok=AutoTokenizer.from_pretrained('./models/base');retry(lambda:tok.save_pretrained('output/'+a.run))
path='output/'+a.run+'/tokenizer_config.json';tc=json.load(open(path));tc.pop('extra_special_tokens',None);retry(lambda:write(path,json.dumps(tc,indent=2)))
entry=f'\n## Export {a.run} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: ✅ 成功\n- **Score**: 待提交 | Improvement: N/A | Best: 见最近提交\n- **训练类型**: RFT/SFT训练模型权重线性组合（非新训练）\n- **超参数**: '+json.dumps(cfg,ensure_ascii=False)+f'\n- **做了什么**: CPU计算 ({1-a.alpha})*{a.left}+({a.alpha})*{a.right}，处理{changed}个权重张量，保存完整模型。两个输入均为本任务实际训练后的模型。\n- **为什么**: 通过插值或外推控制第二轮RFT更新强度，测试泛化与训练强度的折衷；避免再次训练占用评测GPU。\n- **问题/进步**: 保留所有已提交最佳模型。\n- **关键代码**: `torch.lerp(left.float(),right.float(),alpha)`\n- **下一步建议**: 全测试集提交比较。\n'
def append():
 with open('summary.md','a') as f:f.write(entry)
retry(append);print('Export complete',a.run,flush=True)
