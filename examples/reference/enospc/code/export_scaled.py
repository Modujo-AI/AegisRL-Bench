import json,os,argparse,time
from datetime import datetime,timezone
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from retry_io import retry
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--source',default='v3');p.add_argument('--scale',type=float,default=.5);a=p.parse_args()
torch.set_num_threads(8)
os.makedirs('logs/'+a.run,exist_ok=True)
cfg=json.load(open('logs/'+a.source+'/settings.json'));cfg.update(export_type='scaled trained LoRA merge',adapter_scale=a.scale,source_run=a.source,run=a.run,device='cpu')
def write(path,text):
 with open(path,'w') as f:f.write(text)
retry(lambda:write('logs/'+a.run+'/settings.json',json.dumps(cfg,indent=2)))
model=AutoModelForCausalLM.from_pretrained('./models/base',torch_dtype=torch.bfloat16,device_map='cpu')
model=PeftModel.from_pretrained(model,'output/'+a.source+'-adapter')
count=0
for module in model.modules():
 if isinstance(getattr(module,'scaling',None),dict) and 'default' in module.scaling:
  module.scaling['default']*=a.scale;count+=1
model=model.merge_and_unload();model.config.use_cache=True
retry(lambda:model.save_pretrained('output/'+a.run,safe_serialization=True))
tok=AutoTokenizer.from_pretrained('./models/base');retry(lambda:tok.save_pretrained('output/'+a.run))
path='output/'+a.run+'/tokenizer_config.json';tc=json.load(open(path));tc.pop('extra_special_tokens',None);retry(lambda:write(path,json.dumps(tc,indent=2)))
entry=f'\n## Export {a.run} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: ✅ 成功\n- **Score**: 待提交 | Improvement: N/A | Best: 65.58\n- **训练类型**: SFT训练适配器的缩放合并（非新训练）\n- **超参数**: '+json.dumps(cfg,ensure_ascii=False)+f'\n- **做了什么**: 将{a.source}训练得到的{count}个LoRA模块的有效权重更新乘以{a.scale}，合并到基础模型，导出完整模型。CPU执行，未占用评测GPU。\n- **为什么**: 高强度SFT曾降低准确率，而低强度训练提高准确率；测试更小参数更新能否保留数学能力并保留格式学习收益。\n- **问题/进步**: 保留原最佳模型；此版本含非零训练更新，并非复制基础模型。\n- **关键代码**: `module.scaling["default"] *= scale; model.merge_and_unload()`\n- **下一步建议**: 完整测试集评估并与最佳结果比较。\n'
def append():
 with open('summary.md','a') as f:f.write(entry)
retry(append);print('Export complete',a.run,count,flush=True)
