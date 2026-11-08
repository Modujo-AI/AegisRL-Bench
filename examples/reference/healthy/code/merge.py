import argparse,json
from pathlib import Path
from datetime import datetime,timezone
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--weights',nargs='+',type=float,required=True);a=p.parse_args()
assert len(a.sources)==len(a.weights)
torch.set_num_threads(8)
model=AutoModelForCausalLM.from_pretrained('./models/base',dtype=torch.bfloat16)
model=PeftModel.from_pretrained(model,f'./output/{a.sources[0]}_adapter',adapter_name='a0')
for i,source in enumerate(a.sources[1:],1):model.load_adapter(f'./output/{source}_adapter',adapter_name=f'a{i}')
model.add_weighted_adapter([f'a{i}' for i in range(len(a.sources))],a.weights,'combined',combination_type='cat');model.set_adapter('combined')
model=model.merge_and_unload(adapter_names=['combined']);model.save_pretrained(f'output/{a.name}');AutoTokenizer.from_pretrained('./models/base').save_pretrained(f'output/{a.name}')
p=Path(f'output/{a.name}/tokenizer_config.json');d=json.loads(p.read_text());d.pop('extra_special_tokens',None);p.write_text(json.dumps(d,indent=2))
settings=dict(training_type='trained checkpoint weighted merge',seed=20261108,sources=a.sources,weights=a.weights,source_settings={s:json.loads(Path(f'logs/{s}_settings.json').read_text()) for s in a.sources})
Path(f'logs/{a.name}_settings.json').write_text(json.dumps(settings,indent=2))
with open('summary.md','a') as f:f.write(f'\n## Model adjustment {a.name} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: ✅ 成功\n- **训练类型**: trained checkpoint weighted merge\n- **超参数**: `{json.dumps(settings)}`\n- **做了什么**: 将{a.sources}的真实训练LoRA增量按{a.weights}加权，以cat方式精确表达加权增量后合并完整模型。\n- **为什么**: 已观察弱化SFT增量优于完整增量，尝试保留数学能力并结合不同训练信号；此过程无新增梯度训练。\n- **下一步建议**: 释放GPU后官方提交。\n')
print('Saved',a.name)
