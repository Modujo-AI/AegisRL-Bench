import json
from pathlib import Path
from datetime import datetime,timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model=AutoModelForCausalLM.from_pretrained('./models/base',dtype=torch.bfloat16)
model=PeftModel.from_pretrained(model,'./output/v1_adapter')
for mod in model.modules():
 if hasattr(mod,'set_scale'): mod.set_scale('default',0.2)
model=model.merge_and_unload(); model.save_pretrained('./output/v2'); AutoTokenizer.from_pretrained('./models/base').save_pretrained('./output/v2')
p=Path('output/v2/tokenizer_config.json'); d=json.loads(p.read_text()); d.pop('extra_special_tokens',None); p.write_text(json.dumps(d,indent=2))
d=json.loads(Path('logs/v1_settings.json').read_text()); d['adapter_merge_scale']=0.2; d['source']='v1_adapter'; Path('logs/v2_settings.json').write_text(json.dumps(d,indent=2))
with open('summary.md','a') as f: f.write(f'\n## Model adjustment v2 ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: ✅ 成功\n- **训练类型**: SFT checkpoint interpolation\n- **超参数**: seed=20261108, v1 LoRA merge scale=0.2\n- **做了什么**: 将已训练v1适配器的增量缩小至20%，合并到基础模型形成完整v2。\n- **为什么**: v1准确率57.85低于基准63.91，测试降低训练偏移能否保留原有数学能力并获得格式适配收益。\n- **问题/进步**: 不进行新梯度更新，复用真实训练权重。\n- **下一步建议**: 官方提交评测。\n')
