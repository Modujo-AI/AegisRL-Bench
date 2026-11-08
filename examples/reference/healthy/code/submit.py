import json,os,sys,time,urllib.request
from pathlib import Path
from datetime import datetime,timezone
name=sys.argv[1]
settings=json.loads(Path(f'logs/{name}_settings.json').read_text())
req=urllib.request.Request(os.environ['GRADING_SERVER_URL']+'/submit',data=json.dumps({'model_path':f'./output/{name}','gpu':'0'}).encode(),headers={'Content-Type':'application/json'})
start=time.time()
try:
 with urllib.request.urlopen(req,timeout=1500) as r: raw=r.read().decode()
except Exception as e:
 raw=json.dumps({'error':str(e)})
Path(f'logs/{name}_submission.json').write_text(raw)
try: result=json.loads(raw)
except: result={}
with open('summary.md','a') as f:
 f.write(f'\n## Submission {name} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: {"✅ 成功" if "score" in result else "❌ 失败"}\n- **Score**: {result.get("score","N/A")} | Improvement: {result.get("improvement","N/A")} | Best: {result.get("best","N/A")}\n- **训练类型**: {settings.get("training_type", "SFT")} (LoRA merged full model)\n- **超参数**: `{json.dumps(settings)}`\n- **做了什么**: 训练进程结束并释放GPU后，将 output/{name} 完整模型提交到官方评测；等待完整测试集评测结果，用时{time.time()-start:.1f}秒。\n- **为什么**: 使用官方准确率反馈选择下一轮训练策略，保留已提交最佳模型。\n- **问题/进步**: 官方返回如下。\n- **下一步建议**: 根据分数和剩余时间继续训练或保留最佳结果。\n- **Grading server response (verbatim)**:\n```json\n{raw}\n```\n')
print(raw,flush=True)
