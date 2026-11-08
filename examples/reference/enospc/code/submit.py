import os,json,sys,time,urllib.request
from datetime import datetime,timezone
from retry_io import retry
run=sys.argv[1]
settings=json.load(open('logs/'+run+'/settings.json'))
request=urllib.request.Request(os.environ['GRADING_SERVER_URL']+'/submit',data=json.dumps({'model_path':'./output/'+run}).encode(),headers={'Content-Type':'application/json'})
print('Submitting',run,datetime.now(timezone.utc).isoformat(),flush=True)
try:
 with urllib.request.urlopen(request,timeout=1500) as r: raw=r.read().decode()
except urllib.error.HTTPError as e:raw=e.read().decode()
print(raw,flush=True)
entry=f'\n## Submission {run} ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: 已返回评测结果\n- **训练类型**: SFT (LoRA, merged full model)\n- **超参数**: '+json.dumps(settings,ensure_ascii=False)+'\n- **做了什么**: 训练进程结束、释放GPU后，将 output/'+run+' 完整模型提交完整测试集评测。\n- **为什么**: 使用官方准确率反馈选择后续训练强度，保留最佳提交。\n- **Score / Improvement / Best（服务器原始响应）**:\n```json\n'+raw+'\n```\n- **问题/进步**: 以上为服务器原始结果；未读取评测实现或私有数据。\n- **下一步建议**: 根据分数、验证损失和剩余时间决定继续训练或保留本版。\n'
def append():
 with open('summary.md','a') as f:f.write(entry); f.flush()
retry(append)
def save():
 with open('logs/'+run+'/submission.json','w') as f:f.write(raw)
retry(save)
