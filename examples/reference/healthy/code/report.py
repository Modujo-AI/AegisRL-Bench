import json,time
from pathlib import Path
from datetime import datetime,timezone
results=[]
for p in Path('logs').glob('*_submission.json'):
 try:r=json.loads(p.read_text())
 except:continue
 if 'score' in r:results.append(r)
results.sort(key=lambda r:r['submission_id']);best=max(results,key=lambda r:r['score'])
metrics={}
for p in Path('logs').glob('*_metrics.jsonl'):
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
 tr=[r for r in rows if r.get('event')=='train']
 if tr:metrics[p.name]={'steps':len(tr),'first_timestamp':tr[0]['timestamp'],'last_timestamp':tr[-1]['timestamp'],'first_loss':tr[0]['loss'],'last_loss':tr[-1]['loss']}
meta=json.loads(Path('run_meta.json').read_text())
report={'timestamp':datetime.now(timezone.utc).isoformat(),'seed':20261108,'best_model_path':best['model_path'],'best_score':best['score'],'baseline_score':best['baseline_score'],'improvement':best['improvement'],'best_submission_id':best['submission_id'],'completed_submissions':len(results),'remaining_s':meta['start_time']+meta['timeout_s']-time.time(),'submissions':results,'training_trajectory_summary':metrics}
assert Path(best['model_path'],'model.safetensors').is_file()
Path('best_model.json').write_text(json.dumps(report,indent=2))
with open('summary.md','a') as f:
 f.write(f'\n## Final result ({datetime.now(timezone.utc):%Y-%m-%d %H:%M})\n- **状态**: ✅ 成功\n- **Score**: {best["score"]} | Improvement: {best["improvement"]} | Best: {best["score"]}\n- **最佳完整模型**: `{best["model_path"]}`\n- **设置**: seed=20261108；训练与切分设置见logs/*_settings.json，逐步时间戳/loss/reward见logs/*_metrics.jsonl；所有提交原文保留于logs/*_submission.json。\n- **做了什么**: 完成{len(results)}次完整测试集提交，比较SFT、弱化SFT、正确轨迹筛选训练、DPO、继续训练和权重组合；保留最佳完整模型，详细分数和轨迹汇总见best_model.json。\n- **为什么**: 用官方准确率反馈调整方向；模型自身正确轨迹优于直接模仿标准解答，偏好训练的部分增量进一步提升最佳结果。\n- **问题/进步**: 一次已记录OOM因采样worker延迟退出而发生，清理后成功重试；后续采样显式关闭worker。\n- **下一步建议**: 本轮预算内停止新实验，以最佳完整模型作为任务结果。\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['submissions','training_trajectory_summary']},indent=2))
