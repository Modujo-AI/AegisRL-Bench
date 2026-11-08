"""Infrastructure wrapper around the pinned, unmodified upstream evaluator/server."""
import json,os,shutil,sys,time
from pathlib import Path
from flask import request,jsonify
from rdagent.scenarios.rl.autorl_bench.benchmarks import get_evaluator
from rdagent.scenarios.rl.autorl_bench.core import server as upstream
CONTROL=Path('/control'); CONTROL.mkdir(exist_ok=True)
if not (CONTROL/'rdagent').exists(): (CONTROL/'rdagent').symlink_to('/upstream/rdagent')
BASE='Qwen/Qwen2.5-1.5B-Instruct'
if sys.argv[1]=='baseline':
    result=get_evaluator('gsm8k').run_eval('/model',str(CONTROL/'baseline'),model_name=BASE,gpu_count=1)
    (CONTROL/'baseline.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    sys.exit(1 if result.get('error') else 0)
class IsolatedEvaluator:
    def run_eval(self,model_path,workspace_path,**kwargs):
        index=len(list((CONTROL/'evaluations').glob('*'))) if (CONTROL/'evaluations').exists() else 0
        dest=CONTROL/'evaluations'/f'{index+1:03d}';dest.mkdir(parents=True)
        # Freeze each submitted model, preventing later writes from changing a score.
        src=Path(model_path);root=Path('/workspace').resolve();device=root.stat().st_dev
        if not src.is_relative_to(root/'output'):raise ValueError('Submit a trained model under output/')
        for f in src.rglob('*'):
            if not f.resolve().is_relative_to(root) or f.stat().st_dev!=device:
                raise ValueError('Output must reside on the project filesystem')
        shutil.copytree(src,dest/'model')
        started=time.time()
        result=get_evaluator('gsm8k').run_eval(str(dest/'model'),str(dest),**kwargs)
        (dest/'result.json').write_text(json.dumps({'started':started,'finished':time.time(),**result},indent=2))
        return result
server=upstream.init_server('gsm8k',BASE,'/workspace')
server.scores_file=CONTROL/'scores.json'
server.get_evaluator=lambda:IsolatedEvaluator()
server.set_baseline(json.loads((CONTROL/'baseline.json').read_text())['score'])
# Timing/control records must remain writable when the project filesystem is full.
original_update=upstream.update_run_meta
original_read=upstream.read_run_meta
upstream.update_run_meta=lambda workspace,**kwargs:original_update(CONTROL/'meta',**kwargs)
upstream.read_run_meta=lambda workspace:original_read(CONTROL/'meta')
@upstream.app.before_request
def restrict_control():
    if request.path=='/set_baseline':return jsonify(error='Not available to task agents'),403
upstream.app.run(host='0.0.0.0',port=5000,threaded=True)
