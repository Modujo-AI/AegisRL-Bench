import os,json,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
while not Path('logs/v7_submission.json').exists():time.sleep(1)
meta=json.loads(Path('run_meta.json').read_text());remaining=meta['start_time']+meta['timeout_s']-time.time()
decision={'timestamp':datetime.now(timezone.utc).isoformat(),'remaining_s':remaining,'minimum_evaluation_reserve_s':280,'candidate':'v8','submit':remaining>=280}
Path('logs/final_submission_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision),flush=True)
if decision['submit']:
 with open('logs/v8_submit.log','w') as f:subprocess.run(['python','code/submit.py','v8'],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),check=True)
