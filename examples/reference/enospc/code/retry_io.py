import time, errno

def retry(fn):
 while True:
  try:return fn()
  except Exception as e:
   if getattr(e,'errno',None)!=errno.ENOSPC and not any(x in str(e).lower() for x in ['no space left','os error 28']):raise
   print('Workspace full; retrying write in 5 seconds',flush=True); time.sleep(5)
