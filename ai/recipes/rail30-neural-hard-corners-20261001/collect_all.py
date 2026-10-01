import subprocess,sys
from pathlib import Path
b=Path('logs/neural-player/expanded-100-hard-corners-20261001')
for seed in range(20261210,20261216):
 out=b/str(seed);out.mkdir(exist_ok=True)
 with (out/"collect.log").open("w") as log:
  subprocess.run([sys.executable,str(b/"collect.py"),"--seed",str(seed),"--output",str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
