"""Exercise launcher signal ordering with fake binaries, never real hardware."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def test_terminal_interrupt_brakes_runner_then_stops_master_once(tmp_path):
    root = tmp_path
    for name in ('ai/bin','sw/build','vision/build','bin'):
        (root/name).mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'bin/play.sh', root/'ai/bin/play.sh')
    (root/'vision/build/blobtrack').write_text('');(root/'vision/build/blobtrack').chmod(0o755)
    def executable(name, text):
        p=root/name;p.write_text(text);p.chmod(0o755)
    executable('bin/make','#!/bin/sh\nexit 0\n')
    executable('sw/build/cdpr_master',f'''#!{sys.executable}
import signal,time
from pathlib import Path
count=0
def stop(*args):
 global count
 count+=1
 with open('events','a') as f:f.write('master_signal_'+str(count)+'\\n')
 if count>1:raise SystemExit(8)
 time.sleep(.3)
 with open('events','a') as f:f.write('master_disabled\\n')
 raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
Path('master_ready').touch()
while True:time.sleep(.05)
''')
    executable('bin/python3',f'''#!{sys.executable}
import sys,time,signal
from pathlib import Path
if len(sys.argv)==2 and sys.argv[1]=='-':
 sys.stdin.read()
 raise SystemExit(0 if Path('master_ready').exists() else 1)
def stop(*args):
 with open('events','a') as f:f.write('runner_brake\\n')
 time.sleep(.2)
 with open('events','a') as f:f.write('runner_closed\\n')
 raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
Path('runner_ready').touch()
while True:time.sleep(.05)
''')
    env=dict(os.environ,PATH=str(root/'bin')+os.pathsep+os.environ['PATH'])
    process=subprocess.Popen(['bash','ai/bin/play.sh','--policy','builtin:hold'],cwd=root,
                             env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+8
        while not (root/'runner_ready').exists() and time.monotonic()<deadline:
            if process.poll() is not None:break
            time.sleep(.02)
        assert (root/'runner_ready').exists()
        os.killpg(process.pid,signal.SIGINT)
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==0,(stdout,stderr)
        assert (root/'events').read_text().splitlines()==['runner_brake','runner_closed','master_signal_1','master_disabled']
    finally:
        if process.poll() is None:
            process.kill();process.wait()
