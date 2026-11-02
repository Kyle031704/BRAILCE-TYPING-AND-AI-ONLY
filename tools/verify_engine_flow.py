import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.0.107', username='raspberrypi5', password='raspberrypi5', timeout=10)

script = r'''
import sys
sys.path.insert(0, '/home/raspberrypi5/brailce')
from brailce.speech.piper_engine import PiperEngine
import time

engine = PiperEngine()
print('PiperEngine is_available:', engine.is_available())

# Simulate typing C, A, T rapidly:
print("Simulating rapid typing of C, A, T...")
engine.speak('C')
engine.speak('A')
engine.speak('T')

time.sleep(2.5)

# Simulate pressing Space to pronounce the completed word "cat":
print("Simulating space press for word 'cat'...")
engine.speak('cat', interrupt=False)

time.sleep(1.5)
print("Finished flow test!")
'''

sftp = ssh.open_sftp()
with sftp.file('/home/raspberrypi5/test_flow.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command('/home/raspberrypi5/brailce/venv/bin/python3 /home/raspberrypi5/test_flow.py')
out = stdout.read().decode('utf-8', errors='replace').encode('ascii', errors='backslashreplace').decode('ascii')
print(out)
err = stderr.read().decode('utf-8', errors='replace')
if err:
    print("ERR:", err)

# Ensure everything is killed so user has clean terminal
ssh.exec_command("pkill -9 -f piper 2>/dev/null; pkill -9 -f aplay 2>/dev/null; pkill -9 -f 'python3.*brailce' 2>/dev/null")

ssh.close()
