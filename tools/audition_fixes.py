import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.0.107', username='raspberrypi5', password='raspberrypi5', timeout=10)

script = '''
import subprocess
import time

def play(text, scale="1.0"):
    cmd = ["echo", text]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    piper = subprocess.Popen(
        ["/home/raspberrypi5/piper/piper/piper", "--model", "/home/raspberrypi5/piper/en_US-ryan-medium.onnx", "--output-raw", "--length-scale", scale],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    p1.stdout.close()
    aplay = subprocess.Popen(
        ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-q", "-"],
        stdin=piper.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    piper.stdout.close()
    aplay.wait()
    piper.wait()

print("--- Testing word CAT ---")
print("1. Playing 'cat' (lowercase)")
play("cat")
time.sleep(0.5)

print("2. Playing 'CAT' (uppercase)")
play("CAT")
time.sleep(0.5)

print("3. Playing 'Cat.' (sentence with period)")
play("Cat.")
time.sleep(0.5)

print("4. Playing 'cat' with length-scale 1.1")
play("cat", scale="1.1")
time.sleep(0.5)

print("--- Testing M, N, S variants ---")
for text in ["M", "Em", "Em.", "N", "En", "En.", "S", "Ess", "Ess."]:
    cmd_esp = ["espeak-ng", "-v", "en-us", "-q", "--ipa", text]
    p = subprocess.Popen(cmd_esp, stdout=subprocess.PIPE)
    ipa = p.communicate()[0].decode().strip()
    print(f"Variant: {text:6} -> {ipa}")
    play(text)
    time.sleep(0.3)

print("--- Testing punctuation (. and ,) ---")
for text in ["period", "Period.", "dot", "comma", "Comma."]:
    cmd_esp = ["espeak-ng", "-v", "en-us", "-q", "--ipa", text]
    p = subprocess.Popen(cmd_esp, stdout=subprocess.PIPE)
    ipa = p.communicate()[0].decode().strip()
    print(f"Punct: {text:8} -> {ipa}")
    play(text)
    time.sleep(0.3)

print("--- Testing divided by vs multiplied by ---")
for text in ["divided by", "Divided by.", "multiplied by", "Multiplied by."]:
    print(f"Math: {text}")
    play(text)
    time.sleep(0.3)
'''

sftp = ssh.open_sftp()
with sftp.file('/home/raspberrypi5/audition_fixes.py', 'w') as f:
    f.write(script)
sftp.close()

stdin, stdout, stderr = ssh.exec_command('python3 /home/raspberrypi5/audition_fixes.py')
out = stdout.read().decode('utf-8', errors='replace').encode('ascii', errors='backslashreplace').decode('ascii')
print(out)
err = stderr.read().decode('utf-8', errors='replace')
if err:
    print("ERR:", err)
ssh.close()
