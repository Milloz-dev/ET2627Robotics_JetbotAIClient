import subprocess
import platform

JETBOT_IP = "192.168.137.134"

def ping(ip):
    system = platform.system().lower()

    if system == "windows":
        command = ["ping", "-n", "1", ip]
    else:
        command = ["ping", "-c", "1", ip]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.returncode == 0

if ping(JETBOT_IP):
    print("✅ JetBot connected")
else:
    print("❌ JetBot not connected")