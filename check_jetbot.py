import subprocess
import platform

JETBOT_IPS = ["192.168.137.134", "192.168.137.95","192.168.137.79"]


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


connected_ips = []

for ip in JETBOT_IPS:
    if ping(ip):
        connected_ips.append(ip)

# Results
if connected_ips:
    print("✅ JetBot(s) connected:", connected_ips)
else:
    print("❌ No JetBot connected")