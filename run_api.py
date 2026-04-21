import subprocess
import time

while True:
    print("Starting Flask API...")

    process = subprocess.Popen([
        "python",
        "app.py"
    ])

    process.wait()

    print("API crashed. Restarting in 3 seconds...")

    time.sleep(3)

