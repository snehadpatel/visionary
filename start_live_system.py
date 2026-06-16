import subprocess
import os
import signal
import sys
import time
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def kill_port(port):
    """Aggressively kill any process on the specified port."""
    try:
        subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, check=False, stderr=subprocess.DEVNULL)
    except:
        pass

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    print("🚀 Starting Visionary Live System...")
    
    # 0. Optimization: Unlock full Mac RAM for AI models
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # 1. Clean up existing processes
    print("🧹 Cleaning up existing processes on ports 8000 and 3000...")
    kill_port(8000)
    kill_port(3000)
    time.sleep(1)
    
    # 1. Start Backend
    print("\n[1/3] Starting Python Backend Server (Port 8000)...")
    venv_python = os.path.join(project_root, "visionary_env", "bin", "python")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable
    
    backend_proc = subprocess.Popen(
        [python_exe, "main.py"],
        cwd=os.path.join(project_root, "backend")
    )
    time.sleep(2)
    
    # 2. Start Frontend
    print("\n[2/3] Starting React Frontend Server (Port 3000)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "0.0.0.0"],
        cwd=os.path.join(project_root, "frontend")
    )
    time.sleep(2)
    
    # 3. Start LocalTunnel
    print("\n[3/3] Starting LocalTunnel for secure mobile access...")
    tunnel_proc = subprocess.Popen(
        ["npx", "localtunnel", "--port", "3000"],
        cwd=os.path.join(project_root, "frontend")
    )
    
    local_ip = get_local_ip()
    print("\n" + "="*50)
    print("✨ VISIONARY LIVE IS READY! ✨")
    print("="*50)
    print(f"\n🖥️  To test on your Mac, open:")
    print(f"    https://localhost:3000/")
    print(f"\n📱 To test on your Mobile Device (on same Wi-Fi), open:")
    print(f"    https://{local_ip}:3000/")
    print("\n🌐 Wait a few seconds for LocalTunnel to print your public link below:")
    print("   (Use the public link if the Wi-Fi link gives camera errors)")
    print("\nPress Ctrl+C to shut down all servers.")
    print("="*50 + "\n")

    try:
        # Keep the script running
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping all servers...")
        backend_proc.send_signal(signal.SIGTERM)
        frontend_proc.send_signal(signal.SIGTERM)
        tunnel_proc.send_signal(signal.SIGTERM)
        sys.exit(0)

if __name__ == "__main__":
    main()
