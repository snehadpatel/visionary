import torch
try:
    print("Attempting to load MiDaS DPT_Large...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    print("✅ MiDaS loaded successfully!")
except Exception as e:
    print(f"❌ Error loading MiDaS: {e}")
