import os
import sys

# Ensure 'src' is in the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_redesign import train

if __name__ == "__main__":
    print("🚀 Starting Custom Redesign Model Training...")
    print("📍 Target: Real-Time Mobile Performance (30 FPS)")
    print("🛠️  Platform: Apple Silicon (MPS)")
    
    try:
        train()
    except KeyboardInterrupt:
        print("\n🛑 Training paused. Progress saved.")
    except Exception as e:
        print(f"\n❌ Training Error: {e}")
