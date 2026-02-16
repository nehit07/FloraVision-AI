import sys
from pathlib import Path

# Add project root and src to path
root = Path(__file__).parent.parent
sys.path.append(str(root))
sys.path.append(str(root / "src"))

from floravision.graph import run_diagnosis_full
from floravision.state import PlantState

def test_batch_processing():
    # Simulate 3 different images
    images = [
        b"image_1_data",
        b"image_2_data",
        b"image_3_data"
    ]
    
    print(f"Starting batch test with {len(images)} images...")
    
    results = []
    for i, img_bytes in enumerate(images):
        print(f"  Processing image {i+1}...")
        state = run_diagnosis_full(img_bytes, season="summer", mock=True)
        results.append(state)
        
    assert len(results) == 3
    print(f"✅ Batch test passed: {len(results)} reports generated.")
    
    for i, res in enumerate(results):
        print(f"  Report {i+1}: {res.plant_name} ({res.severity or 'Healthy'})")

if __name__ == "__main__":
    test_batch_processing()
