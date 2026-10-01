"""Quick peek: what does the full dual-model result look like now?"""
from ml_pipeline.src.inference.pipeline import classify_image

path = r"C:\Users\jennifer015\Desktop\fourth year\thesis\hueview_tool\data\processed\images\MST-1\28_Brazilian Faces28-01_face_1.jpg"

result = classify_image(open(path, "rb").read())

print("ILLUMINATION:", result["illumination"])

b = result["models"]["baseline"]
print(f"\nBASELINE:  scc={b['scc']}  conf={b['confidence']:.3f}  undertone={b['undertone']['label']}")

h = result["models"]["hueview"]
print(f"\nHUEVIEW headline:  scc={h['scc']}  conf={h['confidence']:.3f}  margin={h['margin']:.3f}")
print(f"HUEVIEW undertone: {h['undertone']['label']}")
print("\nPER-REGION:")
for r in h["regions"]:
    print(f"  {r['name']:<28} scc={r['scc']}  conf={(r['confidence'] or 0):.3f}")