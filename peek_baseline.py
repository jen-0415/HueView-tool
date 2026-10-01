"""Quick peek: what does the real baseline classification actually return?"""
from ml_pipeline.src.inference.pipeline import classify_image

path = r"C:\Users\jennifer015\Desktop\fourth year\thesis\hueview_tool\data\processed\faces\MST-1\28_Brazilian Faces28-01_face_1.jpg"

result = classify_image(open(path, "rb").read())

b = result["models"]["baseline"]
print("SCC:", b["scc"])
print("Confidence:", b["confidence"])
print("Margin:", b["margin"])
print("Probabilities:", b["probabilities"])
print("Undertone:", b["undertone"]["label"])