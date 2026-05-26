import json

with open('json/Custom_CNN_metrics.json', 'r') as f:
    data = json.load(f)

for cls in data:
    report = data[cls]['classification_report']
    print(f"{cls:20s} → P: {report['precision']:.2f}  R: {report['recall']:.2f}  F1: {report['f1-score']:.2f}")