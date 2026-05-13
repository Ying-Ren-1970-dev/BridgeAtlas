import json

with open('test_results_20260511_231603.json', 'r') as f:
    data = json.load(f)

print("Examining first 3 failed tests:\n")

for i, test in enumerate(data['tests'][:3], 1):
    print(f"{'='*60}")
    print(f"Test {i}: {test['scenario_type']} - {test['element']}")
    print(f"Query: {test['query']}")
    print(f"Expected pages: {test['expected_pages']}")
    print(f"Returned pages: {test['returned_pages']}")
    print(f"Status: {test['status']}")
    print(f"Precision: {test['metrics']['precision']:.2f}, Recall: {test['metrics']['recall']:.2f}")
    print()
