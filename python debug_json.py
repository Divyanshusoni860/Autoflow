import json

with open('dataset/fcb/train.json', 'r') as f:
    data = json.load(f)

print(f"Top-level keys: {list(data.keys())}")

# Let's peek into one key
first_key = list(data.keys())[0]
print(f"\nFirst key: {first_key}")
print(f"Value under that key (type): {type(data[first_key])}")
print(f"Sample data:\n{data[first_key]}")
