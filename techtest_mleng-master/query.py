import requests
import json

with open('users.json', 'r') as f:
    content = f.readline()
    while content:
        json_content = json.loads(content)
        print(f"Querying: {json_content['name']} who likes {json_content['likes']}")
        result = requests.post('http://localhost:5000/infer', json=content)
        
        print(result.json())
        content = f.readline()