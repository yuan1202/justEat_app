import requests
import json
import random
import re

import multiprocessing
import grequests

import numpy as np

import matplotlib.pyplot as plt


# ---
# ---
# ### load data

with open('users.json', 'r') as f:
    contents = f.readlines()
    
parsed_contents = [json.loads(l) for l in contents]

with open('users_batch.json', 'r') as f:
    contents_batch = f.readlines()
    
parsed_contents_batch = [json.loads(l) for l in contents_batch]


print('------------------------------------------------')
print('basic test single queries')

for c in contents:
    print('--------------')
    result = requests.post('http://localhost:5000/infer', json=c)
    print(result.json())
    print('elapsed {:.2f} seconds.'.format(result.elapsed.total_seconds()))


print('------------------------------------------------')
print('stress test single queries')

def request_job(q):
    c = contents[random.choice(range(5))]
    result = requests.post('http://localhost:5000/infer', json=c)
    assert result.status_code == 200
    q.put(result.elapsed.total_seconds())

q = multiprocessing.Queue()

n_requests = 1000
processes = []

for i in range(n_requests):
    p = multiprocessing.Process(target=request_job, args=(q,))
    p.start()
    processes.append(p)

for p in processes:
    # call to ensure all processes finish
    p.join() 

assert q.qsize() == n_requests

elapsed_times = []
for _ in range(q.qsize()):
    elapsed_times.append(q.get())
    
q.close()
    
print('average elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.mean(elapsed_times)))

_, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(range(len(elapsed_times)), elapsed_times)
ax.grid()
ax.set_ylabel('elapsed time (seconds)')
ax.set_xlabel('request index')

# same test using grequests
n_requests = 1000
rs = (grequests.post('http://localhost:5000/infer', json=contents[random.choice(range(5))]) for i in range(n_requests))
rts = grequests.map(rs)

elapsed_times = []
for rt in rts:
    if rt is None:
        print(rt)
    else:
        elapsed_times.append(rt.elapsed.total_seconds())
        
print('average elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.mean(elapsed_times)))
print('worst elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.max(elapsed_times)))
print('best elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.min(elapsed_times)))

_, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(range(len(elapsed_times)), elapsed_times)
ax.grid()
ax.set_ylabel('elapsed time (seconds)')
ax.set_xlabel('request index')


print('------------------------------------------------')
print('batched query test')

result = requests.post('http://localhost:5000/batch', json=json.dumps(parsed_contents_batch))
print('elapsed {:.2f} seconds.'.format(result.elapsed.total_seconds()))
result = result.json()
parsed_result = [dict((('name', c['name']), ('likes', c['likes']), *list(r.items()))) for c, r in zip(parsed_contents_batch, result)]

for r in parsed_result:
    if (r['best'] == None) or (r['best'] == 'None'):
        print(r)


# ##### check against single request to see if answer match

check_index = 550
print(parsed_result[check_index])
result = requests.post('http://localhost:5000/infer', json=contents_batch[check_index])
print(result.json())
print('elapsed {:.2f} seconds.'.format(result.elapsed.total_seconds()))


print('------------------------------------------------')
print('stress test batch request')

dumped_contents_batch = json.dumps(parsed_contents_batch)

def request_batchJob(q):
    result = requests.post('http://localhost:5000/batch', json=dumped_contents_batch)
    assert result.status_code == 200
    q.put(result.elapsed.total_seconds())

q = multiprocessing.Queue()

n_requests = 10
processes = []

for i in range(n_requests):
    p = multiprocessing.Process(target=request_batchJob, args=(q,))
    p.start()
    processes.append(p)

for p in processes:
    # call to ensure all processes finish
    p.join() 

assert q.qsize() == n_requests

elapsed_times = []
for _ in range(q.qsize()):
    elapsed_times.append(q.get())
    
q.close()
    
print('average elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.mean(elapsed_times)))
print('worst elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.max(elapsed_times)))
print('best elapsed time for {} requests is {:.6f} seconds.'.format(n_requests, np.min(elapsed_times)))

_, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.plot(range(len(elapsed_times)), elapsed_times)
ax.grid()
ax.set_ylabel('elapsed time (seconds)')
ax.set_xlabel('request index')


# ---
# ---
# ### error handling

# single query
for c in contents:
    print('--------------')
    error_injection = random.randint(0, 1)
    if error_injection:
        
        error_type = random.randint(0, 3)
        print('error {} injected'.format(error_type))
        
        if error_type == 0:
            c = c.replace('vector', 'blablabla')
        elif error_type == 1:
            c = c.replace(random.choice(re.findall('(-?\d.\d*)', c)), 'NaN')
        else:
            c = ''
        print(c)
    result = requests.post('http://localhost:5000/infer', json=c)
    if error_injection:
        print(result.status_code, result.json())
    else:
        print(result.json())
    print('elapsed {:.2f} seconds.'.format(result.elapsed.total_seconds()))

# batch query
with open('users_batch.json', 'r') as f:
    contents_batch = f.readlines()
    
error_type = 1
print('error {} selected'.format(error_type))

bad_requests = np.random.choice(range(len(contents_batch)), 2, replace=False)
for i in range(len(contents_batch)):
    if i in bad_requests:

        if error_type == 0:
            contents_batch[i] = contents_batch[i].replace('vector', 'blablabla')
        elif error_type == 1:
            contents_batch[i] = contents_batch[i].replace(random.choice(re.findall('(-?\d.\d*)', contents_batch[i])), 'NaN')
        else:
            contents_batch[i] = ''
        print(contents_batch[i])
    
parsed_contents_batch = [json.loads(l) for l in contents_batch]

result = requests.post('http://localhost:5000/batch', json=json.dumps(parsed_contents_batch))
print('elapsed {:.2f} seconds.'.format(result.elapsed.total_seconds()))

result = result.json()

if error_type == 1:
    for idx in bad_requests:
        print(result[idx])
else:
    print(result)
