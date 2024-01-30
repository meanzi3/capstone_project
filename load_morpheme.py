import json
import os
import math

# for i in range(1, 19):
#     os.makedirs('morpheme/{0:02d}'.format(i))

path = 'data/morpheme'
direction = ['D', 'F', 'L', 'R', 'U']
min_start = 10
max_end = 0
(min_start_i, min_start_j, min_start_k) = (0, 0, 0)
(max_end_i, max_end_j, max_end_k) = (0, 0, 0)

for i in range(1, 19):
    for j in range(1, 5):
        for k in range(5):
            # print(str(i) + ', ' + str(j))
            file_name = '{:02d}/NIA_SL_WORD15{:02d}_REAL{:02d}_'.format(i, j, i) + direction[k] + '_morpheme.json'
            with open(os.path.join(path, file_name), 'r', encoding='UTF8') as f:
                last_json = json.load(f)
                start = last_json['data'][0]['start']
                end = last_json['data'][0]['end']
                if start < min_start:
                    min_start = start
                    (min_start_i, min_start_j, min_start_k) = (i, j, k)
                if end > max_end:
                    max_end = end
                    (max_end_i, max_end_j, max_end_k) = (i, j, k)

print('min_start updated at REAL{:02d}, WORD15{:02d}, {}'.format(min_start_i, min_start_j, direction[min_start_k]))
print('max_end updated at REAL{:02d}, WORD15{:02d}, {}'.format(max_end_i, max_end_j, direction[max_end_k]))
print()
print('second basis min_start: ', min_start)
print('second basis max_end: ', max_end)
print('frame basis min_start: ', math.trunc(min_start * 30))
print('frame basis max_end: ', math.ceil(max_end * 30))