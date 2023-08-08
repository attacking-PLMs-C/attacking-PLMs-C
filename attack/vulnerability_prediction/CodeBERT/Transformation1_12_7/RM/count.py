import os
import json
import operator

if __name__ == "__main__":
    cnt = 0
    file_candidate = {}
    with open('all_candidates_count.txt', 'r') as fp:
        all_candidates = fp.read().strip().split('\n')
        for candidate in all_candidates:
            fname = candidate.split('  ')[0]
            v_count = int(candidate.split('  ')[-1])
            if v_count not in file_candidate:
                file_candidate[v_count] = 1
            else:
                file_candidate[v_count] += 1
            if v_count >= 5:
                cnt += 1

    file_candidate = dict(sorted(file_candidate.items(), key=operator.itemgetter(0)))
    with open('file_candidates.json', 'w') as  fp:
        json.dump(file_candidate, fp)
    print(cnt)