# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 09:09:05 2020

@author: mimif
"""

import os
from collections import defaultdict

root_dir = "D:\\PointCloudCourse\\HW9\\ICP_plane\\"

fnames = os.listdir(root_dir)

testcase2counts = defaultdict(lambda : 0)

for fname in fnames:
    if "log" in fname or "pca" in fname:
        continue
    # testcase, _ = fname.rsplit('_',1)
    testcase, _, _ = fname.rsplit('_',2)
    testcase2counts[testcase] += 1
    
remove_count = 0
for fname in fnames:
    # testcase, it = fname.rsplit('_',1)
    # it = int(it[:-4])
    if "log" in fname or "pca" in fname:
        continue
    testcase, it, _ = fname.rsplit('_',2)
    it = int(it)
    # print(testcase, it)
    if it != testcase2counts[testcase]-1 and \
        it % 10 != 0:
        os.remove(os.path.join(root_dir, fname))
        # print(fname)
        remove_count += 1
    else:
        # print(fname)
        pass

print("remove", remove_count, "/", len(fnames))