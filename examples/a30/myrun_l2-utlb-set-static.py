#!/usr/bin/python3

import subprocess
import time
import pagemap
import random
from run_util import *

prepare(['./l2-utlb-set-static'])

# extract PTEs from the pagemap
target = 0xb7
target_va = target_va + target * (1 * 1024 * 1024)
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

cmd = ["./l2-utlb-set-static"]
a_out = subprocess.Popen(cmd)
b_out = subprocess.Popen(['nvidia-smi', '-f', '/dev/null'])
b_out.wait() # in case the driver warms up very slowly
time.sleep(1)
subprocess.Popen(['sudo', modifier_path, hex(pte_pa), hex(pte_val)])
time.sleep(12)
print("modification done")
a_out.kill()
if a_out.poll() != None:
    print("eviction successful")
else:
    print("no eviction")