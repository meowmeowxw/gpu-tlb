#!/usr/bin/python3

import subprocess
import time
import pagemap
from run_util import *

prepare(["./fill"])

# extract PTEs from the pagemap
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')

print(f"{hex(target_va)}: {ptes[target_va]}")
print(f"{hex(dummy_va)}: {ptes[dummy_va]}")

pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

print(f"pte_pa: {hex(pte_pa)}")
print(f"pte_val: {hex(pte_val)}")

# do experiments
a_out = subprocess.Popen(['./fill'])
b_out = subprocess.Popen(['nvidia-smi', '-f', '/dev/null'])
b_out.wait() # in case the driver warms up very slowly
time.sleep(2)
subprocess.Popen(['sudo', modifier_path, hex(pte_pa), hex(pte_val)])

time.sleep(10)

a_out.kill()
if a_out.poll() != None:
  print('eviction successful')
else:
  print('no eviction')


