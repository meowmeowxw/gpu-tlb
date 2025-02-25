#!/usr/bin/python3

import subprocess
import time
import pagemap
import os
import sys
from run_util import *


cmd = ["./l1-dtlb-cap", "1"]
prepare(cmd)

# extract PTEs from the pagemap
ptes = pagemap.retrieve_ptes(tmp_path + "pagemap")
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

print(f"{hex(target_va)}: {ptes[target_va]}")
print(f"{hex(dummy_va)}: {ptes[dummy_va]}")
print(f"pte_pa: {hex(pte_pa)}")
print(f"pte_val: {hex(pte_val)}")

if len(sys.argv) > 1:
  start_range = int(sys.argv[1])
  end_range = start_range + 1
else:
  start_range = 1
  end_range = 56

for SMID1 in range(start_range, end_range):
  b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
  time.sleep(1)
  b_out.kill()
  print(f"[*] testing: {SMID1}")
  cmd[1] = str(SMID1)
  launch_eviction(cmd, pte_pa, pte_val, sleep_time=10)


