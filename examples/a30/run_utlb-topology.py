#!/usr/bin/python3

import pagemap
import sys
from run_util import *
from eviction_sets import *

BASE_ADDR_SMID0 = 0x700000000000
BASE_ADDR_SMID1 = 0x702000000000

num_smid0 = 4
num_smid1 = 20

if len(sys.argv) > 2:
    num_smid0 = int(sys.argv[1])
    num_smid1 = int(sys.argv[2])
elif len(sys.argv) == 2:
    print("[*] pass both num_smid0 and num_smid1")
    exit(1)

es_smid0 = get_eviction_set(18, n=num_smid0, indexed=True, base_address=BASE_ADDR_SMID0)
target = es_smid0[0]
target_va = BASE_ADDR_SMID0 + target * (1 * 1024 * 1024)
es_smid0 = list(map(str, es_smid0))
write_to_file(es_smid0, "./out/eviction_set_smid0.txt")
print(f"[*] eviction set for smid0: {es_smid0}")

es_smid1 = list(map(str, get_eviction_set(18, n=num_smid1, indexed=True, base_address=BASE_ADDR_SMID1)))
write_to_file(es_smid1, "./out/eviction_set_smid1.txt")
print(f"[*] eviction set for smid1: {es_smid1}")

cmd = ["./utlb-topology", "0", "1"]
print(f"[*] cmd: {' '.join(cmd)}")
prepare(cmd, sleep_time=3)
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')
print(f"[*] target_va: {hex(target_va)}")
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

for smid1 in range(1, 56):
    cmd[2] = str(smid1)
    print(f"[*] cmd: {' '.join(cmd)}")
    if launch_eviction(cmd, pte_pa, pte_val, sleep_time=10):
        print(f"[*] valid: {smid1}")
