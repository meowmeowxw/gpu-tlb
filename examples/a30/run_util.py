#!/usr/bin/python3

import subprocess
import time
import os



dumper_path = "../../dumper/dumper"
extractor_path = "../../extractor/extractor"
modifier_path = "../../modifier/modifier"
tmp_path = "/tmp/"

target_va = 0x700000000000
dummy_va = 0x7f0000000000
code_pa = 0x0007c00000


def prepare(argv):
    os.system("make")
    #===============================================================================
    # modify PTE for target_va to make it point to the dummy_va"s page frame
    #===============================================================================
    sudo = subprocess.Popen(["sudo", "echo", ""])
    sudo.wait()
    a_out = subprocess.Popen(argv)
    b_out = subprocess.Popen(["nvidia-smi", "-f", "/dev/null"])
    b_out.wait() # in case the driver warms up very slowly
    time.sleep(2)
    tmp_path = "/tmp/"
    dump = subprocess.Popen([dumper_path, "-b", "0x10000000", "-o", tmp_path + "dump"])
    dump.wait()
    pagemap_file = open(tmp_path + "pagemap", "w")
    extract = subprocess.Popen([extractor_path, tmp_path + "dump"], stdout=pagemap_file)
    extract.wait()
    pagemap_file.close()
    a_out.kill()
    time.sleep(5)



