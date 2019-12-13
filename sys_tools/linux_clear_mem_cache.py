# -*- coding: utf8 -*-
import subprocess

def linux_clear_mem_cache():
    out=subprocess.check_output("free",shell = True)
    mem=out.split('\n')[1].split()
    if float(mem[3])/float(mem[1])<0.02:
        print("clear linux mem cache")
        subprocess.call("sync; echo 1 > /proc/sys/vm/drop_caches", shell = True)