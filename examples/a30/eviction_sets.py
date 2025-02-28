#!/usr/bin/env python3

xored = [
    [44, 36, 28, 20],
    [45, 37, 29, 21],
    [46, 38, 30, 22],
    [39, 31, 23],
    [40, 32, 24],
    [41, 33, 25],
    [42, 34, 26],
    [43, 35, 27],
]

slice_xor = [i for i in range(25, 47)]

def get_virtual_address(address):
    cache_set = [0 for _ in range(len(xored))]
    for i, values in enumerate(xored):
        for v in values:
            mask = 1 << v
            cache_set[i] ^= ((address & mask) >> v)

    return int("".join(map(str, cache_set)), 2)


def get_eviction_set(cache_set, base_address=0x700000000000, indexed=True, n=9):
    start_addr = base_address
    addresses = []
    while len(addresses) < n:
        start_addr += 0x100000
        if get_virtual_address(start_addr) == cache_set:
            addresses.append(start_addr)
    if indexed:
        results = []
        for addr in addresses:
            results.append((base_address ^ addr) >> 20)
        return results
    return addresses


def get_slice_address(addr):
    slice = 0
    for sx in slice_xor:
        slice = slice ^ ((addr >> (sx)) & 1)
    return slice

def get_slice(eviction_set):
    new_eviction_set = []
    for addr in eviction_set:
        slice = 0
        for sx in slice_xor:
            slice = slice ^ ((addr >> (sx)) & 1)
        if slice == 1:
            new_eviction_set.append(addr)
    return new_eviction_set



if __name__ == "__main__":
    eviction_set = get_eviction_set(13, indexed=True)
    for index in eviction_set:
        print(hex(index))
    # for addr in eviction_set:
    #     print(hex(addr), hex((addr >> 20) ^ 0x7000000))



