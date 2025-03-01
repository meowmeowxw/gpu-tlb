#include <cuda.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>

#ifndef COMMON_H
#define COMMON_H


struct __eviction_set {
    uint8_t *base_address;
    uint32_t *indexes;
    uint32_t count;
};

__device__ void sort(uint64_t *arr, uint32_t n) {
    for (uint32_t i = 0; i < n - 1; ++i) {
        for (uint32_t j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                uint64_t temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

__device__ uint64_t median(uint64_t *cycle_count, uint32_t count) {
    sort(cycle_count, count);
    return cycle_count[(uint32_t)count / 2];
}

__global__ void put(uint64_t *page, uint64_t x1, uint64_t x2) {
    page[0] = x1;
    page[1] = x2;
}

void parse_eviction_set(const char *filename, struct __eviction_set *es) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("fopen");
        exit(-1);
    }
    fscanf(fp, "%lx", (uint64_t *)&es->base_address);
    fscanf(fp, "%u", &es->count);
    uint32_t number = 0;
    uint32_t count = 0;
    es->indexes = (uint32_t *)malloc(sizeof(uint32_t) * es->count);
    while (fscanf(fp, "%u", &number) == 1) {
        es->indexes[count++] = number;
    }
    // printf("base_address: 0x%lx\n", (uint64_t)es->base_address);
    // printf("count: %u\n", es->count);
    // for (int i = 0; i < es->count; ++i) {
    //     printf("indexes[%u]: %u\n", i, es->indexes[i]);
    // }
    assert(es->count == count);
    fclose(fp);
}

#endif
