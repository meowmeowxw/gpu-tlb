#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "common.h"

#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)
#define STRIDE_SIZE (1L * 1024L * 1024L)

#define BASE_ADDR_RECEIVER  0x700000000000
#define BASE_ADDR_SENDER    0x702000000000
#define DUMMY_ADDR 0x7F0000000000

#define PAGE_NUM_RECEIVER 20000
#define PAGE_NUM_SENDER   80000
#define PAGE_NUM_L1  16
#define PAGE_FILL_NUM  6000
#define WAIT_TIME 500000000L
#define WAIT_TIME2 100000000L
#define ITERATIONS 5

#define BLK_NUM 100
#define SHARED_MEM (96 * 1024)
#define SMID0 0
#define SMID1 2


__device__ void rest(uint64_t wait_time) {
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    clk0 = clock64();
    clk1 = 0;
    while (clk1 < wait_time)
        clk1 = clock64() - clk0;
}

__device__ uint64_t iterate_pages(volatile uint64_t *page_l2, volatile uint64_t *page_l1,
                                  uint32_t sm) {
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    uint64_t cycle_count[ITERATIONS + 1] = {0};
    uint32_t count = 0;
    volatile uint64_t *ptr;
    volatile uint64_t *evt;

l2_l3_eviction:
    asm volatile("membar.cta;");
    clk0 = clock64();
    ptr = (uint64_t *)page_l2[0];
    clk1 = clock64();
    asm volatile("membar.cta;");

    cycle_count[count] = clk1 - clk0;
    // printf("[%u] time: %lu\n", sm, cycle_count[count]);
    for (; ptr != page_l2; ptr = (uint64_t *)ptr[0]) {
        // printf("accessing page: 0x%lx\n", (uint64_t)ptr);
        ++ptr[2];
    }

// l1_eviction:
    for (evt = (uint64_t *)page_l1[0]; evt != page_l1; evt = (uint64_t *)evt[0]) {
        atomicAdd((unsigned long long int *)(&evt[2]), 1);
        // ++evt[2];
    }

    count++;

    rest(WAIT_TIME);

    if (count > ITERATIONS) {
        return cycle_count[1];
        // return median(cycle_count, count);
    }
    goto l2_l3_eviction;
}

__device__ void receiver(volatile uint64_t *page_l2, volatile uint64_t *page_l1, uint32_t len) {
    uint64_t cycle_count = 0;
    printf("[rcv] page_l2: 0x%lx, page_l2[0]: 0x%lx\n", (uint64_t)page_l2, (uint64_t)page_l2[0]);
    uint32_t message[10] = {0};
    // for (int i = 0; i < 100; i++) {
    //     rest(WAIT_TIME);
    // }
    for (int i = 0; i < len; ++i) {
        printf("[rcv] page_l2[0]: 0x%lx\n", (uint64_t)page_l2[0]);
        // asm volatile("membar.cta;");
        cycle_count = iterate_pages(page_l2, page_l1, SMID0);
        if (cycle_count < 1100) {
            message[i] = 0;
            // printf("[!] RECEIVED: 0\n");
        } else {
            message[i] = 1;
            // printf("[!] RECEIVED: 1\n");
        }
        printf("[-] receiver | cycle_count: %lu\n", cycle_count);
    }
    for (int i = 0; i < len; ++i) {
        printf("[rcv] message[%u]: %u\n", i, message[i]);
    }
}

__device__ void sender(volatile uint64_t *page_l2, volatile uint64_t *page_l1, uint32_t *to_send, uint32_t len) {
    printf("[snd] page_l2: 0x%lx, page_l2[0]: 0x%lx\n", (uint64_t)page_l2, (uint64_t)page_l2[0]);
    for (int i = 0; i < len; ++i) {
        if (to_send[i] == 1) {
            printf("[!] sender | 1\n");
            iterate_pages(page_l2, page_l1, SMID1);
        } else {
            printf("[!] sender | 0\n");
            for (int j = 0; j < ITERATIONS; ++j) {
                rest(WAIT_TIME);
            }
        }
    }
}

__device__ uint32_t barrier = 0;

__global__ void loop(volatile uint64_t *page_sender, volatile uint64_t *page_receiver, volatile uint64_t *page_l1,
                     volatile uint64_t *page_fill) {
    uint32_t smid;
    uint32_t to_send[] = {1, 0, 1, 1, 1, 0};
    uint32_t len = sizeof(to_send) / sizeof(to_send[0]);
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid != SMID0 && smid != SMID1)
        return;

    for (uint32_t i = 0; i < len; i++) {

        for (uint64_t *ptr = (uint64_t *)page_fill[0]; ptr != page_fill; ptr = (uint64_t *)ptr[0]) {
            ++ptr[2];
        }

        atomicAdd(&barrier, 1);
        while (atomicOr(&barrier, 0) != 2) {
            ;
        }
        __threadfence();  // Ensure memory visibility
        atomicExch(&barrier, 0);

        if (smid == SMID0) {
            uint64_t cycle_count = iterate_pages(page_receiver, page_l1, smid);
            printf("[rcv] cycle_count: %lu\n", cycle_count);
            // receiver(page_receiver, page_l1, len);
        } else {
            if (to_send[i] == 1) {
                printf("[!] sender | 1\n");
                iterate_pages(page_sender, page_l1, SMID1);
            } else {
                printf("[!] sender | 0\n");
                for (int j = 0; j < ITERATIONS; ++j) {
                    rest(WAIT_TIME);
                }
            }
            // sender(page_sender, page_l1, to_send, len);
        }
    }
}

int main(int argc, char *argv[]) {
    uint8_t *chunk0 = NULL;
    uint8_t *chunk1 = NULL;
    uint64_t *list_rcv[PAGE_NUM_RECEIVER];
    uint64_t *list_snd[PAGE_NUM_SENDER];
    uint64_t *list_l1[PAGE_NUM_L1];
    uint64_t **list_fill =       (uint64_t **)_malloc(sizeof(uint64_t *) * (PAGE_FILL_NUM + 1));
    struct __eviction_set es_rcv;
    struct __eviction_set es_snd;
    // uint64_t indexes[] = {0xb7, 0x1b6, 0x2b5, 0x3b4, 0x4b3, 0x5b2, 0x6b1, 0x7b0, 0x8bf};
    // uint64_t indexes[] = {0xb7, 0x1b9, 0x2b5, 0x3b4, 0x4b3, 0x5b2, 0x6b1, 0x7b0, 0x8bf};

    parse_eviction_set("./out/eviction_set_rcv.txt", &es_rcv);
    parse_eviction_set("./out/eviction_set_snd.txt", &es_snd);
    int aim = -1;

    cudaDeviceReset();
    cudaFuncSetAttribute(loop, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);

    // hoard a large address space
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);
    cudaMallocManaged(&chunk1, CHUNK1_SIZE);

    aim = es_rcv.indexes[0];

    for (int i = 0; i < PAGE_NUM_RECEIVER; ++i)
        list_rcv[i] = (uint64_t *)(((uint8_t *)BASE_ADDR_RECEIVER) + i * STRIDE_SIZE);
    for (int i = 0; i < PAGE_NUM_SENDER; ++i)
        list_snd[i] = (uint64_t *)(((uint8_t *)BASE_ADDR_SENDER) + i * STRIDE_SIZE);
    for (int i = 0; i < PAGE_NUM_L1; ++i)
        list_l1[i] = list_rcv[aim + i + 1];
    for (int i = 1; i < PAGE_FILL_NUM + 1; ++i) {
        list_fill[i] = (uint64_t *)((uint8_t *)(DUMMY_ADDR) + i * STRIDE_SIZE);
    }


    uint64_t *dummy = (uint64_t *)DUMMY_ADDR;
    put<<<1, 1>>>(dummy, 0, 0);

    for (int i = 0; i < es_rcv.count; ++i) {
        int m = es_rcv.indexes[i];
        int n = es_rcv.indexes[(i + 1) % es_rcv.count];
        put<<<1, 1>>>(list_rcv[m], (uint64_t)list_rcv[n], 0xdeadbeef);
    	printf("[rcv] m: 0x%lx, n: 0x%lx\n", (uint64_t)list_rcv[m], (uint64_t)list_rcv[n]);
    }
    // for (int i = 0; i < PAGE_NUM_SENDER; ++i) {
    //     put<<<1, 1>>>(list_snd[i], (uint64_t)list_snd[(i + 1) % PAGE_NUM_SENDER], 0xdeadbeef);
    // }
    for (int i = 0; i < es_snd.count; ++i) {
        int m = es_snd.indexes[i];
        int n = es_snd.indexes[(i + 1) % es_snd.count];
        put<<<1, 1>>>(list_snd[m], (uint64_t)list_snd[n], 0xdeadbeef);
    	printf("[snd] m: 0x%lx, n: 0x%lx\n", (uint64_t)list_snd[m], (uint64_t)list_snd[n]);
    }
    for (int i = 0; i < PAGE_NUM_L1; ++i) {
        put<<<1, 1>>>(list_l1[i], (uint64_t)list_l1[(i + 1) % PAGE_NUM_L1], 0xdeadbeef);
    }
    for (int i = 1; i < PAGE_FILL_NUM + 1; ++i) {
        int j = (i + 1) % PAGE_FILL_NUM == 0 ? 1 : i + 1;
        put<<<1, 1>>>(list_fill[i], (uint64_t)list_fill[j], 3);
    }

    cudaDeviceSynchronize();

    printf("Done hoarding, aim: 0x%x, page: 0x%lx\n", aim, (uint64_t)list_rcv[aim]);
    loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_snd[es_snd.indexes[0]], list_rcv[aim], list_l1[0], list_fill[1]);
    // loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_snd[0], list_rcv[aim], list_l1[0]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaFree(chunk0);
    cudaFree(chunk1);

    free(es_rcv.indexes);
    free(es_snd.indexes);
    free(list_fill);
}
