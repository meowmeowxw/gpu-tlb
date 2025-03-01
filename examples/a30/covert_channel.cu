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

#define PAGE_NUM_RECEIVER 10000
#define PAGE_NUM_SENDER   10000
#define PAGE_NUM_L1  16
#define WAIT_TIME 500000000L
#define WAIT_TIME2 100000000L

#define BLK_NUM 100
#define SHARED_MEM (96 * 1024)
#define SMID0 0
#define SMID1 1


__device__ void rest(uint64_t wait_time) {
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    clk0 = clock64();
    clk1 = 0;
    while (clk1 < wait_time)
        clk1 = clock64() - clk0;
}

__device__ uint64_t iterate_pages(volatile uint64_t *page_l2, volatile uint64_t *page_l1) {
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    uint64_t cycle_count[10] = {0};
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
    printf("[*] time: %lu\n", cycle_count[count]);
    for (; ptr != page_l2; ptr = (uint64_t *)ptr[0]) {
        // printf("accessing page: 0x%lx\n", (uint64_t)ptr);
        ++ptr[2];
    }

// l1_eviction:
    for (evt = (uint64_t *)page_l1[0]; evt != page_l1; evt = (uint64_t *)evt[0]) {
        ++evt[2];
    }

    count++;

    rest(WAIT_TIME);

    if (count > 10) {
        return median(cycle_count, count);
    }

    goto l2_l3_eviction;
}

__device__ void receiver(volatile uint64_t *page_l2, volatile uint64_t *page_l1, uint32_t len) {
    uint64_t cycle_count = 0;
    printf("[!] page_l2: 0x%lx, page_l2[0]: 0x%lx\n", (uint64_t)page_l2, (uint64_t)page_l2[0]);
    for (int i = 0; i < len; ++i) {
        cycle_count = iterate_pages(page_l2, page_l1);
        if (cycle_count < 1100) {
            printf("[!] RECEIVED: 0\n");
        } else {
            printf("[!] RECEIVED: 1\n");
        }
        printf("[-] receiver | cycle_count: %lu\n", cycle_count);
    }
}

__device__ void sender(volatile uint64_t *page_l2, volatile uint64_t *page_l1, uint32_t *to_send, uint32_t len) {

    // uint64_t cycle_count = 0;

    for (int i = 0; i < len; ++i) {
        if (to_send[i] == 1) {
            printf("[!] sender | 1\n");
            iterate_pages(page_l2, page_l1);
        } else {
            printf("[!] sender | 0\n");
            for (int j = 0; j < 10; ++j) {
                rest(WAIT_TIME);
            }
        }
    }
}

__global__ void loop(volatile uint64_t *page_sender, volatile uint64_t *page_receiver, volatile uint64_t *page_l1) {
    uint32_t smid;
    uint32_t to_send[] = {1, 0, 1, 1, 0};
    uint32_t len = sizeof(to_send) / sizeof(to_send[0]);
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid != SMID0 && smid != SMID1)
        return;

    if (smid == SMID0) {
        receiver(page_receiver, page_l1, len);
    } else {
        sender(page_sender, page_l1, to_send, len);
    }
}

int main(int argc, char *argv[]) {
    uint8_t *chunk0 = NULL;
    uint8_t *chunk1 = NULL;
    uint64_t *list_rcv[PAGE_NUM_RECEIVER];
    uint64_t *list_snd[PAGE_NUM_SENDER];
    uint64_t *list_l1[PAGE_NUM_L1];
    struct __eviction_set es_rcv;
    struct __eviction_set es_snd;
    // uint64_t indexes[] = {0xb7, 0x1b6, 0x2b5, 0x3b4, 0x4b3, 0x5b2, 0x6b1, 0x7b0, 0x8bf};
    // uint64_t indexes[] = {0xb7, 0x1b9, 0x2b5, 0x3b4, 0x4b3, 0x5b2, 0x6b1, 0x7b0, 0x8bf};

    parse_eviction_set("eviction_set_rcv.txt", &es_rcv);
    parse_eviction_set("eviction_set_snd.txt", &es_snd);
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

    for (int i = 0; i < es_rcv.count; ++i) {
        int m = es_rcv.indexes[i];
        int n = es_rcv.indexes[(i + 1) % es_rcv.count];
        put<<<1, 1>>>(list_rcv[m], (uint64_t)list_rcv[n], 0xdeadbeef);
    	printf("[rcv] m: 0x%lx, n: 0x%lx\n", (uint64_t)list_rcv[m], (uint64_t)list_rcv[n]);
    }
    // for (int i = 0; i < PAGE_NUM_SENDER; ++i) {
    //     put<<<1, 1>>>(list_snd[i], (uint64_t)list_snd[(i + 1) % PAGE_NUM_SENDER], 0xdeadbeef);
    // }
    // for (int i = 0; i < es_snd.count; ++i) {
    //     int m = es_snd.indexes[i];
    //     int n = es_snd.indexes[(i + 1) % es_snd.count];
    //     put<<<1, 1>>>(list_snd[m], (uint64_t)list_snd[n], 0xdeadbeef);
    // 	printf("[snd] m: 0x%lx, n: 0x%lx\n", (uint64_t)list_snd[m], (uint64_t)list_snd[n]);
    // }
    for (int i = 0; i < PAGE_NUM_L1; ++i) {
        put<<<1, 1>>>(list_l1[i], (uint64_t)list_l1[(i + 1) % PAGE_NUM_L1], 0xdeadbeef);
    }

    cudaDeviceSynchronize();

    printf("Done hoarding, aim: 0x%x, page: 0x%lx\n", aim, (uint64_t)list_rcv[aim]);
    // loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_snd[es_snd.indexes[0]], list_rcv[aim], list_l1[0]);
    loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_snd[0], list_rcv[aim], list_l1[0]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaFree(chunk0);
    cudaFree(chunk1);

    free(es_rcv.indexes);
    free(es_snd.indexes);
}
