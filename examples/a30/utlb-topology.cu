#include "common.h"
#include <cuda.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)
#define STRIDE_SIZE (1L * 1024L * 1024L)

#define BASE_ADDR_SMID0 0x700000000000
#define BASE_ADDR_SMID1 0x702000000000
#define DUMMY_ADDR 0x7F0000000000

#define PAGE0_NUM 8000
#define PAGE1_NUM 30000
#define PAGE_DTLB_NUM 16
#define WAIT_TIME 5000000000L

#define BLK_NUM 100
#define SHARED_MEM (96 * 1024)

uint32_t SMID0 = 0;
uint32_t SMID1 = 1;

__global__ void loop(volatile uint64_t *page0, volatile uint64_t *page1, volatile uint64_t *page2, uint64_t x,
                     uint32_t SMID0, uint32_t SMID1) {
    uint64_t y = x;
    volatile uint64_t *ptr;
    volatile uint64_t *evt;
    uint64_t clk0 = 0;
    uint64_t clk1 = 0;
    uint32_t smid;

    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    if (smid != SMID0 && smid != SMID1)
        return;
    // if (smid != SMID0)
    //   return;

    if (smid == SMID0) {
        while (y == x) {
            // printf("l2: accessing %p, page0[0]: %p\n", page0, page0[0]);
            for (ptr = (uint64_t *)page0[0]; ptr != page0; ptr = (uint64_t *)ptr[0]) {
                // printf("l2: accessing %p\n", ptr);
                ++ptr[2];
            }
            for (evt = (uint64_t *)page2[0]; evt != page2; evt = (uint64_t *)evt[0]) {
                // printf("l1: accessing %p\n", evt);
                ++evt[2];
            }

            // printf("timer routine\n");
            clk0 = clock64();
            clk1 = 0;
            while (clk1 < WAIT_TIME)
                clk1 = clock64() - clk0;

            y = ptr[1];
            printf("y: %lx\n", y);
        }
    } else if (smid == SMID1) {
        // clk0 = clock64();
        // clk1 = 0;
        // while (clk1 < WAIT_TIME1)
        //   clk1 = clock64() - clk0;
        // printf("l2: accessing %p, page1[0]: %p\n", page1, page1[0]);
        while (y == x) {
            // printf("looped page1: %p, page1[0]: %p\n");
            for (ptr = (uint64_t *)page1[0]; ptr != page1; ptr = (uint64_t *)ptr[0]) {
                // printf("ptr: %p, ptr[0]: %p\n", ptr, ptr[0]);
                ++ptr[2];
            }
            for (evt = (uint64_t *)page2[0]; evt != page2; evt = (uint64_t *)evt[0])
                ++evt[2];
            y = ptr[1];
        }
    }

    page0[1] = 0;
    page1[1] = 0;
}

int main(int argc, char *argv[]) {
    uint8_t *chunk0 = NULL;
    uint8_t *chunk1 = NULL;
    uint8_t *base = NULL;
    uint64_t *list_smid0[PAGE0_NUM];
    uint64_t *list_smid1[PAGE1_NUM];
    uint64_t *list_dtlb[PAGE_DTLB_NUM];
    struct __eviction_set es_smid0;
    struct __eviction_set es_smid1;

    int aim = -1;
    uint64_t *dummy = NULL;

    parse_eviction_set("./out/eviction_set_smid0.txt", &es_smid0);
    parse_eviction_set("./out/eviction_set_smid1.txt", &es_smid1);

    cudaDeviceReset();
    cudaFuncSetAttribute(loop, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);

    // hoard a large address space
    cudaMallocManaged(&chunk0, CHUNK0_SIZE);

    cudaMallocManaged(&chunk1, CHUNK1_SIZE);

    SMID0 = atoi(argv[1]);
    SMID1 = atoi(argv[2]);

    aim = es_smid0.indexes[0];

    base = (uint8_t *)BASE_ADDR_SMID0;
    for (int i = 0; i < PAGE0_NUM; ++i)
        list_smid0[i] = (uint64_t *)(base + i * STRIDE_SIZE);
    base = (uint8_t *)BASE_ADDR_SMID1;
    for (int i = 0; i < PAGE1_NUM; ++i)
        list_smid1[i] = (uint64_t *)(base + i * STRIDE_SIZE);
    for (int i = 0; i < PAGE_DTLB_NUM; ++i)
        list_dtlb[i] = list_smid0[aim + i + 1];
    dummy = (uint64_t *)DUMMY_ADDR;

    put<<<1, 1>>>(dummy, 0, 0);

    for (int i = 0; i < es_smid0.count; ++i) {
        int m = es_smid0.indexes[i];
        int n = es_smid0.indexes[(i + 1) % es_smid0.count];
        put<<<1, 1>>>(list_smid0[m], (uint64_t)list_smid0[n], 0xdeadbeef);
        printf("[smid0] m: 0x%lx, n: 0x%lx\n", (uint64_t)list_smid0[m], (uint64_t)list_smid0[n]);
    }
    for (int i = 0; i < es_smid1.count; ++i) {
        int m = es_smid1.indexes[i];
        int n = es_smid1.indexes[(i + 1) % es_smid1.count];
        put<<<1, 1>>>(list_smid1[m], (uint64_t)list_smid1[n], 0xdeadbeef);
        printf("[smid1] m: 0x%lx, n: 0x%lx\n", (uint64_t)list_smid1[m], (uint64_t)list_smid1[n]);
    }

    for (int i = 0; i < PAGE_DTLB_NUM; ++i)
        put<<<1, 1>>>(list_dtlb[i], (uint64_t)list_dtlb[(i + 1) % PAGE_DTLB_NUM], 0xdeadbeef);
    cudaDeviceSynchronize();

    printf("Done hoarding, aim: 0x%x, page: 0x%lx, smid: 0x%x\n", aim, (uint64_t)list_smid0[aim], SMID1);
    loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_smid0[aim], list_smid1[es_smid1.indexes[0]], list_dtlb[0],
                                     0xdeadbeef, SMID0, SMID1);
    // loop<<<BLK_NUM, 1, SHARED_MEM>>>(list_smid0[aim], list_smid1[atoi(argv[2])], list_dtlb[0], 0xdeadbeef);
    cudaDeviceSynchronize();

    cudaFree(chunk0);
    cudaFree(chunk1);

    free(es_smid0.indexes);
    free(es_smid1.indexes);
}
