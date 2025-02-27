#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>

#define CHUNK0_SIZE (64L * 1024L * 1024L * 1024L * 1024L + 0x55554000000L)
#define CHUNK1_SIZE (41L * 1024L * 1024L * 1024L * 1024L + 0x0ffc8000000L)
#define STRIDE_SIZE (1L * 1024L * 1024L)

#define BASE_ADDR   0x700000000000
#define DUMMY_ADDR  0x7F0000000000

#define PAGE0_NUM   8000
#define PAGE1_NUM   9000
#define PAGE2_NUM   16
#define WAIT_TIME   5000000000L // about 5 seconds on RTX3080
#define WAIT_TIME1  1000000000L // about 5 seconds on RTX3080

#define BLK_NUM     100
#define SHARED_MEM  (96 * 1024)
#define SMID0       0
// #define SMID1       5 // IMPORTANT: SM0 and SM3 are in different GPCs on RTX3080

uint32_t SMID1 = 1;

__global__ void 
loop(volatile uint64_t *page0, volatile uint64_t *page1, volatile uint64_t *page2, uint64_t x, uint32_t SMID1)
{
  uint64_t y = x;
  volatile uint64_t *ptr;
  volatile uint64_t *evt;
  uint64_t clk0 = 0;
  uint64_t clk1 = 0;
  uint32_t smid;
  
  asm("mov.u32 %0, %%smid;" : "=r" (smid));
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
            for (evt = (uint64_t *)page2[0]; evt != page2; evt = (uint64_t *)evt[0])
              ++evt[2];
          }
        y = ptr[1];
    }
  }
  
  page0[1] = 0;
  page1[1] = 0;
}

__global__ void
put(uint64_t *page, uint64_t x1, uint64_t x2)
{
  page[0] = x1;
  page[1] = x2;
}

int 
main(int argc, char *argv[])
{
  uint8_t *chunk0 = NULL;
  uint8_t *chunk1 = NULL;
  uint8_t *base = NULL;
  uint64_t *list0[PAGE0_NUM];
  uint64_t *list1[PAGE1_NUM];
  uint64_t *list2[PAGE2_NUM];
  // uint64_t indexes[] = {0xb7, 0x1b6, 0x2b5, 0x3b4, 0x4b3, 0x5b2, 0x6b1, 0x7b0, 0x8bf};
  // uint64_t indexes[] = {0xb7, 0x1b9, 0x2b5, 0x3b4, 0x4b3, 0x5b2, 0x6b1, 0x7b0, 0x8bf};
  uint32_t indexes[32] = {0};

  int aim = -1;
  uint64_t *dummy = NULL;

  cudaDeviceReset();
  cudaFuncSetAttribute(loop, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM);
  
  // hoard a large address space
  cudaMallocManaged(&chunk0, CHUNK0_SIZE);
  cudaMallocManaged(&chunk1, CHUNK1_SIZE);
  
  SMID1 = atoi(argv[1]);
  aim = atoi(argv[2]);

  base = (uint8_t *)BASE_ADDR;
  for (int i = 0; i < PAGE0_NUM; ++i)
    list0[i] = (uint64_t *)(base + i * STRIDE_SIZE);
  // base += PAGE0_NUM * STRIDE_SIZE;
  base = (uint8_t *)0x700200000000;
  for (int i = 0; i < PAGE1_NUM; ++i)
    list1[i] = (uint64_t *)(base + i * STRIDE_SIZE);
  for (int i = 0; i < PAGE2_NUM; ++i)
    list2[i] = list0[aim + i + 1];
  dummy = (uint64_t *)DUMMY_ADDR;
  
  // do dummy first to make its physical address unchanged when changing inputs
  put<<<1, 1>>>(dummy, 0, 0);
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  for (int i = 2; i < argc; ++i) {
    int j = (i + 1) % argc == 0 ? 2 : i + 1;
    int m = atoi(argv[i]);
    int n = atoi(argv[j]);
    put<<<1, 1>>>(list0[m], (uint64_t)list0[n], 0xdeadbeef);
    // printf("[0] m: 0x%lx, n: 0x%lx\n", (uint64_t)list0[m], (uint64_t)list0[n]);
  }

  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // for (int i = 0; i < PAGE1_NUM; ++i)
  //   put<<<1, 1>>>(list1[i], (uint64_t)list1[(i + 1) % PAGE1_NUM], 0xdeadbeef);

  FILE *file = fopen("./eviction_set_smid1.txt", "r");
  if (file == NULL) {
      perror("Error opening file");
      return 1;
  }

    size_t count = 0;         // Number of elements in the array

    // Read numbers from file line by line
    uint32_t number;
    while (fscanf(file, "%u", &number) == 1) {  // Read an unsigned integer
        indexes[count++] = number;
    }
    fclose(file);
    // for (size_t i = 0; i < count; i++) {
    //   printf("%u\n", indexes[i]);
    // }


  for (int i = 0; i < count; ++i) {
    int j = (i + 1) % count == 0 ? 0 : i + 1;
    int m = indexes[i];
    int n = indexes[j];
    put<<<1, 1>>>(list1[m], (uint64_t)list1[n], 0xdeadbeef);
    // printf("[1] m: 0x%lx, n: 0x%lx\n", (uint64_t) list1[m], (uint64_t) list1[n]);
  }

  for (int i = 0; i < PAGE2_NUM; ++i)
    put<<<1, 1>>>(list2[i], (uint64_t)list2[(i + 1) % PAGE2_NUM], 0xdeadbeef);
  cudaDeviceSynchronize();

  printf("Done hoarding, aim: 0x%x, page: 0x%lx, smid: 0x%x\n", aim, (uint64_t)list0[aim], SMID1);
  loop<<<BLK_NUM, 1, SHARED_MEM>>>(list0[aim], list1[indexes[0]], list2[0], 0xdeadbeef, SMID1);
  // loop<<<BLK_NUM, 1, SHARED_MEM>>>(list0[aim], list1[atoi(argv[2])], list2[0], 0xdeadbeef);
  cudaDeviceSynchronize();
  
  cudaFree(chunk0);
  cudaFree(chunk1);
}


