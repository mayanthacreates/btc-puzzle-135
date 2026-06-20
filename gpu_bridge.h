#ifndef GPU_BRIDGE_H
#define GPU_BRIDGE_H
#include <stdint.h>

/* implemented in kangaroo.c, called by gpu_driver.m */
int  bridge_njump(void);
void bridge_jump(int i, uint32_t* jx, uint32_t* jy, uint32_t* jd);
void bridge_target(uint32_t* tx, uint32_t* ty);
void bridge_qshift(uint32_t* qx, uint32_t* qy);
void bridge_rangeL(uint32_t* l);
int  bridge_dpbits(void);
int  bridge_wbits(void);
int  bridge_solved(void);
void bridge_add_jumps(uint64_t n);
void bridge_set_kangaroos(int n);
int  bridge_feed_dp(const uint32_t* x8, const uint32_t* d8, uint8_t type);

/* implemented in gpu_driver.m, called by kangaroo.c */
void* gpu_thread(void* arg);

#endif
