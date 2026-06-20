/* GPU engine for the fused bot. Runs the Metal kangaroo, seeds its own herds
 * using the SAME jump table + shifted-problem convention as the CPU (via the
 * bridge), and feeds every distinguished point into the shared CPU DP table.
 * So a GPU tame can collide with a CPU wild and vice-versa. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include "group.h"
#include "gpu_bridge.h"

#define KB 128
typedef struct { uint32_t njump,dpmask,steps,dpcap; } KParams;

static void gpu_to_fe(const uint32_t*g,fe*a){ for(int i=0;i<4;i++) a->n[i]=((uint64_t)g[2*i])|(((uint64_t)g[2*i+1])<<32); }
static void fe_to_gpu(const fe*a,uint32_t*g){ for(int i=0;i<4;i++){ g[2*i]=(uint32_t)a->n[i]; g[2*i+1]=(uint32_t)(a->n[i]>>32);} }
static void l_sc_add(sc*r,const sc*a,const sc*b){ __uint128_t c=0;
    for(int i=0;i<4;i++){ __uint128_t t=(__uint128_t)a->n[i]+b->n[i]+c; r->n[i]=(uint64_t)t; c=t>>64; } }

void* gpu_thread(void* arg){
 @autoreleasepool {
    (void)arg;
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    if(!dev){ fprintf(stderr,"[gpu] no Metal device; running CPU-only\n"); return NULL; }
    NSError*err=nil;
    NSString*src=[NSString stringWithContentsOfFile:@"gpu_field.metal" encoding:NSUTF8StringEncoding error:&err];
    if(!src){ fprintf(stderr,"[gpu] gpu_field.metal not found; CPU-only\n"); return NULL; }
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];
    if(!lib){ fprintf(stderr,"[gpu] kernel compile failed; CPU-only\n"); return NULL; }
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kang_batch"] error:&err];
    if(!pso){ fprintf(stderr,"[gpu] pipeline failed; CPU-only\n"); return NULL; }
    id<MTLCommandQueue> q=[dev newCommandQueue];

    int wbits=bridge_wbits(), NJ=bridge_njump(), dpbits=bridge_dpbits();

    /* adaptive herd size: ~2^(wbits/2 - 4), clamped [2^14, 2^19] so small
     * puzzles don't flood and large ones don't take forever to seed */
    int sh=wbits/2-4; if(sh>19)sh=19; if(sh<14)sh=14;
    long target=1L<<sh;
    int threads=(int)(target/KB); if(threads<8) threads=8;
    { const char*e=getenv("GPUTHREADS"); if(e) threads=atoi(e); }
    int N=threads*KB, half=N/2;
    uint32_t steps=256; uint32_t dpcap=1u<<22;

    ge G; ge_generator(&G);

    /* jump table (identical to CPU's, via bridge) */
    id<MTLBuffer> bJX=[dev newBufferWithLength:NJ*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bJY=[dev newBufferWithLength:NJ*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bJD=[dev newBufferWithLength:NJ*8*4 options:MTLResourceStorageModeShared];
    uint32_t*JX=(uint32_t*)bJX.contents,*JY=(uint32_t*)bJY.contents,*JD=(uint32_t*)bJD.contents;
    for(int i=0;i<NJ;i++) bridge_jump(i,JX+i*8,JY+i*8,JD+i*8);

    /* Qshift from bridge */
    ge Qs; { uint32_t qx[8],qy[8]; bridge_qshift(qx,qy); gpu_to_fe(qx,&Qs.x); gpu_to_fe(qy,&Qs.y); Qs.inf=0; }

    /* kangaroo state + seed (incremental adds, herds spread across [0,W)) */
    id<MTLBuffer> bKX=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bKY=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bKD=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bKT=[dev newBufferWithLength:(size_t)N options:MTLResourceStorageModeShared];
    uint32_t*KX=(uint32_t*)bKX.contents,*KY=(uint32_t*)bKY.contents,*KD=(uint32_t*)bKD.contents;
    uint8_t*KT=(uint8_t*)bKT.contents;
    {
      sc stride; stride.n[0]=stride.n[1]=stride.n[2]=stride.n[3]=0;
      int sb=(wbits>=15)?(wbits-15):1; stride.n[sb/64]|=(1ULL<<(sb%64)); if(stride.n[0]==0&&sb<64)stride.n[0]=1;
      ge stepG; ge_scalar_mul(&stepG,&stride,&G);
      sc d; d.n[0]=d.n[1]=d.n[2]=d.n[3]=0;
      ge cur; ge_set_infinity(&cur);
      for(int i=0;i<half;i++){
        ge tp2=cur; if(tp2.inf) tp2=stepG;
        fe_to_gpu(&tp2.x,KX+i*8); fe_to_gpu(&tp2.y,KY+i*8); fe_to_gpu((fe*)&d,KD+i*8); KT[i]=0;
        ge wp; if(cur.inf) wp=Qs; else ge_add(&wp,&Qs,&cur);
        int wi=half+i;
        fe_to_gpu(&wp.x,KX+wi*8); fe_to_gpu(&wp.y,KY+wi*8); fe_to_gpu((fe*)&d,KD+wi*8); KT[wi]=1;
        if(cur.inf) cur=stepG; else ge_add(&cur,&cur,&stepG);
        l_sc_add(&d,&d,&stride);
        if((i&1023)==0 && bridge_solved()) return NULL;   /* abort seeding if CPU already solved */
      }
    }

    id<MTLBuffer> bSDEN=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bSPRE=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bDPx=[dev newBufferWithLength:(size_t)dpcap*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bDPd=[dev newBufferWithLength:(size_t)dpcap*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bDPt=[dev newBufferWithLength:(size_t)dpcap options:MTLResourceStorageModeShared];
    id<MTLBuffer> bDPc=[dev newBufferWithLength:4 options:MTLResourceStorageModeShared];
    uint32_t*DPx=(uint32_t*)bDPx.contents,*DPd=(uint32_t*)bDPd.contents; uint8_t*DPt=(uint8_t*)bDPt.contents;
    uint32_t*DPc=(uint32_t*)bDPc.contents;

    KParams P={ (uint32_t)NJ,(uint32_t)((1u<<dpbits)-1),steps,dpcap };
    id<MTLBuffer> bP=[dev newBufferWithBytes:&P length:sizeof(P) options:MTLResourceStorageModeShared];
    bridge_set_kangaroos(N);

    while(!bridge_solved()){
        *DPc=0;
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        id<MTLBuffer> bufs[]={bKX,bKY,bKD,bKT,bJX,bJY,bJD,bSDEN,bSPRE,bDPx,bDPd,bDPt,bDPc,bP};
        for(int i=0;i<14;i++) [e setBuffer:bufs[i] offset:0 atIndex:i];
        [e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        bridge_add_jumps((uint64_t)N*steps);
        uint32_t nd=*DPc; if(nd>dpcap)nd=dpcap;
        for(uint32_t j=0;j<nd;j++){
            if(bridge_feed_dp(DPx+j*8,DPd+j*8,DPt[j])) break;
        }
    }
    return NULL;
 }
}
