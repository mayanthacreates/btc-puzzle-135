/* Validate the Metal GPU field multiply against CPU field.h. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include "group.h"

static uint64_t st=0x9e3779b97f4a7c15ULL;
static uint64_t xs(void){ uint64_t x=st; x^=x<<13; x^=x>>7; x^=x<<17; st=x; return x; }
static void rand_fe(fe*r){ r->n[0]=xs(); r->n[1]=xs(); r->n[2]=xs(); r->n[3]=xs(); fe_reduce_weak(r); }

/* fe (4x u64) <-> gpu limbs (8x u32) */
static void fe_to_gpu(const fe*a, uint32_t*g){
    for(int i=0;i<4;i++){ g[2*i]=(uint32_t)(a->n[i]&0xffffffff); g[2*i+1]=(uint32_t)(a->n[i]>>32); }
}
static void gpu_to_fe(const uint32_t*g, fe*a){
    for(int i=0;i<4;i++) a->n[i]=((uint64_t)g[2*i]) | (((uint64_t)g[2*i+1])<<32);
}

int main(int argc,char**argv){
 @autoreleasepool {
    int N = (argc>1)?atoi(argv[1]):4096;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if(!dev){ printf("no Metal device\n"); return 1; }
    NSError* err=nil;
    NSString* src=[NSString stringWithContentsOfFile:@"gpu_field.metal" encoding:NSUTF8StringEncoding error:&err];
    if(!src){ printf("read kernel fail: %s\n",err.localizedDescription.UTF8String); return 1; }
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];
    if(!lib){ printf("compile fail: %s\n",err.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> fn=[lib newFunctionWithName:@"test_mul"];
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:fn error:&err];
    if(!pso){ printf("pipeline fail: %s\n",err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q=[dev newCommandQueue];

    size_t bytes=(size_t)N*8*sizeof(uint32_t);
    id<MTLBuffer> bA=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bO=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    uint32_t*A=(uint32_t*)bA.contents,*B=(uint32_t*)bB.contents,*O=(uint32_t*)bO.contents;

    fe*expect=malloc(N*sizeof(fe));
    for(int i=0;i<N;i++){
        fe a,b,e; rand_fe(&a); rand_fe(&b); fe_mul(&e,&a,&b);
        fe_to_gpu(&a,A+i*8); fe_to_gpu(&b,B+i*8); expect[i]=e;
    }

    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bA offset:0 atIndex:0];
    [enc setBuffer:bB offset:0 atIndex:1];
    [enc setBuffer:bO offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    int bad=0;
    for(int i=0;i<N;i++){
        fe g; gpu_to_fe(O+i*8,&g);
        if(!fe_equal(&g,&expect[i])){
            if(bad<3){ char hg[65],he[65]; fe_get_hex(&g,hg); fe_get_hex(&expect[i],he);
                printf("MISMATCH i=%d\n gpu=%s\n cpu=%s\n",i,hg,he); }
            bad++;
        }
    }
    printf("GPU field-mul: tested=%d  mismatches=%d  %s\n",N,bad,bad?"FAIL":"OK");
    free(expect);

    /* ---- fe_sub validation ---- */
    {
        for(int i=0;i<N;i++){ fe a,b; rand_fe(&a); rand_fe(&b); fe_to_gpu(&a,A+i*8); fe_to_gpu(&b,B+i*8); }
        fe*ex=malloc(N*sizeof(fe));
        for(int i=0;i<N;i++){ fe a,b; gpu_to_fe(A+i*8,&a); gpu_to_fe(B+i*8,&b); fe_sub(&ex[i],&a,&b); }
        id<MTLFunction> fs=[lib newFunctionWithName:@"test_sub"];
        id<MTLComputePipelineState> ps=[dev newComputePipelineStateWithFunction:fs error:&err];
        id<MTLCommandBuffer> cbx=[q commandBuffer]; id<MTLComputeCommandEncoder> ex2=[cbx computeCommandEncoder];
        [ex2 setComputePipelineState:ps]; [ex2 setBuffer:bA offset:0 atIndex:0]; [ex2 setBuffer:bB offset:0 atIndex:1]; [ex2 setBuffer:bO offset:0 atIndex:2];
        [ex2 dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [ex2 endEncoding]; [cbx commit]; [cbx waitUntilCompleted];
        int bs=0; for(int i=0;i<N;i++){ fe g; gpu_to_fe(O+i*8,&g); if(!fe_equal(&g,&ex[i])) bs++; }
        printf("GPU fe_sub:    tested=%d  mismatches=%d  %s\n",N,bs,bs?"FAIL":"OK");
        free(ex);
    }
    /* ---- in-place fe_mul validation ---- */
    {
        for(int i=0;i<N;i++){ fe a,b; rand_fe(&a); rand_fe(&b); fe_to_gpu(&a,A+i*8); fe_to_gpu(&b,B+i*8); }
        fe*ex=malloc(N*sizeof(fe));
        for(int i=0;i<N;i++){ fe a,b; gpu_to_fe(A+i*8,&a); gpu_to_fe(B+i*8,&b); fe_mul(&ex[i],&a,&b); }
        id<MTLFunction> f=[lib newFunctionWithName:@"test_ipmul"];
        id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:f error:&err];
        id<MTLCommandBuffer> cbx=[q commandBuffer]; id<MTLComputeCommandEncoder> e2=[cbx computeCommandEncoder];
        [e2 setComputePipelineState:p]; [e2 setBuffer:bA offset:0 atIndex:0]; [e2 setBuffer:bB offset:0 atIndex:1]; [e2 setBuffer:bO offset:0 atIndex:2];
        [e2 dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [e2 endEncoding]; [cbx commit]; [cbx waitUntilCompleted];
        int bz=0; for(int i=0;i<N;i++){ fe g; gpu_to_fe(O+i*8,&g); if(!fe_equal(&g,&ex[i])) bz++; }
        printf("GPU ip-mul:    tested=%d  mismatches=%d  %s\n",N,bz,bz?"FAIL":"OK");
        free(ex);
    }
    /* ---- fe_inv validation ---- */
    {
        fe*ex=malloc(N*sizeof(fe));
        for(int i=0;i<N;i++){ fe a; rand_fe(&a); if(fe_is_zero(&a)) a.n[0]=1; fe_to_gpu(&a,A+i*8); fe_inv_fast(&ex[i],&a); }
        id<MTLFunction> fi=[lib newFunctionWithName:@"test_inv"];
        id<MTLComputePipelineState> pi=[dev newComputePipelineStateWithFunction:fi error:&err];
        id<MTLCommandBuffer> cbx=[q commandBuffer]; id<MTLComputeCommandEncoder> ei=[cbx computeCommandEncoder];
        [ei setComputePipelineState:pi]; [ei setBuffer:bA offset:0 atIndex:0]; [ei setBuffer:bO offset:0 atIndex:1];
        [ei dispatchThreads:MTLSizeMake(N,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [ei endEncoding]; [cbx commit]; [cbx waitUntilCompleted];
        int bi=0; for(int i=0;i<N;i++){ fe g; gpu_to_fe(O+i*8,&g); if(!fe_equal(&g,&ex[i])){ if(bi<2){char x[65],y[65];fe_get_hex(&g,x);fe_get_hex(&ex[i],y);printf(" inv gpu=%s\n inv cpu=%s\n",x,y);} bi++; } }
        printf("GPU fe_inv:    tested=%d  mismatches=%d  %s\n",N,bi,bi?"FAIL":"OK");
        free(ex);
    }

    /* ---- EC point-add validation ---- */
    id<MTLFunction> fn2=[lib newFunctionWithName:@"test_ecadd"];
    id<MTLComputePipelineState> pso2=[dev newComputePipelineStateWithFunction:fn2 error:&err];
    if(!pso2){ printf("ec pipeline fail: %s\n",err.localizedDescription.UTF8String); return 1; }
    int M=2048;
    size_t pbytes=(size_t)M*16*sizeof(uint32_t);
    id<MTLBuffer> bP=[dev newBufferWithLength:pbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bQ=[dev newBufferWithLength:pbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bR=[dev newBufferWithLength:pbytes options:MTLResourceStorageModeShared];
    uint32_t*P=(uint32_t*)bP.contents,*Q=(uint32_t*)bQ.contents,*R=(uint32_t*)bR.contents;
    ge*eR=malloc(M*sizeof(ge));
    for(int i=0;i<M;i++){
        sc ka,kb; ka.n[0]=xs();ka.n[1]=xs()&0xffff;ka.n[2]=0;ka.n[3]=0;
        kb.n[0]=xs();kb.n[1]=xs()&0xffff;kb.n[2]=0;kb.n[3]=0;
        ge A2,B2,S; ge_scalar_base(&A2,&ka); ge_scalar_base(&B2,&kb);
        if(fe_equal(&A2.x,&B2.x)){ kb.n[0]^=1; ge_scalar_base(&B2,&kb); }
        ge_add(&S,&A2,&B2);
        fe_to_gpu(&A2.x,P+i*16); fe_to_gpu(&A2.y,P+i*16+8);
        fe_to_gpu(&B2.x,Q+i*16); fe_to_gpu(&B2.y,Q+i*16+8);
        eR[i]=S;
    }
    id<MTLCommandBuffer> cb2=[q commandBuffer];
    id<MTLComputeCommandEncoder> enc2=[cb2 computeCommandEncoder];
    [enc2 setComputePipelineState:pso2];
    [enc2 setBuffer:bP offset:0 atIndex:0];
    [enc2 setBuffer:bQ offset:0 atIndex:1];
    [enc2 setBuffer:bR offset:0 atIndex:2];
    [enc2 dispatchThreads:MTLSizeMake(M,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc2 endEncoding]; [cb2 commit]; [cb2 waitUntilCompleted];
    int bad2=0;
    for(int i=0;i<M;i++){
        fe gx,gy; gpu_to_fe(R+i*16,&gx); gpu_to_fe(R+i*16+8,&gy);
        if(!fe_equal(&gx,&eR[i].x)||!fe_equal(&gy,&eR[i].y)){
            if(bad2<3){ char a[65],b[65]; fe_get_hex(&gx,a); fe_get_hex(&eR[i].x,b);
                printf("EC MISMATCH i=%d\n gpu.x=%s\n cpu.x=%s\n",i,a,b); }
            bad2++;
        }
    }
    printf("GPU ec-add:    tested=%d  mismatches=%d  %s\n",M,bad2,bad2?"FAIL":"OK");
    free(eR);
    return (bad||bad2)?1:0;
 }
}
