/* Standalone GPU kangaroo solver - validates the Metal kangaroo end-to-end.
 * Usage: ./gpu_solve <bits>           (selftest: random key in [2^(bits-1),2^bits))
 *        ./gpu_solve <pub> <L> <R>    (real target) */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <time.h>
#include "group.h"

/* ---- sc helpers ---- */
static void sc_add(sc*r,const sc*a,const sc*b){ __uint128_t c=0;
    for(int i=0;i<4;i++){ __uint128_t t=(__uint128_t)a->n[i]+b->n[i]+c; r->n[i]=(uint64_t)t; c=t>>64; } }
static void sc_sub(sc*r,const sc*a,const sc*b){ __uint128_t br=0;
    for(int i=0;i<4;i++){ __uint128_t d=(__uint128_t)a->n[i]-b->n[i]-br; r->n[i]=(uint64_t)d; br=(d>>64)&1; } }

/* ---- fe/sc <-> gpu (8x u32) ---- */
static void fe_to_gpu(const fe*a,uint32_t*g){ for(int i=0;i<4;i++){ g[2*i]=(uint32_t)a->n[i]; g[2*i+1]=(uint32_t)(a->n[i]>>32); } }
static void gpu_to_sc(const uint32_t*g,sc*a){ for(int i=0;i<4;i++) a->n[i]=((uint64_t)g[2*i])|(((uint64_t)g[2*i+1])<<32); }
static void sc_to_gpu(const sc*a,uint32_t*g){ for(int i=0;i<4;i++){ g[2*i]=(uint32_t)a->n[i]; g[2*i+1]=(uint32_t)(a->n[i]>>32); } }

#define KB 128
typedef struct { uint32_t njump,dpmask,steps,dpcap; } KParams;

/* ---- minimal host DP table ---- */
typedef struct { uint8_t *used; uint32_t *xk; uint32_t *dk; uint8_t *tp; uint64_t slots,mask; } DPT;
static DPT T;
static void dpt_init(int log2){ T.slots=1ULL<<log2; T.mask=T.slots-1;
    T.used=calloc(T.slots,1); T.xk=calloc(T.slots*8,4); T.dk=calloc(T.slots*8,4); T.tp=calloc(T.slots,1); }
/* returns 1 + fills tame_d,wild_d on a tame/wild collision */
static int dpt_insert(const uint32_t*x,const uint32_t*d,uint8_t type,sc*tame_d,sc*wild_d){
    uint64_t h=((uint64_t)x[0]*0x9E3779B97F4A7C15ULL)^((uint64_t)x[4]*0xC2B2AE3D27D4EB4FULL);
    uint64_t i=h&T.mask;
    for(uint64_t p=0;p<T.slots;p++){
        uint64_t s=(i+p)&T.mask;
        if(!T.used[s]){ T.used[s]=1; for(int k=0;k<8;k++){ T.xk[s*8+k]=x[k]; T.dk[s*8+k]=d[k]; } T.tp[s]=type; return 0; }
        int same=1; for(int k=0;k<8;k++) if(T.xk[s*8+k]!=x[k]){ same=0; break; }
        if(same){
            if(T.tp[s]!=type){
                if(type==0){ gpu_to_sc(d,tame_d); gpu_to_sc(&T.dk[s*8],wild_d); }
                else        { gpu_to_sc(&T.dk[s*8],tame_d); gpu_to_sc(d,wild_d); }
                return 1;
            }
            return 0;
        }
    }
    return 0;
}

int main(int argc,char**argv){
 @autoreleasepool {
    if(argc<2){ printf("usage: gpu_solve <bits> | gpu_solve <pub> <L> <R>\n"); return 1; }

    ge target; sc L; int wbits; sc realkey; int have_real=0;
    if(argc==2){
        int bits=atoi(argv[1]);                 /* selftest supports bits <= 64 */
        uint64_t s=0xCAFEBABEDEADBEEFULL^(uint64_t)time(NULL);
        s^=s<<13;s^=s>>7;s^=s<<17;
        sc key; key.n[0]=s; key.n[1]=0; key.n[2]=0; key.n[3]=0;
        key.n[0] &= (bits>=64)?~0ULL:((1ULL<<bits)-1);
        key.n[0] |= (1ULL<<(bits-1));           /* ensure top bit set */
        ge_scalar_base(&target,&key);
        L.n[0]=L.n[1]=L.n[2]=L.n[3]=0; L.n[(bits-1)/64]|=(1ULL<<((bits-1)%64));
        wbits=bits-1; realkey=key; have_real=1;
        char kh[65]; fe t; t.n[0]=key.n[0];t.n[1]=key.n[1];t.n[2]=key.n[2];t.n[3]=key.n[3]; fe_get_hex(&t,kh);
        printf("[gpu selftest] bits=%d secret=%s\n",bits,kh);
    } else {
        if(!ge_decompress(&target,argv[1])){ printf("bad pubkey\n"); return 1; }
        sc R; sc_set_hex(&L,argv[2]); sc_set_hex(&R,argv[3]);
        sc W; sc_sub(&W,&R,&L); wbits=0; for(int i=3;i>=0;i--) if(W.n[i]){ int b=63; while(!((W.n[i]>>b)&1))b--; wbits=i*64+b+1; break; }
    }

    /* Qshift = target - L*G */
    ge Qs; { ge G; ge_generator(&G); ge LG; ge_scalar_mul(&LG,&L,&G); ge nLG; ge_neg(&nLG,&LG); ge_add(&Qs,&target,&nLG); }

    /* ---- Metal setup ---- */
    id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
    NSError*err=nil;
    NSString*src=[NSString stringWithContentsOfFile:@"gpu_field.metal" encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&err];
    if(!lib){ printf("compile: %s\n",err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"kang_batch"] error:&err];
    if(!pso){ printf("pipeline: %s\n",err.localizedDescription.UTF8String); return 1; }
    fprintf(stderr,"[gpu] KB=%d maxThreadsPerThreadgroup=%lu execWidth=%lu\n",
            KB,(unsigned long)pso.maxTotalThreadsPerThreadgroup,(unsigned long)pso.threadExecutionWidth);
    id<MTLCommandQueue> q=[dev newCommandQueue];

    int NJ=512;
    int threads = (argc==2)? (1<<9) : (1<<13);   /* GPU threads; each owns KB kangaroos */
    { const char*e=getenv("THREADS"); if(e) threads=atoi(e); }
    int N = threads*KB;                            /* total kangaroos */
    int half=N/2;
    uint32_t steps=256; { const char*e=getenv("STEPS"); if(e) steps=atoi(e); }
    int lgN=0; { int t=N; while(t>1){lgN++;t>>=1;} }
    int dpbits = wbits/2 - lgN - 2; if(dpbits<1)dpbits=1; if(dpbits>24)dpbits=24;
    { const char*e=getenv("DPBITS"); if(e) dpbits=atoi(e); }
    uint32_t dpcap = 1u<<23;

    /* ---- jump table on CPU ---- */
    ge G; ge_generator(&G);
    id<MTLBuffer> bJX=[dev newBufferWithLength:NJ*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bJY=[dev newBufferWithLength:NJ*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bJD=[dev newBufferWithLength:NJ*8*4 options:MTLResourceStorageModeShared];
    uint32_t*JX=(uint32_t*)bJX.contents,*JY=(uint32_t*)bJY.contents,*JD=(uint32_t*)bJD.contents;
    { uint64_t s=0x243F6A8885A308D3ULL; int jb=wbits/2+1;
      for(int i=0;i<NJ;i++){ sc dj; dj.n[0]=dj.n[1]=dj.n[2]=dj.n[3]=0;
        int full=jb/64,rem=jb%64;
        for(int k=0;k<full;k++){ s^=s<<13;s^=s>>7;s^=s<<17; dj.n[k]=s; }
        if(rem){ s^=s<<13;s^=s>>7;s^=s<<17; dj.n[full]=s&((1ULL<<rem)-1); }
        if(!dj.n[0]&&!dj.n[1]&&!dj.n[2]&&!dj.n[3]) dj.n[0]=1;
        ge J; ge_scalar_mul(&J,&dj,&G);
        fe_to_gpu(&J.x,JX+i*8); fe_to_gpu(&J.y,JY+i*8); sc_to_gpu(&dj,JD+i*8);
      }
    }

    /* ---- seed kangaroos (incremental adds for speed) ---- */
    id<MTLBuffer> bKX=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bKY=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bKD=[dev newBufferWithLength:(size_t)N*8*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bKT=[dev newBufferWithLength:(size_t)N options:MTLResourceStorageModeShared];
    uint32_t*KX=(uint32_t*)bKX.contents,*KY=(uint32_t*)bKY.contents,*KD=(uint32_t*)bKD.contents;
    uint8_t*KT=(uint8_t*)bKT.contents;
    { /* stride so herds spread across [0,W) */
      sc stride; stride.n[0]=stride.n[1]=stride.n[2]=stride.n[3]=0;
      int sb=(wbits>=15)?(wbits-15):1; stride.n[sb/64]|=(1ULL<<(sb%64)); if(stride.n[0]==0&&sb<64)stride.n[0]=1;
      ge stepG; ge_scalar_mul(&stepG,&stride,&G);
      /* tame: point=(d)*G, d = i*stride ; wild: point=Qs+(d)*G */
      sc d; d.n[0]=d.n[1]=d.n[2]=d.n[3]=0;
      ge cur; ge_set_infinity(&cur);          /* 0*G */
      for(int i=0;i<half;i++){
        ge pt = cur; if(pt.inf){ pt=cur; }
        /* tame point = cur (= d*G); if inf use a tiny offset */
        ge tp2 = pt;
        if(tp2.inf){ tp2=stepG; }
        fe_to_gpu(&tp2.x,KX+i*8); fe_to_gpu(&tp2.y,KY+i*8); sc_to_gpu(&d,KD+i*8); KT[i]=0;
        /* wild point = Qs + d*G */
        ge wp; if(pt.inf) wp=Qs; else ge_add(&wp,&Qs,&pt);
        int wi=half+i;
        fe_to_gpu(&wp.x,KX+wi*8); fe_to_gpu(&wp.y,KY+wi*8); sc_to_gpu(&d,KD+wi*8); KT[wi]=1;
        /* advance */
        if(cur.inf) cur=stepG; else ge_add(&cur,&cur,&stepG);
        sc_add(&d,&d,&stride);
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

    dpt_init(23);
    KParams P={ (uint32_t)NJ,(uint32_t)((1u<<dpbits)-1),steps,dpcap };
    id<MTLBuffer> bP=[dev newBufferWithBytes:&P length:sizeof(P) options:MTLResourceStorageModeShared];

    printf("[gpu] threads=%d steps/launch=%d dpbits=%d NJ=%d jumpmean~2^%d\n",N,steps,dpbits,NJ,wbits/2);
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    uint64_t totaljumps=0; int solved=0; sc answer;

    for(int launch=0; launch<100000 && !solved; launch++){
        *DPc=0;
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> e=[cb computeCommandEncoder];
        [e setComputePipelineState:pso];
        id<MTLBuffer> bufs[]={bKX,bKY,bKD,bKT,bJX,bJY,bJD,bSDEN,bSPRE,bDPx,bDPd,bDPt,bDPc,bP};
        for(int i=0;i<14;i++) [e setBuffer:bufs[i] offset:0 atIndex:i];
        int tg=64; { const char*ev=getenv("TGSIZE"); if(ev) tg=atoi(ev); }
        [e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        totaljumps += (uint64_t)N*steps;

        uint32_t nd=*DPc; if(nd>dpcap)nd=dpcap;
        for(uint32_t j=0;j<nd && !solved;j++){
            sc td,wd;
            if(dpt_insert(DPx+j*8,DPd+j*8,DPt[j],&td,&wd)){
                sc q2; sc_sub(&q2,&td,&wd); sc k; sc_add(&k,&L,&q2);
                ge Tt; ge_scalar_base(&Tt,&k);
                if(fe_equal(&Tt.x,&target.x)&&fe_equal(&Tt.y,&target.y)){ solved=1; answer=k; }
            }
        }
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
        double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        fprintf(stderr,"\r[gpu %.1fs] jumps=%.3e rate=%.1f Mj/s DPs=%u   ",el,(double)totaljumps,totaljumps/el/1e6,nd);
    }
    fprintf(stderr,"\n");
    if(solved){ fe a; a.n[0]=answer.n[0];a.n[1]=answer.n[1];a.n[2]=answer.n[2];a.n[3]=answer.n[3];
        char kh[65]; fe_get_hex(&a,kh); printf("GPU SOLVED key = %s\n",kh);
        if(have_real){ fe r; r.n[0]=realkey.n[0];r.n[1]=realkey.n[1];r.n[2]=realkey.n[2];r.n[3]=realkey.n[3];
            char rh[65]; fe_get_hex(&r,rh); printf("expected       = %s\n%s\n",rh,strcmp(kh,rh)?"MISMATCH":"MATCH"); } }
    else printf("not solved (ran out of launches)\n");
    return solved?0:1;
 }
}
