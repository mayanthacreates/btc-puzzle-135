#include <metal_stdlib>
using namespace metal;

/* secp256k1 field Fp on GPU, 8 x 32-bit limbs, little-endian.
 * p = 2^256 - 2^32 - 977 ;  2^256 = c (mod p), c = 2^32 + 977 = 0x1000003D1.
 * No 128-bit type on Metal: all intermediates fit in ulong (64-bit). */

constant uint P32[8] = {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,
                        0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};

static inline bool ge_p(thread const uint a[8]){
    for(int i=7;i>=0;i--){ if(a[i]!=P32[i]) return a[i]>P32[i]; }
    return true;
}
static inline void sub_p(thread uint a[8]){
    ulong borrow=0;
    for(int i=0;i<8;i++){
        ulong cur=(ulong)a[i]-(ulong)P32[i]-borrow;
        a[i]=(uint)cur;
        borrow=(cur>>32)&1u;
    }
}

/* 256x256 -> 512 (16 limbs) schoolbook */
static inline void mul256(thread const uint a[8], thread const uint b[8], thread uint out[16]){
    uint r[16];
    for(int i=0;i<16;i++) r[i]=0u;
    for(int i=0;i<8;i++){
        ulong carry=0;
        for(int j=0;j<8;j++){
            ulong cur=(ulong)r[i+j] + (ulong)a[i]*(ulong)b[j] + carry;
            r[i+j]=(uint)cur;
            carry=cur>>32;
        }
        r[i+8]=(uint)carry;
    }
    for(int i=0;i<16;i++) out[i]=r[i];
}

/* reduce 512-bit (16 limbs) mod p -> 8 limbs */
static inline void reduce(thread uint t[16], thread uint r[8]){
    uint acc[10];
    for(int i=0;i<8;i++) acc[i]=t[i];
    acc[8]=0u; acc[9]=0u;

    /* + H*977 at offset 0  (H = t[8..15]) */
    ulong carry=0;
    for(int k=0;k<8;k++){ ulong cur=(ulong)acc[k]+(ulong)t[8+k]*977UL+carry; acc[k]=(uint)cur; carry=cur>>32; }
    acc[8]+=(uint)carry;

    /* + H<<32 at offset 1 */
    carry=0;
    for(int k=0;k<8;k++){ ulong cur=(ulong)acc[1+k]+(ulong)t[8+k]+carry; acc[1+k]=(uint)cur; carry=cur>>32; }
    acc[9]+=(uint)carry;

    /* fold acc[8..9] (value above 2^256) back via *c */
    ulong high=((ulong)acc[8])|(((ulong)acc[9])<<32);
    /* high*977 -> limbs w0,w1,w2 */
    ulong hlo=high&0xFFFFFFFFUL, hhi=high>>32;
    ulong q0=hlo*977UL, q1=hhi*977UL;
    ulong w0=q0&0xFFFFFFFFUL;
    ulong w1=(q0>>32)+(q1&0xFFFFFFFFUL);
    ulong w2=(q1>>32)+(w1>>32); w1&=0xFFFFFFFFUL;
    uint addv[8]={(uint)w0,(uint)w1,(uint)w2,0u,0u,0u,0u,0u};
    /* + high<<32 (high at limb offset 1) */
    { ulong cc=0;
      ulong c1=(ulong)addv[1]+(high&0xFFFFFFFFUL)+cc; addv[1]=(uint)c1; cc=c1>>32;
      ulong c2=(ulong)addv[2]+(high>>32)+cc;          addv[2]=(uint)c2; cc=c2>>32;
      ulong c3=(ulong)addv[3]+cc;                     addv[3]=(uint)c3; }

    carry=0;
    for(int k=0;k<8;k++){ ulong cur=(ulong)acc[k]+addv[k]+carry; acc[k]=(uint)cur; carry=cur>>32; }
    if(carry){ ulong e=carry*0x1000003D1UL; ulong cc=0;
        ulong c0=(ulong)acc[0]+(e&0xFFFFFFFFUL)+cc; acc[0]=(uint)c0; cc=c0>>32;
        ulong c1=(ulong)acc[1]+(e>>32)+cc;          acc[1]=(uint)c1; cc=c1>>32;
        for(int k=2;k<8;k++){ ulong cur=(ulong)acc[k]+cc; acc[k]=(uint)cur; cc=cur>>32; }
    }
    for(int i=0;i<8;i++) r[i]=acc[i];
    if(ge_p(r)) sub_p(r);
    if(ge_p(r)) sub_p(r);
}

static inline void fe_mul(thread const uint a[8], thread const uint b[8], thread uint r[8]){
    uint t[16]; mul256(a,b,t); reduce(t,r);
}
static inline void fe_sqr(thread const uint a[8], thread uint r[8]){ fe_mul(a,a,r); }

static inline void fe_add(thread const uint a[8], thread const uint b[8], thread uint r[8]){
    ulong carry=0;
    for(int i=0;i<8;i++){ ulong cur=(ulong)a[i]+(ulong)b[i]+carry; r[i]=(uint)cur; carry=cur>>32; }
    if(carry){ ulong e=0x1000003D1UL, cc=0;
        ulong c0=(ulong)r[0]+(e&0xFFFFFFFFUL)+cc; r[0]=(uint)c0; cc=c0>>32;
        ulong c1=(ulong)r[1]+(e>>32)+cc;          r[1]=(uint)c1; cc=c1>>32;
        for(int k=2;k<8;k++){ ulong cur=(ulong)r[k]+cc; r[k]=(uint)cur; cc=cur>>32; }
    }
    if(ge_p(r)) sub_p(r);
}
static inline void fe_sub(thread const uint a[8], thread const uint b[8], thread uint r[8]){
    ulong borrow=0; uint t[8];
    for(int i=0;i<8;i++){ ulong cur=(ulong)a[i]-(ulong)b[i]-borrow; t[i]=(uint)cur; borrow=(cur>>32)&1u; }
    if(borrow){ ulong c=0; for(int i=0;i<8;i++){ ulong cur=(ulong)t[i]+(ulong)P32[i]+c; t[i]=(uint)cur; c=cur>>32; } }
    for(int i=0;i<8;i++) r[i]=t[i];
}
static inline void fe_sqrn(thread const uint a[8], int n, thread uint r[8]){
    uint t[8]; for(int i=0;i<8;i++) t[i]=a[i];
    for(int i=0;i<n;i++) fe_sqr(t,t);
    for(int i=0;i<8;i++) r[i]=t[i];
}
/* inverse via Fermat a^(p-2). Small footprint (2 working arrays). */
constant uint PM2[8]={0xFFFFFC2Du,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,
                      0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
static inline void fe_inv(thread const uint a[8], thread uint r[8]){
    uint result[8], base[8];
    for(int i=0;i<8;i++){ result[i]=0u; base[i]=a[i]; }
    result[0]=1u;
    for(int i=0;i<256;i++){
        if((PM2[i>>5]>>(i&31))&1u) fe_mul(result,base,result);
        fe_sqr(base,base);
    }
    for(int i=0;i<8;i++) r[i]=result[i];
}
/* affine point add, P=(x1,y1) Q=(x2,y2), x1!=x2. out=(x3,y3) */
static inline void ec_add(thread const uint x1[8], thread const uint y1[8],
                          thread const uint x2[8], thread const uint y2[8],
                          thread uint x3[8], thread uint y3[8]){
    uint num[8],den[8],inv[8],lam[8],lam2[8],t[8];
    fe_sub(y2,y1,num);
    fe_sub(x2,x1,den);
    fe_inv(den,inv);
    fe_mul(num,inv,lam);
    fe_sqr(lam,lam2);
    fe_sub(lam2,x1,t); fe_sub(t,x2,x3);
    fe_sub(x1,x3,t);   fe_mul(lam,t,y3); fe_sub(y3,y1,y3);
}

kernel void test_ipmul(device const uint* A [[buffer(0)]],
                       device const uint* B [[buffer(1)]],
                       device uint*       O [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]){
    uint a[8],b[8]; for(int i=0;i<8;i++){ a[i]=A[gid*8+i]; b[i]=B[gid*8+i]; }
    fe_mul(a,b,a);            /* in-place: output aliases first input */
    for(int i=0;i<8;i++) O[gid*8+i]=a[i];
}
kernel void test_inv(device const uint* A [[buffer(0)]],
                     device uint*       O [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]){
    uint a[8],r[8]; for(int i=0;i<8;i++) a[i]=A[gid*8+i];
    fe_inv(a,r);
    for(int i=0;i<8;i++) O[gid*8+i]=r[i];
}
kernel void test_sub(device const uint* A [[buffer(0)]],
                     device const uint* B [[buffer(1)]],
                     device uint*       O [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]){
    uint a[8],b[8],r[8]; for(int i=0;i<8;i++){ a[i]=A[gid*8+i]; b[i]=B[gid*8+i]; }
    fe_sub(a,b,r);
    for(int i=0;i<8;i++) O[gid*8+i]=r[i];
}

/* one thread per pair: R = P + Q (affine). Layout: 16 u32 each (x8,y8). */
kernel void test_ecadd(device const uint* P [[buffer(0)]],
                       device const uint* Q [[buffer(1)]],
                       device uint*       R [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]){
    uint x1[8],y1[8],x2[8],y2[8],x3[8],y3[8];
    for(int i=0;i<8;i++){ x1[i]=P[gid*16+i]; y1[i]=P[gid*16+8+i];
                          x2[i]=Q[gid*16+i]; y2[i]=Q[gid*16+8+i]; }
    ec_add(x1,y1,x2,y2,x3,y3);
    for(int i=0;i<8;i++){ R[gid*16+i]=x3[i]; R[gid*16+8+i]=y3[i]; }
}

/* one thread per (a,b) pair: O = A*B mod p */
kernel void test_mul(device const uint* A [[buffer(0)]],
                     device const uint* B [[buffer(1)]],
                     device uint*       O [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]){
    uint a[8],b[8],t[16],r[8];
    for(int i=0;i<8;i++){ a[i]=A[gid*8+i]; b[i]=B[gid*8+i]; }
    mul256(a,b,t);
    reduce(t,r);
    for(int i=0;i<8;i++) O[gid*8+i]=r[i];
}
