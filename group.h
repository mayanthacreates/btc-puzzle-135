/* secp256k1 group operations (affine), built on field.h.
 * Generic ec_add/ec_double use a full inversion each (setup only).
 * Hot-loop batched addition lives in the kangaroo engine. */
#ifndef GROUP_H
#define GROUP_H

#include "field.h"

typedef struct { fe x, y; int inf; } ge;

/* generator G */
static const char *G_X = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
static const char *G_Y = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

static inline void ge_set_infinity(ge *r){ r->inf = 1; fe_set_u64(&r->x,0); fe_set_u64(&r->y,0); }
static inline void ge_generator(ge *r){ r->inf=0; fe_set_hex(&r->x,G_X); fe_set_hex(&r->y,G_Y); }

/* b coefficient is 7; curve y^2 = x^3 + 7 */
static inline void fe_pow(fe *r, const fe *a, const uint64_t e[4]){
    fe result; fe_set_u64(&result,1); fe base=*a;
    for(int i=0;i<4;i++){ uint64_t w=e[i];
        for(int bit=0;bit<64;bit++){ if(w&1ULL) fe_mul(&result,&result,&base);
            fe_sqr(&base,&base); w>>=1; } }
    *r=result;
}

static inline void ge_double(ge *r, const ge *a){
    if(a->inf || fe_is_zero(&a->y)){ ge_set_infinity(r); return; }
    fe x2,num,den,lam,t1,t2,x3,y3,three;
    fe_sqr(&x2,&a->x);                 /* x^2 */
    fe_set_u64(&three,3);
    fe_mul(&num,&x2,&three);           /* 3x^2 */
    fe_add(&den,&a->y,&a->y);          /* 2y */
    fe_inv(&t1,&den);
    fe_mul(&lam,&num,&t1);             /* lambda */
    fe_sqr(&t2,&lam);                  /* lambda^2 */
    fe_sub(&x3,&t2,&a->x);
    fe_sub(&x3,&x3,&a->x);             /* x3 = lam^2 - 2x */
    fe_sub(&t1,&a->x,&x3);
    fe_mul(&y3,&lam,&t1);
    fe_sub(&y3,&y3,&a->y);             /* y3 = lam(x-x3)-y */
    r->inf=0; r->x=x3; r->y=y3;
}

static inline void ge_neg(ge *r, const ge *a){
    fe pp; pp.n[0]=FE_P[0];pp.n[1]=FE_P[1];pp.n[2]=FE_P[2];pp.n[3]=FE_P[3];
    r->inf=a->inf; r->x=a->x; if(a->inf){ r->y=a->y; } else fe_sub(&r->y,&pp,&a->y);
}

static inline void ge_add(ge *r, const ge *a, const ge *b){
    if(a->inf){ *r=*b; return; }
    if(b->inf){ *r=*a; return; }
    if(fe_equal(&a->x,&b->x)){
        if(fe_equal(&a->y,&b->y)){ ge_double(r,a); return; }
        ge_set_infinity(r); return;          /* a == -b */
    }
    fe num,den,lam,t,x3,y3;
    fe_sub(&num,&b->y,&a->y);
    fe_sub(&den,&b->x,&a->x);
    fe_inv(&t,&den);
    fe_mul(&lam,&num,&t);
    fe_sqr(&t,&lam);
    fe_sub(&x3,&t,&a->x);
    fe_sub(&x3,&x3,&b->x);
    fe_sub(&t,&a->x,&x3);
    fe_mul(&y3,&lam,&t);
    fe_sub(&y3,&y3,&a->y);
    r->inf=0; r->x=x3; r->y=y3;
}

/* scalar (256-bit), little-endian limbs */
typedef struct { uint64_t n[4]; } sc;
static inline void sc_set_u64(sc *r,uint64_t v){ r->n[0]=v; r->n[1]=r->n[2]=r->n[3]=0; }
static inline void sc_set_hex(sc *r,const char *h){ fe tmp; fe_set_hex(&tmp,h);
    r->n[0]=tmp.n[0]; r->n[1]=tmp.n[1]; r->n[2]=tmp.n[2]; r->n[3]=tmp.n[3]; }
static inline int sc_bit(const sc *a,int i){ return (a->n[i>>6]>>(i&63))&1ULL; }

/* r = k * P  (double-and-add, MSB first). Setup-speed only. */
static inline void ge_scalar_mul(ge *r, const sc *k, const ge *P){
    ge acc; ge_set_infinity(&acc);
    int started=0;
    for(int i=255;i>=0;i--){
        if(started) ge_double(&acc,&acc);
        if(sc_bit(k,i)){ if(!started){ acc=*P; started=1; } else ge_add(&acc,&acc,P); }
    }
    *r=acc;
}
static inline void ge_scalar_base(ge *r,const sc *k){ ge G; ge_generator(&G); ge_scalar_mul(r,k,&G); }

/* decompress compressed pubkey: prefix(02/03) + 32-byte x.
 * returns 1 on success (point on curve). */
static inline int ge_decompress(ge *r, const char *hex33){
    /* hex33: 66 hex chars */
    int parity = (hex33[1]=='3') ? 1 : 0;   /* 02 even, 03 odd */
    char xh[65]; memcpy(xh,hex33+2,64); xh[64]=0;
    fe x; fe_set_hex(&x,xh);
    fe x2,x3,rhs,seven,y,y2;
    fe_sqr(&x2,&x); fe_mul(&x3,&x2,&x);
    fe_set_u64(&seven,7); fe_add(&rhs,&x3,&seven);
    static const uint64_t sqrt_exp[4]={
        0xffffffffbfffff0cULL,0xffffffffffffffffULL,
        0xffffffffffffffffULL,0x3fffffffffffffffULL};
    fe_pow(&y,&rhs,sqrt_exp);
    fe_sqr(&y2,&y);
    if(!fe_equal(&y2,&rhs)) return 0;        /* not on curve */
    if((int)(y.n[0]&1ULL)!=parity){ fe py; fe pp;
        pp.n[0]=FE_P[0];pp.n[1]=FE_P[1];pp.n[2]=FE_P[2];pp.n[3]=FE_P[3];
        fe_sub(&py,&pp,&y); y=py; }
    r->inf=0; r->x=x; r->y=y;
    return 1;
}

#endif
