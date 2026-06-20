/* secp256k1 prime field Fp arithmetic, 4x64 limbs, little-endian.
 * p = 2^256 - 2^32 - 977  =  2^256 - 0x1000003D1
 * Validated against Python bignum (see test_field.c / check_field.py).
 */
#ifndef FIELD_H
#define FIELD_H

#include <stdint.h>
#include <string.h>

typedef struct { uint64_t n[4]; } fe;   /* n[0] = least significant */

static const uint64_t FE_C = 0x1000003D1ULL;            /* 2^256 mod p */
static const uint64_t FE_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

static inline void fe_set_u64(fe *r, uint64_t v) {
    r->n[0] = v; r->n[1] = r->n[2] = r->n[3] = 0;
}
static inline int fe_is_zero(const fe *a) {
    return (a->n[0] | a->n[1] | a->n[2] | a->n[3]) == 0;
}
static inline int fe_equal(const fe *a, const fe *b) {
    return ((a->n[0]^b->n[0]) | (a->n[1]^b->n[1]) |
            (a->n[2]^b->n[2]) | (a->n[3]^b->n[3])) == 0;
}

/* compare a vs p; return 1 if a >= p */
static inline int fe_ge_p(const uint64_t a[4]) {
    if (a[3] != FE_P[3]) return a[3] > FE_P[3];
    if (a[2] != FE_P[2]) return a[2] > FE_P[2];
    if (a[1] != FE_P[1]) return a[1] > FE_P[1];
    return a[0] >= FE_P[0];
}
/* subtract p in place (assumes a >= p) */
static inline void fe_sub_p(uint64_t a[4]) {
    __uint128_t b = 0; int i;
    for (i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a[i] - FE_P[i] - b;
        a[i] = (uint64_t)t;
        b = (t >> 64) & 1;
    }
}
static inline void fe_reduce_weak(fe *r) {     /* bring into [0,p) given r<2p-ish */
    if (fe_ge_p(r->n)) fe_sub_p(r->n);
    if (fe_ge_p(r->n)) fe_sub_p(r->n);
}

static inline void fe_add(fe *r, const fe *a, const fe *b) {
    __uint128_t c = 0; int i;
    for (i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a->n[i] + b->n[i] + c;
        r->n[i] = (uint64_t)t; c = t >> 64;
    }
    /* fold the carry-out (value of 2^256) as +FE_C */
    if (c) {
        __uint128_t t = (__uint128_t)r->n[0] + FE_C;
        r->n[0] = (uint64_t)t; c = t >> 64;
        t = (__uint128_t)r->n[1] + c; r->n[1] = (uint64_t)t; c = t >> 64;
        t = (__uint128_t)r->n[2] + c; r->n[2] = (uint64_t)t; c = t >> 64;
        t = (__uint128_t)r->n[3] + c; r->n[3] = (uint64_t)t;
    }
    fe_reduce_weak(r);
}

/* r = a - b (mod p) */
static inline void fe_sub(fe *r, const fe *a, const fe *b) {
    __uint128_t brw = 0; int i;
    uint64_t t[4];
    for (i = 0; i < 4; i++) {
        __uint128_t d = (__uint128_t)a->n[i] - b->n[i] - brw;
        t[i] = (uint64_t)d;
        brw = (d >> 64) & 1;
    }
    if (brw) {  /* underflow: add p back */
        __uint128_t c = 0;
        for (i = 0; i < 4; i++) {
            __uint128_t s = (__uint128_t)t[i] + FE_P[i] + c;
            t[i] = (uint64_t)s; c = s >> 64;
        }
    }
    r->n[0]=t[0]; r->n[1]=t[1]; r->n[2]=t[2]; r->n[3]=t[3];
}

/* full 256x256 -> 512 schoolbook */
static inline void mul256(const uint64_t a[4], const uint64_t b[4], uint64_t out[8]) {
    uint64_t r[8] = {0,0,0,0,0,0,0,0};
    int i, j;
    for (i = 0; i < 4; i++) {
        __uint128_t carry = 0;
        for (j = 0; j < 4; j++) {
            __uint128_t cur = (__uint128_t)r[i+j] + (__uint128_t)a[i]*b[j] + carry;
            r[i+j] = (uint64_t)cur;
            carry = cur >> 64;
        }
        r[i+4] = (uint64_t)carry;
    }
    for (i = 0; i < 8; i++) out[i] = r[i];
}

/* reduce a 512-bit value (8 limbs) mod p into fe */
static inline void fe_reduce512(fe *r, const uint64_t t[8]) {
    /* fold high 256 (H) into low: result = L + H*C  */
    uint64_t m[5];
    __uint128_t carry = 0; int i;
    for (i = 0; i < 4; i++) {
        __uint128_t cur = (__uint128_t)t[4+i]*FE_C + carry;
        m[i] = (uint64_t)cur; carry = cur >> 64;
    }
    m[4] = (uint64_t)carry;

    uint64_t r5[5];
    __uint128_t s = 0;
    for (i = 0; i < 4; i++) {
        s += (__uint128_t)t[i] + m[i];
        r5[i] = (uint64_t)s; s >>= 64;
    }
    s += m[4];
    r5[4] = (uint64_t)s;           /* r5[4] < 2^34 */

    /* fold r5[4]*C into low 4 limbs */
    __uint128_t prod = (__uint128_t)r5[4] * FE_C;   /* < 2^67 */
    uint64_t plo = (uint64_t)prod, phi = (uint64_t)(prod >> 64);
    __uint128_t c;
    c = (__uint128_t)r5[0] + plo;            r->n[0] = (uint64_t)c; c >>= 64;
    c += (__uint128_t)r5[1] + phi;           r->n[1] = (uint64_t)c; c >>= 64;
    c += (__uint128_t)r5[2];                 r->n[2] = (uint64_t)c; c >>= 64;
    c += (__uint128_t)r5[3];                 r->n[3] = (uint64_t)c; c >>= 64;
    uint64_t extra = (uint64_t)c;            /* 0 or 1 */
    if (extra) {
        __uint128_t e = (__uint128_t)extra * FE_C;
        c = (__uint128_t)r->n[0] + (uint64_t)e; r->n[0] = (uint64_t)c; c >>= 64;
        c += (__uint128_t)r->n[1];              r->n[1] = (uint64_t)c; c >>= 64;
        c += (__uint128_t)r->n[2];              r->n[2] = (uint64_t)c; c >>= 64;
        c += (__uint128_t)r->n[3];              r->n[3] = (uint64_t)c;
    }
    fe_reduce_weak(r);
}

static inline void fe_mul(fe *r, const fe *a, const fe *b) {
    uint64_t t[8];
    mul256(a->n, b->n, t);
    fe_reduce512(r, t);
}
static inline void fe_sqr(fe *r, const fe *a) { fe_mul(r, a, a); }

static inline void fe_sqrn(fe *r, const fe *a, int n){ fe t=*a; for(int i=0;i<n;i++) fe_sqr(&t,&t); *r=t; }

/* r = a^-1 mod p via the canonical secp256k1 addition chain (a^(p-2)).
 * ~255 squarings + 15 muls, vs ~500 for Fermat. Validated vs Fermat + Python. */
static inline void fe_inv_fast(fe *r, const fe *a){
    fe x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(&x2,a);    fe_mul(&x2,&x2,a);
    fe_sqr(&x3,&x2);  fe_mul(&x3,&x3,a);
    fe_sqrn(&x6,&x3,3);    fe_mul(&x6,&x6,&x3);
    fe_sqrn(&x9,&x6,3);    fe_mul(&x9,&x9,&x3);
    fe_sqrn(&x11,&x9,2);   fe_mul(&x11,&x11,&x2);
    fe_sqrn(&x22,&x11,11); fe_mul(&x22,&x22,&x11);
    fe_sqrn(&x44,&x22,22); fe_mul(&x44,&x44,&x22);
    fe_sqrn(&x88,&x44,44); fe_mul(&x88,&x88,&x44);
    fe_sqrn(&x176,&x88,88);fe_mul(&x176,&x176,&x88);
    fe_sqrn(&x220,&x176,44);fe_mul(&x220,&x220,&x44);
    fe_sqrn(&x223,&x220,3); fe_mul(&x223,&x223,&x3);
    fe_sqrn(&t,&x223,23);  fe_mul(&t,&t,&x22);
    fe_sqrn(&t,&t,5);      fe_mul(&t,&t,a);
    fe_sqrn(&t,&t,3);      fe_mul(&t,&t,&x2);
    fe_sqrn(&t,&t,2);      fe_mul(&t,&t,a);
    *r=t;
}

/* r = a^-1 mod p  via Fermat: a^(p-2). Reference/validation only. */
static inline void fe_inv(fe *r, const fe *a) {
    /* p-2 = 0xFFFFFFFF...FFFFFFFEFFFFFC2D */
    static const uint64_t pm2[4] = {
        0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    fe result; fe_set_u64(&result, 1);
    fe base = *a;
    int i, bit;
    for (i = 0; i < 4; i++) {
        uint64_t w = pm2[i];
        for (bit = 0; bit < 64; bit++) {
            if (w & 1ULL) fe_mul(&result, &result, &base);
            fe_sqr(&base, &base);
            w >>= 1;
        }
    }
    *r = result;
}

/* hex I/O: 64-char big-endian hex */
static inline int hexval(char c){ if(c>='0'&&c<='9')return c-'0';
    if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return -1; }
static inline void fe_set_hex(fe *r, const char *h) {
    uint8_t b[32]; int i; size_t L = strlen(h);
    /* right-align */
    memset(b, 0, 32);
    int bi = 31, ci = (int)L - 1;
    int half = 0; uint8_t cur = 0;
    while (ci >= 0 && bi >= 0) {
        int v = hexval(h[ci]); if (v < 0) { ci--; continue; }
        if (!half) { cur = (uint8_t)v; half = 1; }
        else { cur |= (uint8_t)(v << 4); b[bi--] = cur; half = 0; }
        ci--;
    }
    if (half && bi >= 0) b[bi] = cur;
    for (i = 0; i < 4; i++) {
        r->n[i] = ((uint64_t)b[31-(i*8+0)])      | ((uint64_t)b[31-(i*8+1)]<<8)  |
                  ((uint64_t)b[31-(i*8+2)]<<16)  | ((uint64_t)b[31-(i*8+3)]<<24) |
                  ((uint64_t)b[31-(i*8+4)]<<32)  | ((uint64_t)b[31-(i*8+5)]<<40) |
                  ((uint64_t)b[31-(i*8+6)]<<48)  | ((uint64_t)b[31-(i*8+7)]<<56);
    }
}
static inline void fe_get_hex(const fe *a, char *out) {
    static const char *hx = "0123456789abcdef";
    int i, k = 0;
    for (i = 3; i >= 0; i--) {
        int s;
        for (s = 60; s >= 0; s -= 4) out[k++] = hx[(a->n[i] >> s) & 0xF];
    }
    out[64] = 0;
}

#endif
