/* Emits random field test vectors + this implementation's results.
 * check_field.py recomputes with Python bignum and diffs. */
#include <stdio.h>
#include <stdlib.h>
#include "field.h"

static uint64_t st;
static uint64_t xs(void){ st^=st<<13; st^=st>>7; st^=st<<17; return st; }

static void rand_fe(fe *r){
    r->n[0]=xs(); r->n[1]=xs(); r->n[2]=xs(); r->n[3]=xs();
    fe_reduce_weak(r);   /* ensure < p */
}

int main(int argc, char **argv){
    int N = (argc>1)? atoi(argv[1]) : 5000;
    st = 0x123456789abcdef0ULL;
    char ha[65],hb[65],hr[65];
    for(int i=0;i<N;i++){
        fe a,b,r; rand_fe(&a); rand_fe(&b);
        fe_get_hex(&a,ha); fe_get_hex(&b,hb);
        fe_add(&r,&a,&b); fe_get_hex(&r,hr); printf("ADD %s %s %s\n",ha,hb,hr);
        fe_sub(&r,&a,&b); fe_get_hex(&r,hr); printf("SUB %s %s %s\n",ha,hb,hr);
        fe_mul(&r,&a,&b); fe_get_hex(&r,hr); printf("MUL %s %s %s\n",ha,hb,hr);
        if(!fe_is_zero(&a)){ fe_inv(&r,&a); fe_get_hex(&r,hr); printf("INV %s %s %s\n",ha,ha,hr);
            fe rf; fe_inv_fast(&rf,&a); char hf[65]; fe_get_hex(&rf,hf); printf("INV %s %s %s\n",ha,ha,hf); }
    }
    return 0;
}
