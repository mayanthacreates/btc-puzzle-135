/* Validate ge_scalar_base and ge_decompress against libsecp256k1. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <secp256k1.h>
#include "group.h"

static uint64_t st;
static uint64_t xs(void){ st^=st<<13; st^=st>>7; st^=st<<17; return st; }

int main(int argc,char**argv){
    int N=(argc>1)?atoi(argv[1]):2000;
    st=0xdeadbeefcafef00dULL;
    secp256k1_context *ctx=secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    int fails=0;
    for(int t=0;t<N;t++){
        uint8_t sk[32];
        for(int i=0;i<32;i++) sk[i]=(uint8_t)(xs()&0xff);
        sk[0]&=0x7f; if(sk[31]==0&&sk[0]==0) sk[31]=1;   /* keep in range, nonzero */
        secp256k1_pubkey pk;
        if(!secp256k1_ec_pubkey_create(ctx,&pk,sk)) continue;
        uint8_t comp[33]; size_t L=33;
        secp256k1_ec_pubkey_serialize(ctx,comp,&L,&pk,SECP256K1_EC_COMPRESSED);
        char chex[67]; for(int i=0;i<33;i++) sprintf(chex+i*2,"%02x",comp[i]); chex[66]=0;

        /* my scalar*G */
        char skhex[65]; for(int i=0;i<32;i++) sprintf(skhex+i*2,"%02x",sk[i]); skhex[64]=0;
        sc k; sc_set_hex(&k,skhex);
        ge P; ge_scalar_base(&P,&k);
        char myx[65]; fe_get_hex(&P.x,myx);
        char refx[65]; memcpy(refx,chex+2,64); refx[64]=0;
        int myparity=(int)(P.y.n[0]&1ULL);
        int refparity=(comp[0]==3)?1:0;
        if(strcmp(myx,refx)!=0 || myparity!=refparity){
            if(fails<3) printf("SCALARMUL MISMATCH sk=%s\n my=%s p%d\n ref=%s p%d\n",
                               skhex,myx,myparity,refx,refparity);
            fails++; continue;
        }
        /* decompress round-trip */
        ge D; if(!ge_decompress(&D,chex)){ if(fails<3)printf("DECOMP FAIL %s\n",chex); fails++; continue; }
        if(!fe_equal(&D.x,&P.x)||!fe_equal(&D.y,&P.y)){ if(fails<3)printf("DECOMP MISMATCH %s\n",chex); fails++; }
    }
    printf("tested=%d fails=%d\n",N,fails);
    secp256k1_context_destroy(ctx);
    return fails?1:0;
}
