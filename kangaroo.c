/* Distinguished-point Pollard kangaroo for secp256k1 interval ECDLP.
 * Tame + wild herds, batched (Montgomery) inversion, pthreads, DP table.
 *
 * Built on validated field.h / group.h.
 *
 * Usage:
 *   ./kangaroo selftest <bits> [threads] [dpbits]
 *   ./kangaroo solve <compressed_pubkey> <Lhex> <Rhex> [threads] [dpbits] [dpslots_log2]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include "group.h"
#include "gpu_bridge.h"

/* ---------- 256-bit scalar helpers ---------- */
static const uint64_t ORDER_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};
static inline void sc_add(sc *r, const sc *a, const sc *b){
    __uint128_t c=0;
    for(int i=0;i<4;i++){ __uint128_t t=(__uint128_t)a->n[i]+b->n[i]+c; r->n[i]=(uint64_t)t; c=t>>64; }
}
/* r = (a - b) mod n, for a,b < n (or a-b small).  returns nothing. */
static inline void sc_sub_modn(sc *r, const sc *a, const sc *b){
    __uint128_t br=0; uint64_t t[4];
    for(int i=0;i<4;i++){ __uint128_t d=(__uint128_t)a->n[i]-b->n[i]-br; t[i]=(uint64_t)d; br=(d>>64)&1; }
    if(br){ __uint128_t c=0; for(int i=0;i<4;i++){ __uint128_t s=(__uint128_t)t[i]+ORDER_N[i]+c; t[i]=(uint64_t)s; c=s>>64; } }
    r->n[0]=t[0]; r->n[1]=t[1]; r->n[2]=t[2]; r->n[3]=t[3];
}
static inline void sc_print(const sc *a){ fe t; t.n[0]=a->n[0];t.n[1]=a->n[1];t.n[2]=a->n[2];t.n[3]=a->n[3];
    char h[65]; fe_get_hex(&t,h); printf("%s",h); }

/* ---------- config / globals ---------- */
#define MAXJ 512                 /* jump table size (power of two) */
#ifndef BATCH
#define BATCH 512                /* kangaroos per thread (override -DBATCH=N) */
#endif

typedef struct {
    fe     jx[MAXJ], jy[MAXJ];   /* jump points */
    sc     jd[MAXJ];             /* jump distances */
    int    njump;                /* = MAXJ */
    int    dpbits;
    uint64_t dpmask;
    ge     target;               /* P */
    ge     Qshift;               /* Q = P - L*G  (unknown DL q = k-L in [0,W)) */
    sc     rangeL;
    int    wbits;
    int    offbits;              /* random start-offset width */
    /* DP table (open addressing) */
    uint64_t slots;
    uint64_t mask;
    uint8_t *used;
    uint64_t *xk;                /* slots*4 */
    sc       *dist;
    uint8_t  *type;
    pthread_mutex_t lock;
    /* solution */
    volatile int solved;
    sc       answer;
    /* stats */
    volatile uint64_t total_jumps;   /* CPU jumps */
    volatile uint64_t gpu_jumps;     /* GPU jumps */
    volatile uint64_t dp_count;  /* distinguished points stored ("net markers") */
    int gpu_kangaroos;           /* reported by the GPU engine once up */
    int do_ckpt;
    volatile int stop_req;       /* set by SIGINT/SIGTERM for graceful save+exit */
} ctx_t;

#define CKPT_MAGIC 0x4B4E47525532353ULL   /* "KNGRU253" */

static ctx_t C;

/* async-signal-safe: only set a flag; the monitor thread does the actual save */
static void on_signal(int s){ (void)s; C.stop_req=1; }

/* xorshift per-thread rng */
static inline uint64_t xs(uint64_t *s){ uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; *s=x; return x; }
static void rand_sc_bits(sc *r, int bits, uint64_t *s){
    r->n[0]=r->n[1]=r->n[2]=r->n[3]=0;
    int full=bits/64, rem=bits%64;
    for(int i=0;i<full;i++) r->n[i]=xs(s);
    if(rem) r->n[full]=xs(s) & ((rem==64)?~0ULL:((1ULL<<rem)-1));
    if(r->n[0]==0 && bits>0) r->n[0]=1;
}

/* ---------- DP table ---------- */
/* returns 1 and fills *out_key if this insert produced a tame/wild collision */
static int dp_insert(const fe *x, const sc *dist, uint8_t type, sc *out_key){
    uint64_t h = (x->n[0]*0x9E3779B97F4A7C15ULL) ^ (x->n[2]*0xC2B2AE3D27D4EB4FULL);
    uint64_t i = h & C.mask;
    int found=0;
    pthread_mutex_lock(&C.lock);
    for(uint64_t probe=0; probe<C.slots; probe++){
        uint64_t s=(i+probe)&C.mask;
        if(!C.used[s]){
            C.used[s]=1; C.dp_count++;
            C.xk[s*4+0]=x->n[0]; C.xk[s*4+1]=x->n[1]; C.xk[s*4+2]=x->n[2]; C.xk[s*4+3]=x->n[3];
            C.dist[s]=*dist; C.type[s]=type;
            break;
        }
        if(C.xk[s*4+0]==x->n[0] && C.xk[s*4+1]==x->n[1] &&
           C.xk[s*4+2]==x->n[2] && C.xk[s*4+3]==x->n[3]){
            if(C.type[s]!=type){
                /* collision across herds: k = tame_dist - wild_dist */
                sc tame, wild;
                if(type==0){ tame=*dist; wild=C.dist[s]; }
                else        { tame=C.dist[s]; wild=*dist; }
                sc_sub_modn(out_key,&tame,&wild);
                found=1;
            }
            break; /* same x already present */
        }
    }
    pthread_mutex_unlock(&C.lock);
    return found;
}

/* ===================== GPU bridge ===================== */
/* fe/sc (4x u64) <-> gpu limbs (8x u32) */
static void fe_to_gpu(const fe*a,uint32_t*g){ for(int i=0;i<4;i++){ g[2*i]=(uint32_t)a->n[i]; g[2*i+1]=(uint32_t)(a->n[i]>>32);} }
static void sc_to_gpu(const sc*a,uint32_t*g){ for(int i=0;i<4;i++){ g[2*i]=(uint32_t)a->n[i]; g[2*i+1]=(uint32_t)(a->n[i]>>32);} }
static void gpu_to_fe(const uint32_t*g,fe*a){ for(int i=0;i<4;i++) a->n[i]=((uint64_t)g[2*i])|(((uint64_t)g[2*i+1])<<32);}

int  bridge_njump(void){ return C.njump; }
void bridge_jump(int i,uint32_t*jx,uint32_t*jy,uint32_t*jd){ fe_to_gpu(&C.jx[i],jx); fe_to_gpu(&C.jy[i],jy); sc_to_gpu(&C.jd[i],jd); }
void bridge_target(uint32_t*tx,uint32_t*ty){ fe_to_gpu(&C.target.x,tx); fe_to_gpu(&C.target.y,ty); }
void bridge_qshift(uint32_t*qx,uint32_t*qy){ fe_to_gpu(&C.Qshift.x,qx); fe_to_gpu(&C.Qshift.y,qy); }
void bridge_rangeL(uint32_t*l){ sc_to_gpu(&C.rangeL,l); }
int  bridge_dpbits(void){ return C.dpbits; }
int  bridge_wbits(void){ return C.wbits; }
int  bridge_solved(void){ return C.solved || C.stop_req; }
void bridge_add_jumps(uint64_t n){ __sync_fetch_and_add(&C.gpu_jumps,n); }
void bridge_set_kangaroos(int n){ C.gpu_kangaroos=n; }

/* feed one GPU-found DP into the shared table; 1 if it solved */
int bridge_feed_dp(const uint32_t*x8,const uint32_t*d8,uint8_t type){
    fe x; gpu_to_fe(x8,&x); sc dist; gpu_to_fe(d8,(fe*)&dist);
    sc q;
    if(dp_insert(&x,&dist,type,&q)){
        sc k; sc_add(&k,&C.rangeL,&q);
        ge T; ge_scalar_base(&T,&k);
        if(fe_equal(&T.x,&C.target.x)&&fe_equal(&T.y,&C.target.y)){
            pthread_mutex_lock(&C.lock);
            if(!C.solved){ C.solved=1; C.answer=k; }
            pthread_mutex_unlock(&C.lock);
            return 1;
        }
    }
    return 0;
}

/* ---------- jump-table setup ---------- */
static void build_jumps(int wbits, uint64_t seed){
    uint64_t s=seed?seed:0x243F6A8885A308D3ULL;
    int jb = wbits/2 + 1;        /* mean jump ~ sqrt(W) */
    ge G; ge_generator(&G);
    for(int i=0;i<MAXJ;i++){
        sc d; rand_sc_bits(&d, jb, &s);
        if(d.n[0]==0&&d.n[1]==0&&d.n[2]==0&&d.n[3]==0) d.n[0]=1;
        C.jd[i]=d;
        ge J; ge_scalar_mul(&J,&d,&G);
        C.jx[i]=J.x; C.jy[i]=J.y;
    }
    C.njump=MAXJ;
}

/* ---------- worker ---------- */
typedef struct { int id; int threads; uint64_t seed; } targ_t;

static void *worker(void *arg){
    targ_t *ta=(targ_t*)arg;
    uint64_t s = ta->seed ^ (0x9E3779B97F4A7C15ULL*(ta->id+1));

    /* per-kangaroo state */
    static __thread fe px[BATCH], py[BATCH];
    static __thread sc dist[BATCH];
    static __thread uint8_t typ[BATCH];
    fe den[BATCH], inv[BATCH], tmp[BATCH];

    ge G; ge_generator(&G);

    /* Shifted problem on [0,W): tame DL=dist, wild DL=q+dist where q=k-L.
     * Both herds spread uniformly across [0,W) (offbits = wbits). */
    for(int i=0;i<BATCH;i++){
        sc off; rand_sc_bits(&off, C.offbits, &s);
        ge OG; ge_scalar_mul(&OG,&off,&G);
        if((i&1)==0){
            /* tame: point = off*G, dist = off */
            px[i]=OG.x; py[i]=OG.y; dist[i]=off; typ[i]=0;
        } else {
            /* wild: point = Q + off*G, dist = off */
            ge Pw; ge_add(&Pw,&C.Qshift,&OG);
            px[i]=Pw.x; py[i]=Pw.y; dist[i]=off; typ[i]=1;
        }
    }

    uint64_t localjumps=0;
    while(!C.solved && !C.stop_req){
        /* compute denominators d_i = jx[idx_i] - px_i */
        for(int i=0;i<BATCH;i++){
            int idx = (int)(px[i].n[0] & (MAXJ-1));
            fe_sub(&den[i], &C.jx[idx], &px[i]);
            if(fe_is_zero(&den[i])) den[i].n[0]=1;   /* guard; reseeded below */
        }
        /* batch invert den[] -> inv[] */
        tmp[0]=den[0];
        for(int i=1;i<BATCH;i++) fe_mul(&tmp[i],&tmp[i-1],&den[i]);
        fe acc; fe_inv_fast(&acc,&tmp[BATCH-1]);
        for(int i=BATCH-1;i>=1;i--){
            fe_mul(&inv[i],&acc,&tmp[i-1]);
            fe_mul(&acc,&acc,&den[i]);
        }
        inv[0]=acc;
        /* step each kangaroo */
        for(int i=0;i<BATCH;i++){
            int idx = (int)(px[i].n[0] & (MAXJ-1));
            fe num,lam,lam2,x3,y3,t;
            fe_sub(&num,&C.jy[idx],&py[i]);
            fe_mul(&lam,&num,&inv[i]);
            fe_sqr(&lam2,&lam);
            fe_sub(&x3,&lam2,&px[i]);
            fe_sub(&x3,&x3,&C.jx[idx]);
            fe_sub(&t,&px[i],&x3);
            fe_mul(&y3,&lam,&t);
            fe_sub(&y3,&y3,&py[i]);
            px[i]=x3; py[i]=y3;
            sc_add(&dist[i],&dist[i],&C.jd[idx]);
            /* distinguished? */
            if((px[i].n[0] & C.dpmask)==0){
                sc q;
                if(dp_insert(&px[i],&dist[i],typ[i],&q)){
                    /* q = k - L; recover k and verify k*G == target P */
                    sc k; sc_add(&k,&C.rangeL,&q);
                    ge T; ge_scalar_base(&T,&k);
                    if(fe_equal(&T.x,&C.target.x)&&fe_equal(&T.y,&C.target.y)){
                        pthread_mutex_lock(&C.lock);
                        if(!C.solved){ C.solved=1; C.answer=k; }
                        pthread_mutex_unlock(&C.lock);
                    }
                }
            }
        }
        localjumps += BATCH;
        if((localjumps & 0xFFFFF)==0){ __sync_fetch_and_add(&C.total_jumps, 0x100000); }
    }
    return NULL;
}

/* ---------- human-readable formatting ---------- */
static void fmt_count(double v, char *out){
    if(v>=1e12)      sprintf(out,"%.2f trillion",v/1e12);
    else if(v>=1e9)  sprintf(out,"%.2f billion",v/1e9);
    else if(v>=1e6)  sprintf(out,"%.2f million",v/1e6);
    else if(v>=1e3)  sprintf(out,"%.1f thousand",v/1e3);
    else             sprintf(out,"%.0f",v);
}
static void fmt_time(double s, char *out){
    long t=(long)s; int h=(int)(t/3600), m=(int)((t%3600)/60), sec=(int)(t%60);
    if(h>0)      sprintf(out,"%dh %dm %ds",h,m,sec);
    else if(m>0) sprintf(out,"%dm %ds",m,sec);
    else         sprintf(out,"%ds",sec);
}

/* ---------- checkpoint / resume ---------- */
static void save_checkpoint(void){
    pthread_mutex_lock(&C.lock);
    FILE *f=fopen("checkpoint.tmp","wb");
    if(!f){ pthread_mutex_unlock(&C.lock); return; }
    uint64_t magic=CKPT_MAGIC, cnt=0, tj=C.total_jumps; int dp=C.dpbits;
    for(uint64_t i=0;i<C.slots;i++) if(C.used[i]) cnt++;
    fwrite(&magic,8,1,f);
    fwrite(C.target.x.n,8,4,f); fwrite(C.rangeL.n,8,4,f);
    fwrite(&dp,4,1,f); fwrite(&C.slots,8,1,f); fwrite(&cnt,8,1,f); fwrite(&tj,8,1,f);
    for(uint64_t i=0;i<C.slots;i++) if(C.used[i]){
        fwrite(&C.xk[i*4],8,4,f); fwrite(C.dist[i].n,8,4,f); fwrite(&C.type[i],1,1,f);
    }
    fclose(f);
    rename("checkpoint.tmp","checkpoint.bin");
    pthread_mutex_unlock(&C.lock);
}
static int load_checkpoint(void){
    FILE *f=fopen("checkpoint.bin","rb"); if(!f) return 0;
    uint64_t magic; if(fread(&magic,8,1,f)!=1||magic!=CKPT_MAGIC){ fclose(f); return 0; }
    uint64_t tx[4],rl[4],sl,cnt,tj; int dp;
    if(fread(tx,8,4,f)!=4||fread(rl,8,4,f)!=4||fread(&dp,4,1,f)!=1||
       fread(&sl,8,1,f)!=1||fread(&cnt,8,1,f)!=1||fread(&tj,8,1,f)!=1){ fclose(f); return 0; }
    if(memcmp(tx,C.target.x.n,32)||memcmp(rl,C.rangeL.n,32)||dp!=C.dpbits){
        fprintf(stderr,"checkpoint is for a different target/range/dpbits - ignoring\n"); fclose(f); return 0; }
    for(uint64_t i=0;i<cnt;i++){
        fe x; sc d; uint8_t ty; sc q;
        if(fread(x.n,8,4,f)!=4||fread(d.n,8,4,f)!=4||fread(&ty,1,1,f)!=1) break;
        if(dp_insert(&x,&d,ty,&q)){ sc k; sc_add(&k,&C.rangeL,&q);
            ge T; ge_scalar_base(&T,&k);
            if(fe_equal(&T.x,&C.target.x)){ C.solved=1; C.answer=k; } }
    }
    C.total_jumps=tj; C.dp_count=cnt; fclose(f);
    fprintf(stderr,"resumed: %llu net markers restored, prior jumps=%.3e\n",(unsigned long long)cnt,(double)tj);
    return 1;
}

/* ===================== GREENROO green dashboard ===================== */
#define CG  "\033[92m"   /* bright green */
#define CGD "\033[32m"   /* green */
#define CB  "\033[1m"    /* bold */
#define CD  "\033[2m"    /* dim */
#define CR  "\033[0m"    /* reset */

static void short_target(char*out,int n){
    char h[65]; fe_get_hex(&C.target.x,h);
    snprintf(out,n,"%.10s…%s",h,h+54);
}
static void dash_banner(void){
    fprintf(stderr,"\n"
      CG CB "    ╔═══════════════════════════════════════════════════╗\n" CR
      CG CB "    ║   ▄▖▖   " CR CG "G R E E N R O O" CB "                          ║\n" CR
      CG CB "    ║  ▐▌▌▌   " CR CGD "secp256k1 ECDLP kangaroo hunter" CB "        ║\n" CR
      CG CB "    ║   ▘▝▘   " CR CGD "Apple Silicon · CPU + Metal GPU" CB "        ║\n" CR
      CG CB "    ╚═══════════════════════════════════════════════════╝\n" CR "\n");
}
static int g_dash_lines=0;
static int g_threads=0;
static void bar(char*out,double v,double mx){
    int f=(mx>0)?(int)(12.0*v/mx+0.5):0; if(f>12)f=12; if(f<0)f=0;
    int k=0; for(int i=0;i<12;i++){ const char*c=(i<f)?"█":"░";
        out[k++]=c[0]; out[k++]=c[1]; out[k++]=c[2]; } out[k]=0;
}
static void dash_render(int tty,double el,double cpu_r,double gpu_r){
    uint64_t total=C.total_jumps+C.gpu_jumps;
    double tot_r=cpu_r+gpu_r, mx=(cpu_r>gpu_r)?cpu_r:gpu_r;
    char tb[40],kb[40],tgt[64],cbar[40],gbar[40],groo[32];
    fmt_time(el,tb); fmt_count((double)total,kb); short_target(tgt,sizeof tgt);
    bar(cbar,cpu_r,mx); bar(gbar,gpu_r,mx);
    if(C.gpu_kangaroos>0) snprintf(groo,sizeof groo,"%d roos",C.gpu_kangaroos);
    else                  snprintf(groo,sizeof groo,"seeding…");
    if(!tty){
        fprintf(stderr,"[%s] CPU %.0f + GPU %.0f = %.0f M/s | %s keys | net %llu\n",
                tb,cpu_r,gpu_r,tot_r,kb,(unsigned long long)C.dp_count);
        return;
    }
    if(g_dash_lines) fprintf(stderr,"\033[%dA",g_dash_lines);
    int L=0;
    #define DL(...) do{ fprintf(stderr,"\033[2K"); fprintf(stderr,__VA_ARGS__); fprintf(stderr,"\n"); L++; }while(0)
    DL(CG CB "  ┌─ GREENROO ──────────────────── PUZZLE #%d ─┐" CR, C.wbits+1);
    DL(CG "  │ " CR CD "target " CR CG "%s" CR, tgt);
    DL(CG "  │ " CR CD "range  " CR CGD "2^%d … 2^%d-1" CR, C.wbits, C.wbits+1);
    DL(CG "  │" CR);
    DL(CG "  │ " CR CD "uptime " CR CB "%s" CR, tb);
    DL(CG "  │ " CR "CPU " CD "%2d cores  " CR CB CG "%4.0f" CR " M/s " CGD "%s" CR, g_threads, cpu_r, cbar);
    DL(CG "  │ " CR "GPU " CD "%-9s " CR CB CG "%4.0f" CR " M/s " CGD "%s" CR, groo, gpu_r, gbar);
    DL(CG "  │" CR);
    DL(CG "  │ " CR CB "TOTAL    " CR CB CG "%5.0f" CR CB " M keys/sec" CR, tot_r);
    DL(CG "  │ " CR CD "checked  " CR "%s" , kb);
    DL(CG "  │ " CR CD "DP net   " CR "%llu markers", (unsigned long long)C.dp_count);
    DL(CG CB "  └────────────────────────────────────────────┘" CR);
    #undef DL
    g_dash_lines=L;
    fflush(stderr);
}

/* ---------- driver ---------- */
static void run(int threads){
    pthread_t th[64]; targ_t ta[64];
    uint64_t base_seed = 0xB5297A4D1F2E3C6BULL;
    for(int i=0;i<threads;i++){ ta[i].id=i; ta[i].threads=threads; ta[i].seed=base_seed; }
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC,&t0);
    g_threads=threads;
    for(int i=0;i<threads;i++) pthread_create(&th[i],NULL,worker,&ta[i]);

    int tty=isatty(2);
    dash_banner();
    if(tty) fprintf(stderr,"\033[?25l");        /* hide cursor */

    int tick=0; int ckpt_sec=120;
    { const char *e=getenv("CKPT_SEC"); if(e){ int v=atoi(e); if(v>0) ckpt_sec=v; } }
    uint64_t pc=0,pg=0; double pel=0;
    while(!C.solved && !C.stop_req){
        struct timespec ts={1,0}; nanosleep(&ts,NULL);
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC,&t1);
        double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        uint64_t cj=C.total_jumps, gj=C.gpu_jumps;
        double dt=el-pel; if(dt<0.1)dt=1.0;
        double cr=(double)(cj-pc)/dt/1e6, gr=(double)(gj-pg)/dt/1e6;
        pc=cj; pg=gj; pel=el;
        if(tty) dash_render(1,el,cr,gr);
        else if(tick%10==0) dash_render(0,el,cr,gr);
        if(C.do_ckpt && (++tick % ckpt_sec)==0) save_checkpoint();
        if(C.solved || C.stop_req) break;
    }
    if(tty) fprintf(stderr,"\033[?25h");        /* show cursor */
    if(C.stop_req && C.do_ckpt){ fprintf(stderr,"\nstop requested - saving checkpoint...\n"); save_checkpoint(); }
    for(int i=0;i<threads;i++) pthread_join(th[i],NULL);
    if(C.do_ckpt) save_checkpoint();
    fprintf(stderr,"\n");
}

static void alloc_table(int slots_log2){
    C.slots = 1ULL<<slots_log2; C.mask=C.slots-1;
    C.used = calloc(C.slots,1);
    C.xk   = calloc(C.slots*4,sizeof(uint64_t));
    C.dist = calloc(C.slots,sizeof(sc));
    C.type = calloc(C.slots,1);
    if(!C.used||!C.xk||!C.dist||!C.type){ fprintf(stderr,"alloc fail\n"); exit(1); }
    pthread_mutex_init(&C.lock,NULL);
}

static int bitlen_sc(const sc *a){
    for(int i=3;i>=0;i--) if(a->n[i]){ int b=63; while(!((a->n[i]>>b)&1))b--; return i*64+b+1; }
    return 0;
}

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: selftest <bits> [threads] [dpbits] | solve <pub> <L> <R> [threads] [dpbits] [slots_log2]\n"); return 1; }

    if(!strcmp(argv[1],"pub")){          /* print compressed pubkey of a privkey hex */
        sc k; sc_set_hex(&k,argv[2]);
        ge P; ge_scalar_base(&P,&k);
        char xh[65]; fe_get_hex(&P.x,xh);
        printf("%02x%s\n",(P.y.n[0]&1)?3:2,xh);
        return 0;
    }

    if(!strcmp(argv[1],"selftest")){
        int bits=atoi(argv[2]);
        int threads=(argc>3)?atoi(argv[3]):8;
        ge G; ge_generator(&G);
        /* random secret key in [2^(bits-1), 2^bits) */
        uint64_t s=0xCAFEBABEDEADBEEFULL ^ ((uint64_t)time(NULL));
        sc key; rand_sc_bits(&key,bits,&s); key.n[(bits-1)>>6]|=(1ULL<<((bits-1)&63));
        ge P; ge_scalar_mul(&P,&key,&G);
        C.target=P;
        /* range [2^(bits-1), 2^bits) */
        sc L; L.n[0]=L.n[1]=L.n[2]=L.n[3]=0; L.n[(bits-1)>>6]|=(1ULL<<((bits-1)&63));
        C.rangeL=L; C.wbits=bits-1;            /* W = 2^(bits-1) */
        int nk=9; { int t=threads; while(t>1){nk++; t>>=1;} }   /* log2(threads*BATCH) */
        int dpbits=(argc>4)?atoi(argv[4]):(C.wbits/2 - nk - 3);
        if(dpbits<1)dpbits=1;
        C.dpbits=dpbits; C.dpmask=(1ULL<<dpbits)-1;
        C.offbits = C.wbits;                   /* spread both herds across [0,W) */
        { ge LG; ge_scalar_mul(&LG,&L,&G); ge nLG; ge_neg(&nLG,&LG); ge_add(&C.Qshift,&P,&nLG); }
        fprintf(stderr,"[selftest] bits=%d  secret=",bits); sc_print(&key); fprintf(stderr,"\n");
        fprintf(stderr,"           dpbits=%d threads=%d jumpmean~2^%d\n",dpbits,threads,C.wbits/2);
        build_jumps(C.wbits, 0x1234);
        int slots_log2 = (dpbits+2 < 26)?22:24;
        alloc_table(slots_log2);
        run(threads);
        printf("\nSOLVED key = "); sc_print(&C.answer); printf("\n");
        printf("expected   = "); sc_print(&key); printf("\n");
        printf(fe_equal((fe*)&C.answer,(fe*)&key)?"MATCH\n":"-- (different representative, verify via point) --\n");
        return 0;
    }

    if(!strcmp(argv[1],"solve")){
        const char *pub=argv[2];
        ge P; if(!ge_decompress(&P,pub)){ fprintf(stderr,"bad pubkey / not on curve\n"); return 1; }
        C.target=P;
        sc L,R; sc_set_hex(&L,argv[3]); sc_set_hex(&R,argv[4]);
        int threads=(argc>5)?atoi(argv[5]):10;
        sc W; sc_sub_modn(&W,&R,&L); int wb=bitlen_sc(&W);
        C.rangeL=L; C.wbits=wb;
        int nk=9; { int t=threads; while(t>1){nk++; t>>=1;} }   /* log2(threads*BATCH) */
        int dpbits=(argc>6)?atoi(argv[6]):(wb/2 - nk - 3); if(dpbits<1)dpbits=1;
        C.dpbits=dpbits; C.dpmask=(1ULL<<dpbits)-1;
        C.offbits=wb;                          /* spread both herds across [0,W) */
        { ge G2; ge_generator(&G2); ge LG; ge_scalar_mul(&LG,&L,&G2);
          ge nLG; ge_neg(&nLG,&LG); ge_add(&C.Qshift,&P,&nLG); }
        int slots_log2=(argc>7)?atoi(argv[7]):24;
        fprintf(stderr,"[solve] W bits=%d dpbits=%d threads=%d slots=2^%d jumpmean~2^%d\n",
                wb,dpbits,threads,slots_log2,wb/2);
        build_jumps(wb,0x1234);
        alloc_table(slots_log2);
        C.do_ckpt=1;
        load_checkpoint();                     /* resume if checkpoint.bin matches */
        signal(SIGINT,on_signal); signal(SIGTERM,on_signal);   /* graceful stop */
        pthread_t gth; int gpu_on=0;
        if(!getenv("NOGPU")){ if(pthread_create(&gth,NULL,gpu_thread,NULL)==0) gpu_on=1; }
        run(threads);
        if(gpu_on) pthread_join(gth,NULL);
        if(!C.solved){
            fprintf(stderr,"stopped without solving - progress is in checkpoint.bin (resume by relaunching)\n");
            return 0;
        }
        fe a; a.n[0]=C.answer.n[0];a.n[1]=C.answer.n[1];a.n[2]=C.answer.n[2];a.n[3]=C.answer.n[3];
        char kh[65]; fe_get_hex(&a,kh);
        FILE *f=fopen("FOUND.txt","w");
        if(f){ fprintf(f,"pubkey   %s\nprivkey  %s\n",pub,kh); fclose(f); }
        printf("\n\n");
        printf("***********************************************************\n");
        printf("***                                                     ***\n");
        printf("***      KEY FOUND!   saved to FOUND.txt                 ***\n");
        printf("***                                                     ***\n");
        printf("***********************************************************\n");
        printf("  private key = %s\n",kh);
        printf("***********************************************************\n\n");
        return 0;
    }
    fprintf(stderr,"unknown mode\n"); return 1;
}
