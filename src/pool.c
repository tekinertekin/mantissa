#include "pool.h"

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#define TK_HAVE_PTHREADS 1
#endif

#ifndef TK_HAVE_PTHREADS
/* No pthreads (e.g. bare Windows build): run serially. */
void tk_parallel_for(int n, tk_row_fn fn, void *ctx) { fn(ctx, 0, n); }
int  tk_num_threads(void) { return 1; }
#else

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

#define TK_MAX_THREADS 64

static pthread_t      g_threads[TK_MAX_THREADS];
static int            g_workers = 0;                 /* background threads (main is +1) */
static pthread_mutex_t g_lock     = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_dispatch = PTHREAD_MUTEX_INITIALIZER; /* serialize concurrent callers */
static pthread_cond_t  g_work     = PTHREAD_COND_INITIALIZER;
static pthread_cond_t  g_done     = PTHREAD_COND_INITIALIZER;
static unsigned long  g_gen     = 0;                 /* job generation counter */
static int            g_pending = 0;                 /* workers still running the job */
static tk_row_fn      g_fn      = NULL;
static void          *g_ctx     = NULL;
static int            g_n       = 0;

/* Contiguous chunk [*b,*e) for worker `idx` of `total`, splitting n rows. */
static void chunk(int idx, int total, int n, int *b, int *e) {
    int base = n / total, rem = n % total;
    int begin = idx * base + (idx < rem ? idx : rem);
    *b = begin;
    *e = begin + base + (idx < rem ? 1 : 0);
}

static void *worker(void *arg) {
    const int id = (int)(long)arg;                   /* 0 .. g_workers-1 */
    unsigned long seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_lock);
        while (g_gen == seen) pthread_cond_wait(&g_work, &g_lock);
        seen = g_gen;
        tk_row_fn fn = g_fn; void *ctx = g_ctx; int n = g_n, total = g_workers + 1;
        pthread_mutex_unlock(&g_lock);

        int b, e; chunk(id, total, n, &b, &e);
        if (e > b) fn(ctx, b, e);

        pthread_mutex_lock(&g_lock);
        if (--g_pending == 0) pthread_cond_signal(&g_done);
        pthread_mutex_unlock(&g_lock);
    }
    return NULL;
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void pool_create(void) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("MANTISSA_THREADS");
    int want = env ? atoi(env) : (int)cores;
    if (want < 1) want = 1;
    if (want > TK_MAX_THREADS) want = TK_MAX_THREADS;
    g_workers = want - 1;                            /* the calling thread is one worker */
    for (int i = 0; i < g_workers; i++)
        pthread_create(&g_threads[i], NULL, worker, (void *)(long)i);
}
static void pool_init(void) { pthread_once(&g_once, pool_create); }

int tk_num_threads(void) { pool_init(); return g_workers + 1; }

void tk_parallel_for(int n, tk_row_fn fn, void *ctx) {
    pool_init();
    if (g_workers == 0) { fn(ctx, 0, n); return; }   /* single core / disabled */

    pthread_mutex_lock(&g_dispatch);                 /* one GEMV through the pool at a time */
    pthread_mutex_lock(&g_lock);
    g_fn = fn; g_ctx = ctx; g_n = n; g_pending = g_workers;
    g_gen++;
    pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_lock);

    int b, e; chunk(g_workers, g_workers + 1, n, &b, &e);  /* main takes the last chunk */
    if (e > b) fn(ctx, b, e);

    pthread_mutex_lock(&g_lock);
    while (g_pending > 0) pthread_cond_wait(&g_done, &g_lock);
    pthread_mutex_unlock(&g_lock);
    pthread_mutex_unlock(&g_dispatch);
}

#endif /* TK_HAVE_PTHREADS */
