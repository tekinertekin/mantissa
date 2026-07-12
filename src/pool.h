#ifndef MANTISSA_POOL_H
#define MANTISSA_POOL_H

/* Internal fork-join thread pool. The pool is created once (lazily) and reused,
 * so there is no per-call thread creation — safe for a hot path called millions
 * of times. On platforms without pthreads it degrades to a serial call. */

/* `worker` is the chunk's index in [0, tk_num_threads()); use it to index
 * per-thread scratch (e.g. partial reduction buffers). */
typedef void (*tk_row_fn)(void *ctx, int begin, int end, int worker);

/* Split [0, n) into contiguous chunks, run fn(ctx, begin, end, worker) on each
 * across the pool, and block until all are done. Rows are disjoint, so callers
 * write their own output slice with no locking. */
void tk_parallel_for(int n, tk_row_fn fn, void *ctx);

/* Total workers including the calling thread (1 == serial). Honors the
 * MANTISSA_THREADS environment variable; defaults to the online CPU count. */
int tk_num_threads(void);

/* Parallelize a layer op only above this much work (out_dim * in_dim); smaller
 * layers run serially so pool overhead never hurts the many-small-calls case. */
#define TK_MT_MIN_WORK (1 << 18)

#endif /* MANTISSA_POOL_H */
