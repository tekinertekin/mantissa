#ifndef MANTISSA_POOL_H
#define MANTISSA_POOL_H

/* Internal fork-join thread pool. The pool is created once (lazily) and reused,
 * so there is no per-call thread creation — safe for a hot path called millions
 * of times. On platforms without pthreads it degrades to a serial call. */

typedef void (*tk_row_fn)(void *ctx, int begin, int end);

/* Split [0, n) into contiguous chunks, run fn(ctx, begin, end) on each across
 * the pool, and block until all are done. Rows are disjoint, so callers write
 * their own output slice with no locking. */
void tk_parallel_for(int n, tk_row_fn fn, void *ctx);

/* Total workers including the calling thread (1 == serial). Honors the
 * MANTISSA_THREADS environment variable; defaults to the online CPU count. */
int tk_num_threads(void);

#endif /* MANTISSA_POOL_H */
