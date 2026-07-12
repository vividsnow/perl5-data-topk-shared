/*
 * topk.h -- Shared-memory top-k heavy hitters (Space-Saving) for Linux
 *
 * Streaming heavy-hitters: with a fixed set of m counters it tracks the m most
 * frequent keys of a stream and estimates each one's count, using the
 * Space-Saving algorithm. Observing a monitored key bumps its counter;
 * observing a new key when full evicts the smallest counter and hands its count
 * to the newcomer as an over-estimate bound, so a key's true count lies in
 * [count-error, count]. A min-heap over the counts finds the eviction victim in
 * O(log m) and an intrusive hash index tests membership in O(1). The counters
 * live in a shared mapping so several processes feed one summary; a
 * write-preferring futex rwlock with reader-slot dead-process recovery guards
 * mutation. Keys are compared by their first key_size bytes.
 *
 * Layout: Header -> reader_slots[1024] -> slots[m] -> heap[m] -> buckets[nb]
 */

#ifndef TK_H
#define TK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <pthread.h>

#define XXH_INLINE_ALL
#include "xxhash.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#error "topk.h: requires little-endian architecture"
#endif


/* ================================================================
 * Constants
 * ================================================================ */

#define TK_MAGIC        0x4B504F54  /* TopK */
#define TK_VERSION      1
#define TK_ERR_BUFLEN   256
#ifndef TK_READER_SLOTS
#define TK_READER_SLOTS 1024         /* max concurrent reader processes for dead-process recovery */
#endif
#define TK_MIN_CAP      1                /* min number of monitored counters */
#define TK_MAX_CAP      0x1000000ULL     /* 2^24 counters cap */
#define TK_MIN_KEYSIZE  1                /* min inline key bytes */
#define TK_MAX_KEYSIZE  4096             /* max inline key bytes */
#define TK_NIL          0xFFFFFFFFU      /* empty slot-index sentinel (hash chains, heap) */

#define TK_MODE_PLAIN   0U               /* integer counts (classic Space-Saving) */
#define TK_MODE_DECAYED 1U               /* forward-decayed weighted counts (Cormode) */
#define TK_RESCALE_LN   34.0             /* rescale the landmark once alpha*(now-L) exceeds this (g ~ 6e14) */

#define TK_ERR(fmt, ...) do { if (errbuf) snprintf(errbuf, TK_ERR_BUFLEN, fmt, ##__VA_ARGS__); } while (0)

/* ================================================================
 * Structs
 * ================================================================ */

/* Per-process slot for dead-process recovery.  Each shared rwlock counter
 * (the main rwlock-reader count, rwlock_waiters, rwlock_writers_waiting)
 * is mirrored here so a wrlock timeout can attribute and reverse a dead
 * process's contribution instead of waiting for the slow per-op timeout
 * drain. */
typedef struct {
    uint32_t pid;            /* 0 = unclaimed */
    uint32_t subcount;       /* in-flight rdlock acquisitions for this process */
    uint32_t waiters_parked; /* contribution to hdr->rwlock_waiters         */
    uint32_t writers_parked; /* contribution to hdr->rwlock_writers_waiting */
} TkReaderSlot;

struct TkHeader {
    uint32_t magic, version;          /* 0,4 */
    uint32_t capacity;                /* 8   m: number of monitored counters (slots) */
    uint32_t key_size;                /* 12  max inline key bytes */
    uint64_t hash_buckets;            /* 16  number of hash buckets (power of two) */
    uint64_t used;                    /* 24  distinct keys currently monitored (<= capacity) */
    uint64_t seen;                    /* 32  total observations */
    uint64_t slots_off;               /* 40  offset of the slots array */
    uint64_t total_size;              /* 48 */
    uint64_t reader_slots_off;        /* 56 */
    uint64_t heap_off;                /* 64  offset of the min-heap array (slot indices) */
    uint32_t rwlock;                  /* 72 */
    uint32_t rwlock_waiters;          /* 76 */
    uint32_t rwlock_writers_waiting;  /* 80 */
    uint32_t slotless_readers;  /* live readers holding the lock with no reader-slot */
    uint64_t stat_ops;                /* 88 */
    uint64_t bucket_off;              /* 96  offset of the hash bucket array */
    uint32_t mode;                    /* 104 TK_MODE_PLAIN | TK_MODE_DECAYED */
    uint32_t _pad2;                   /* 108 */
    double   alpha;                   /* 112 decay rate ln(2)/half_life (decayed mode; 0 if plain) */
    double   now;                     /* 120 current time: max timestamp seen, or per-add tick */
    double   landmark;                /* 128 forward-decay landmark L (decayed mode) */
    uint8_t  _pad[120];               /* 136..255 */
};
typedef struct TkHeader TkHeader;

_Static_assert(sizeof(TkHeader) == 256, "TkHeader must be 256 bytes");

/* Per-slot record: a monitored (key, count, error) triple.  The key bytes
 * (key_size of them) follow this fixed header in memory; the per-slot stride
 * is sizeof(TkSlot) + align8(key_size).  heap_pos is this slot's index in the
 * min-heap; hnext chains slots that share a hash bucket. */
typedef struct {
    uint64_t count;      /* observed count (an upper bound on the true count) */
    uint64_t error;      /* max over-estimate: true count is in [count-error, count] */
    uint32_t heap_pos;   /* position of this slot in the heap array */
    uint32_t hnext;      /* next slot in the hash chain, or TK_NIL */
    uint32_t key_len;    /* stored key length (<= key_size) */
    uint32_t _pad;       /* keep the trailing key bytes 8-aligned */
} TkSlot;
_Static_assert(sizeof(TkSlot) == 32, "TkSlot must be 32 bytes");

/* ---- Process-local handle ---- */

typedef struct TkHandle {
    TkHeader     *hdr;
    TkReaderSlot *reader_slots;  /* TK_READER_SLOTS entries */
    void         *base;          /* mmap base */
    uint64_t      slots_off;     /* validated, cached: never re-read from the peer-writable header */
    uint64_t      heap_off;      /* validated, cached */
    uint64_t      bucket_off;    /* validated, cached */
    uint32_t      capacity;      /* m, cached */
    uint32_t      key_size;      /* cached */
    uint64_t      hash_mask;     /* hash_buckets - 1, cached */
    uint64_t      stride;        /* per-slot byte stride, cached */
    uint32_t      mode;          /* TK_MODE_* (cached from validated header) */
    double        alpha;         /* decay rate (cached; 0 if plain) */
    size_t        mmap_size;
    char         *path;          /* backing file path (strdup'd) */
    int           backing_fd;    /* memfd or reopened-fd to close on destroy, -1 for file/anon */
    uint32_t      my_slot_idx;   /* UINT32_MAX if all slots taken (no recovery for this handle) */
    uint32_t      cached_pid;    /* getpid() cached at last slot claim */
    uint32_t      cached_fork_gen; /* tk_fork_gen value at last slot claim */
    uint32_t slotless_held; /* rwlock read-locks held with no reader-slot */
} TkHandle;

/* ================================================================
 * Futex-based write-preferring read-write lock
 * with reader-slot dead-process recovery
 * ================================================================ */

#define TK_RWLOCK_SPIN_LIMIT 32
#define TK_LOCK_TIMEOUT_SEC  2  /* FUTEX_WAIT timeout for stale lock detection */

static inline void tk_rwlock_spin_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

/* Extract writer PID from rwlock value (lower 31 bits when write-locked). */
#define TK_RWLOCK_WRITER_BIT 0x80000000U
#define TK_RWLOCK_PID_MASK   0x7FFFFFFFU
#define TK_RWLOCK_WR(pid)    (TK_RWLOCK_WRITER_BIT | ((uint32_t)(pid) & TK_RWLOCK_PID_MASK))

/* Check if a PID is alive. Returns 1 if alive or unknown, 0 if definitely dead. */
/* Liveness via kill(pid,0). NOTE: cannot detect PID reuse -- if a dead
 * lock-holder's PID is recycled to an unrelated live process before recovery
 * runs, this reports "alive" and that slot's orphaned contribution is not
 * reclaimed until the recycled process exits. Robust detection would require
 * a per-slot process-start-time epoch (a header-layout/version change).
 * Documented under "Crash Safety" in the POD. */
static inline int tk_pid_alive(uint32_t pid) {
    if (pid == 0) return 1; /* no owner recorded, assume alive */
    return !(kill((pid_t)pid, 0) == -1 && errno == ESRCH);
}

/* Force-recover a stale write lock left by a dead process.
 * CAS to OUR pid to hold the lock while fixing shared state, then release.
 * Using our pid (not a bare WRITER_BIT sentinel) means a subsequent
 * recovering process can detect and re-recover if we crash mid-recovery. */
static inline void tk_recover_stale_lock(TkHandle *h, uint32_t observed_rwlock) {
    TkHeader *hdr = h->hdr;
    uint32_t mypid = TK_RWLOCK_WR((uint32_t)getpid());
    if (!__atomic_compare_exchange_n(&hdr->rwlock, &observed_rwlock,
            mypid, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
    /* We now hold the write lock as mypid.  No additional shared state needs
     * repair here (this module has no seqlock); just release the lock. */
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static const struct timespec tk_lock_timeout = { TK_LOCK_TIMEOUT_SEC, 0 };

/* Process-global fork-generation counter.  Incremented in the pthread_atfork
 * child callback so every open handle detects a fork transition on the next
 * lock call without paying a getpid() syscall on the hot path. */
static uint32_t tk_fork_gen = 1;
static pthread_once_t tk_atfork_once = PTHREAD_ONCE_INIT;
static void tk_on_fork_child(void) {
    __atomic_add_fetch(&tk_fork_gen, 1, __ATOMIC_RELAXED);
}
static void tk_atfork_init(void) {
    pthread_atfork(NULL, NULL, tk_on_fork_child);
}

/* Ensure this process owns a reader slot.  Called from the lock helpers so
 * that fork()'d children pick up their own slot lazily instead of sharing
 * the parent's.  Hot-path is a single relaxed load + compare; only on a
 * fork-generation mismatch do we touch getpid() and scan slots. */
static inline void tk_claim_reader_slot(TkHandle *h) {
    uint32_t cur_gen = __atomic_load_n(&tk_fork_gen, __ATOMIC_RELAXED);
    if (__builtin_expect(cur_gen == h->cached_fork_gen && h->my_slot_idx != UINT32_MAX, 1))
        return;
    /* Cold path -- register the atfork hook once per process, then claim. */
    pthread_once(&tk_atfork_once, tk_atfork_init);
    /* Re-read after pthread_once: tk_on_fork_child may have bumped it. */
    cur_gen = __atomic_load_n(&tk_fork_gen, __ATOMIC_RELAXED);
    uint32_t now_pid = (uint32_t)getpid();
    h->cached_pid = now_pid;
    if (cur_gen != h->cached_fork_gen) h->slotless_held = 0;  /* fork: child holds none of the parent's slotless read locks */
    h->cached_fork_gen = cur_gen;
    h->my_slot_idx = UINT32_MAX;
    uint32_t start = now_pid % TK_READER_SLOTS;
    for (uint32_t i = 0; i < TK_READER_SLOTS; i++) {
        uint32_t s = (start + i) % TK_READER_SLOTS;
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(&h->reader_slots[s].pid,
                &expected, now_pid, 0,
                __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
            /* Zero all mirror fields, not just subcount: a SIGKILL'd
             * predecessor may have left waiters_parked/writers_parked
             * non-zero, and tk_recover_dead_readers won't drain them
             * once we own the slot (the CAS expects the dead PID). */
            __atomic_store_n(&h->reader_slots[s].subcount, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].waiters_parked, 0, __ATOMIC_RELAXED);
            __atomic_store_n(&h->reader_slots[s].writers_parked, 0, __ATOMIC_RELAXED);
            h->my_slot_idx = s;
            return;
        }
    }
    /* Table full -- leave my_slot_idx = UINT32_MAX so we silently skip
     * tracking for this handle (lock still works; just no recovery). */
}

/* Atomically subtract `sub` from a counter, capped at 0 (never underflows). */
static inline void tk_atomic_sub_cap(uint32_t *p, uint32_t sub) {
    if (!sub) return;
    uint32_t cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        uint32_t want = (cur > sub) ? cur - sub : 0;
        if (__atomic_compare_exchange_n(p, &cur, want,
                1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

/* Try to claim a dead slot (CAS pid -> 0) and drain its parked-waiter
 * contributions back to the global counters.  A no-op if the slot was stolen
 * by another recoverer or had no waiter contribution to drain.
 *
 * Note: subcount/waiters_parked/writers_parked are NOT zeroed here.
 * Between our CAS and a follow-up store, a new process could claim the
 * slot and start populating these fields -- our stores would clobber its
 * state.  tk_claim_reader_slot zeros all three on every claim, so
 * leaving stale values is harmless. */
static inline void tk_drain_dead_slot(TkHandle *h, uint32_t i, uint32_t pid) {
    TkHeader *hdr = h->hdr;
    uint32_t expected = pid;
    /* ACQ_REL on success: RELEASE publishes pid=0 to other observers;
     * ACQUIRE syncs us with prior writes from the dead process to
     * waiters_parked/writers_parked.  On weakly-ordered archs (aarch64)
     * a plain RELAXED load before the CAS could miss those writes;
     * loading them after the CAS keeps them inside the acquire window. */
    if (!__atomic_compare_exchange_n(&h->reader_slots[i].pid, &expected, 0,
            0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        return;
    uint32_t wp    = __atomic_load_n(&h->reader_slots[i].waiters_parked, __ATOMIC_RELAXED);
    uint32_t writp = __atomic_load_n(&h->reader_slots[i].writers_parked, __ATOMIC_RELAXED);
    if (wp)    tk_atomic_sub_cap(&hdr->rwlock_waiters, wp);
    if (writp) tk_atomic_sub_cap(&hdr->rwlock_writers_waiting, writp);
}

/* Scan reader slots for dead-process recovery.
 *
 * For each dead PID with non-zero contributions to the shared rwlock,
 * rwlock_waiters, or rwlock_writers_waiting counters, drain its share back
 * out so live processes don't have to wait for the slow per-op timeout
 * decrement to drain it for them.
 *
 * For the main rwlock counter we use the "no live reader holds -> force-
 * reset to 0" trick (precise) because per-process attribution of the
 * subcount is racy across the inc-counter-then-inc-subcount window. */
static inline void tk_recover_dead_readers(TkHandle *h) {
    if (!h->reader_slots) return;
    TkHeader *hdr = h->hdr;
    int any_live_reader = 0;
    int found_dead_reader = 0;

    /* Pass 1: classify slots.  Slots with dead pid and sc == 0 (no rwlock
     * contribution to lose) are wiped immediately to free the slot for
     * future claimants and drain any orphan parked-waiter counters.  Slots
     * with dead pid and sc > 0 are left intact in this pass: if force-
     * reset cannot fire (because a live reader is concurrently present),
     * wiping the dead slot would lose the only record of its orphan
     * rwlock contribution and strand writers permanently once the live
     * reader releases. */
    for (uint32_t i = 0; i < TK_READER_SLOTS; i++) {
        uint32_t pid = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
        if (pid == 0) continue;
        uint32_t sc = __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED);
        if (tk_pid_alive(pid)) {
            if (sc > 0) any_live_reader = 1;
            continue;
        }
        if (sc > 0) { found_dead_reader = 1; continue; }
        tk_drain_dead_slot(h, i, pid);
    }

    /* Pass 2: only if force-reset will fire.  Issue the rwlock force-
     * reset CAS FIRST, while the window since pass 1's last scan is
     * still narrow (a handful of instructions, as in the original
     * single-pass code).  A new reader that started rdlock between
     * pass 1's scan and the CAS will either:
     *   (a) have already CAS'd rwlock from cur to cur+1 -- our CAS then
     *       fails (cur mismatched), recovery yields and a future
     *       cycle retries; or
     *   (b) be still in the subcount-bump phase -- our CAS sees the
     *       stale cur and resets to 0; the new reader's subsequent CAS
     *       rwlock(0 -> 1) succeeds cleanly.
     * Only after the CAS resolves do we wipe the deferred dead slots,
     * keeping that work outside the race-sensitive window. */
    /* A live reader with no slot (table was full) is invisible to the scan
     * above but still holds a +1 in the lock word; never force-reset under it. */
    if (__atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0)
        any_live_reader = 1;
    if (found_dead_reader && !any_live_reader) {
        /* ACQUIRE: a late reader's subcount++ (before its rwlock CAS) is then visible below. */
        uint32_t cur = __atomic_load_n(&hdr->rwlock, __ATOMIC_ACQUIRE);
        int drain_ok = 1;   /* keep dead slots if the reset doesn't fire */
        if (cur > 0 && cur < TK_RWLOCK_WRITER_BIT) {
            /* Re-scan for a live reader (fail-safe: only suppresses a reset). */
            int live_now = __atomic_load_n(&hdr->slotless_readers, __ATOMIC_RELAXED) > 0;
            for (uint32_t i = 0; !live_now && i < TK_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p && tk_pid_alive(p) &&
                    __atomic_load_n(&h->reader_slots[i].subcount, __ATOMIC_RELAXED) > 0)
                    live_now = 1;
            }
            if (live_now) {
                drain_ok = 0;
            } else if (__atomic_compare_exchange_n(&hdr->rwlock, &cur, 0,
                    0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
                if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
                    syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
            } else {
                drain_ok = 0;   /* rwlock changed under us -- shares may still be live */
            }
        }
        if (drain_ok) {
            for (uint32_t i = 0; i < TK_READER_SLOTS; i++) {
                uint32_t p = __atomic_load_n(&h->reader_slots[i].pid, __ATOMIC_ACQUIRE);
                if (p == 0 || tk_pid_alive(p)) continue;
                tk_drain_dead_slot(h, i, p);
            }
        }
    }
}

/* Inspect the lock word after a futex-wait timeout.  If a dead writer
 * holds it, force-recover the lock.  Otherwise drain dead readers' shares
 * of the rwlock/waiter counters.  Called from rdlock and wrlock ETIMEDOUT
 * branches -- identical recovery logic in both. */
static inline void tk_recover_after_timeout(TkHandle *h) {
    TkHeader *hdr = h->hdr;
    uint32_t val = __atomic_load_n(&hdr->rwlock, __ATOMIC_RELAXED);
    if (val >= TK_RWLOCK_WRITER_BIT) {
        uint32_t pid = val & TK_RWLOCK_PID_MASK;
        if (!tk_pid_alive(pid))
            tk_recover_stale_lock(h, val);
    } else {
        tk_recover_dead_readers(h);
    }
}

/* Park/unpark helpers: bump the global waiter counters together with this
 * process's mirrored slot counters so a wrlock-timeout recovery scan can
 * attribute and reverse a dead PID's contribution.  Kept paired to make
 * accidental drift between global and per-slot counts impossible. */
static inline void tk_park_reader(TkHandle *h) {
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
}
static inline void tk_unpark_reader(TkHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX)
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
}
static inline void tk_park_writer(TkHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
}
static inline void tk_unpark_writer(TkHandle *h) {
    __atomic_sub_fetch(&h->hdr->rwlock_waiters, 1, __ATOMIC_RELAXED);
    __atomic_sub_fetch(&h->hdr->rwlock_writers_waiting, 1, __ATOMIC_RELAXED);
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].waiters_parked, 1, __ATOMIC_RELAXED);
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].writers_parked, 1, __ATOMIC_RELAXED);
    }
}

/* Reader accounting: a reader mirrors its +1 in the lock word so dead-reader
 * recovery can see it. A slotted reader uses its slot subcount; a reader that
 * could not claim a slot (table full) uses the global hdr->slotless_readers,
 * so recovery's force-reset never fires out from under it. leave() peels
 * slotless first so a later slot claim cannot misattribute the decrement. */
static inline void tk_reader_enter(TkHandle *h) {
    if (h->my_slot_idx != UINT32_MAX) {
        __atomic_add_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
        h->slotless_held++;
    }
}
static inline void tk_reader_leave(TkHandle *h) {
    if (h->slotless_held > 0) {
        h->slotless_held--;
        __atomic_sub_fetch(&h->hdr->slotless_readers, 1, __ATOMIC_RELAXED);
    } else if (h->my_slot_idx != UINT32_MAX) {
        __atomic_sub_fetch(&h->reader_slots[h->my_slot_idx].subcount, 1, __ATOMIC_RELAXED);
    }
}

static inline void tk_rwlock_rdlock(TkHandle *h) {
    tk_claim_reader_slot(h);
    TkHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    uint32_t *writers_waiting = &hdr->rwlock_writers_waiting;
    /* Claim subcount BEFORE bumping the shared rwlock counter.  This way
     * a concurrent writer-side recovery scan that sees our PID alive with
     * subcount > 0 will (correctly) defer force-reset, even while we are
     * still spinning trying to win the rwlock CAS.  Without this, a reader
     * killed between rwlock CAS-success and subcount++ would let recovery
     * force-reset rwlock to 0 underneath us, causing a UINT32_MAX wrap on
     * our eventual rdunlock dec. */
    tk_reader_enter(h);
    for (int spin = 0; ; spin++) {
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Write-preferring: when lock is free (cur==0) and writers are
         * waiting, yield to let the writer acquire. When readers are
         * already active (cur>=1), new readers may join freely. */
        if (cur > 0 && cur < TK_RWLOCK_WRITER_BIT) {
            if (__atomic_compare_exchange_n(lock, &cur, cur + 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        } else if (cur == 0 && !__atomic_load_n(writers_waiting, __ATOMIC_RELAXED)) {
            if (__atomic_compare_exchange_n(lock, &cur, 1,
                    1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
                return;
        }
        if (__builtin_expect(spin < TK_RWLOCK_SPIN_LIMIT, 1)) {
            tk_rwlock_spin_pause();
            continue;
        }
        tk_park_reader(h);
        cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        /* Sleep when write-locked OR when yielding to waiting writers */
        if (cur >= TK_RWLOCK_WRITER_BIT || cur == 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &tk_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                tk_unpark_reader(h);
                tk_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        tk_unpark_reader(h);
        spin = 0;
    }
}

static inline void tk_rwlock_rdunlock(TkHandle *h) {
    TkHeader *hdr = h->hdr;
    /* Release the shared counter BEFORE dropping our subcount so that
     * "any live PID with subcount > 0" is a reliable in-flight indicator
     * for the writer-side recovery scan.  Inverting these would create a
     * window where we still own a unit of rwlock but our slot subcount is
     * 0, letting recovery force-reset rwlock underneath us. */
    uint32_t after = __atomic_sub_fetch(&hdr->rwlock, 1, __ATOMIC_RELEASE);
    tk_reader_leave(h);
    if (after == 0 && __atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static inline void tk_rwlock_wrlock(TkHandle *h) {
    tk_claim_reader_slot(h);  /* refresh cached_pid across fork */
    TkHeader *hdr = h->hdr;
    uint32_t *lock = &hdr->rwlock;
    /* Encode PID in the rwlock word itself (0x80000000 | pid) to eliminate
     * any crash window between acquiring the lock and storing the owner. */
    uint32_t mypid = TK_RWLOCK_WR(h->cached_pid);
    for (int spin = 0; ; spin++) {
        uint32_t expected = 0;
        if (__atomic_compare_exchange_n(lock, &expected, mypid,
                1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            return;
        if (__builtin_expect(spin < TK_RWLOCK_SPIN_LIMIT, 1)) {
            tk_rwlock_spin_pause();
            continue;
        }
        tk_park_writer(h);
        uint32_t cur = __atomic_load_n(lock, __ATOMIC_RELAXED);
        if (cur != 0) {
            long rc = syscall(SYS_futex, lock, FUTEX_WAIT, cur,
                              &tk_lock_timeout, NULL, 0);
            if (rc == -1 && errno == ETIMEDOUT) {
                tk_unpark_writer(h);
                tk_recover_after_timeout(h);
                spin = 0;
                continue;
            }
        }
        tk_unpark_writer(h);
        spin = 0;
    }
}

static inline void tk_rwlock_wrunlock(TkHandle *h) {
    TkHeader *hdr = h->hdr;
    __atomic_store_n(&hdr->rwlock, 0, __ATOMIC_RELEASE);
    if (__atomic_load_n(&hdr->rwlock_waiters, __ATOMIC_RELAXED) > 0)
        syscall(SYS_futex, &hdr->rwlock, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

/* ================================================================
 * Layout math + create / open / destroy
 *
 * Layout: Header -> reader_slots[1024] -> slots[m] -> heap[m] -> buckets[nb]
 *   slots[]   : m records of (count, error, key); stride = sizeof(TkSlot)+align8(key_size)
 *   heap[]    : m uint32 slot-indices, a min-heap ordered by slot count
 *   buckets[] : nb uint32 slot-indices (or TK_NIL), an intrusive hash index
 * ================================================================ */

#define TK_SLOT_OK(h, i) ((uint64_t)(i) < (uint64_t)(h)->capacity)

/* Single source of truth for the mmap region layout offsets. */
typedef struct { uint64_t reader_slots, slots, heap, buckets, total; } TkLayout;

/* per-slot byte stride for a given key_size (keeps each slot 8-aligned) */
static inline uint64_t tk_stride(uint32_t key_size) {
    return (uint64_t)sizeof(TkSlot) + (((uint64_t)key_size + 7) & ~(uint64_t)7);
}

/* number of hash buckets for m counters: next_pow2(2*m), floor 8 (load <= 0.5) */
static inline uint64_t tk_buckets_for(uint64_t m) {
    uint64_t want = m * 2;
    if (want < 8) want = 8;
    if ((want & (want - 1)) == 0) return want;
    return 1ULL << (64 - __builtin_clzll(want));
}

static inline TkLayout tk_layout_for(uint32_t m, uint32_t key_size) {
    TkLayout L;
    uint64_t stride = tk_stride(key_size);
    L.reader_slots = sizeof(TkHeader);
    L.slots        = L.reader_slots + (uint64_t)TK_READER_SLOTS * sizeof(TkReaderSlot);
    L.slots        = (L.slots + 7) & ~(uint64_t)7;
    L.heap         = L.slots + (uint64_t)m * stride;
    L.heap         = (L.heap + 7) & ~(uint64_t)7;
    L.buckets      = L.heap + (uint64_t)m * sizeof(uint32_t);
    L.buckets      = (L.buckets + 7) & ~(uint64_t)7;
    L.total        = L.buckets + tk_buckets_for(m) * sizeof(uint32_t);
    return L;
}

static inline uint64_t tk_total_size(uint32_t m, uint32_t key_size) {
    return tk_layout_for(m, key_size).total;
}

static inline void tk_init_header(void *base, uint32_t m, uint32_t key_size, uint32_t mode, double alpha, uint64_t total) {
    TkLayout L = tk_layout_for(m, key_size);
    TkHeader *hdr = (TkHeader *)base;
    /* Zero header + reader-slots + slots + heap, then set every hash bucket to
       TK_NIL (0xFF bytes) so the table starts empty. */
    memset(base, 0, (size_t)L.buckets);
    memset((char *)base + L.buckets, 0xFF, (size_t)(L.total - L.buckets));   /* buckets = TK_NIL */
    hdr->magic            = TK_MAGIC;
    hdr->version          = TK_VERSION;
    hdr->capacity         = m;
    hdr->key_size         = key_size;
    hdr->hash_buckets     = tk_buckets_for(m);
    hdr->used             = 0;
    hdr->seen             = 0;
    hdr->slots_off        = L.slots;
    hdr->total_size       = total;
    hdr->reader_slots_off = L.reader_slots;
    hdr->heap_off         = L.heap;
    hdr->bucket_off       = L.buckets;
    hdr->mode             = mode;
    hdr->alpha            = alpha;
    hdr->now              = 0.0;
    hdr->landmark         = 0.0;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

/* ---- accessors ---- */
static inline TkSlot *tk_slot(TkHandle *h, uint64_t i) {
    return (TkSlot *)((char *)h->base + h->slots_off + i * h->stride);
}
static inline char *tk_slot_key(TkSlot *s) { return (char *)s + sizeof(TkSlot); }
static inline uint32_t *tk_heap(TkHandle *h) {
    return (uint32_t *)((char *)h->base + h->heap_off);
}
static inline uint32_t *tk_buckets(TkHandle *h) {
    return (uint32_t *)((char *)h->base + h->bucket_off);
}

/* Layer B trusted bound: the number of slots guaranteed to lie within the real
 * mapping.  Derived from the process-local mmap_size (fixed at attach, not
 * peer-writable) and the SAME slots_off/stride the accessors use, so a corrupt
 * hdr->capacity can never drive a slot access outside the mapping.  Equals m
 * for a valid table; every clamp below it is a never-taken branch in normal use. */
static inline uint64_t tk_slots_max(TkHandle *h) {
    if (h->slots_off >= h->mmap_size || h->stride == 0) return 0;
    return (h->mmap_size - h->slots_off) / h->stride;
}

static inline TkHandle *tk_setup(void *base, size_t map_size,
                                 const char *path, int backing_fd) {
    TkHeader *hdr = (TkHeader *)base;
    TkHandle *h = (TkHandle *)calloc(1, sizeof(TkHandle));
    if (!h) {
        munmap(base, map_size);
        if (backing_fd >= 0) close(backing_fd);
        return NULL;
    }
    h->hdr          = hdr;
    h->base         = base;
    h->reader_slots = (TkReaderSlot *)((uint8_t *)base + hdr->reader_slots_off);
    /* single validated read of each geometry field; cached so the process-local
       bound and the pointers it feeds stay consistent even if a peer later
       corrupts the shared header. */
    h->slots_off    = hdr->slots_off;
    h->heap_off     = hdr->heap_off;
    h->bucket_off   = hdr->bucket_off;
    h->capacity     = hdr->capacity;
    h->key_size     = hdr->key_size;
    h->hash_mask    = hdr->hash_buckets - 1;
    h->stride       = tk_stride(hdr->key_size);
    h->mode         = hdr->mode;
    h->alpha        = hdr->alpha;
    h->mmap_size    = map_size;
    /* Layer B: if the mapping cannot even hold `capacity` slots the header lied
       about its size; clamp the cached capacity to what actually fits.  (Uses
       h->mmap_size + h->slots_off + h->stride, all set just above.) */
    {
        uint64_t fit = tk_slots_max(h);
        if ((uint64_t)h->capacity > fit) h->capacity = (uint32_t)fit;
    }
    h->path         = path ? strdup(path) : NULL;
    h->backing_fd   = backing_fd;
    h->my_slot_idx  = UINT32_MAX;
    return h;
}

/* Validate a mapped header (shared by tk_create reopen and tk_open_fd). */
static inline int tk_validate_header(const TkHeader *hdr, uint64_t file_size) {
    if (hdr->magic != TK_MAGIC) return 0;
    if (hdr->version != TK_VERSION) return 0;
    if (hdr->mode != TK_MODE_PLAIN && hdr->mode != TK_MODE_DECAYED) return 0;
    if (hdr->mode == TK_MODE_DECAYED && !(hdr->alpha > 0.0 && isfinite(hdr->alpha))) return 0;
    if (hdr->mode == TK_MODE_DECAYED && (!isfinite(hdr->now) || !isfinite(hdr->landmark))) return 0;
    if (hdr->capacity < TK_MIN_CAP || hdr->capacity > TK_MAX_CAP) return 0;
    if (hdr->key_size < TK_MIN_KEYSIZE || hdr->key_size > TK_MAX_KEYSIZE) return 0;
    if (hdr->hash_buckets != tk_buckets_for(hdr->capacity)) return 0;
    if (hdr->used > hdr->capacity) return 0;
    if (hdr->total_size != file_size) return 0;
    if (hdr->total_size != tk_total_size((uint32_t)hdr->capacity, hdr->key_size)) return 0;
    TkLayout L = tk_layout_for((uint32_t)hdr->capacity, hdr->key_size);
    if (hdr->reader_slots_off != L.reader_slots) return 0;
    if (hdr->slots_off != L.slots) return 0;
    if (hdr->heap_off != L.heap) return 0;
    if (hdr->bucket_off != L.buckets) return 0;
    return 1;
}

/* validate the requested capacity + key_size */
static int tk_validate_args(uint64_t capacity, uint64_t key_size, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    if (capacity < TK_MIN_CAP || capacity > TK_MAX_CAP) { TK_ERR("capacity must be between 1 and 2^24"); return 0; }
    if (key_size < TK_MIN_KEYSIZE || key_size > TK_MAX_KEYSIZE) { TK_ERR("key_size must be between 1 and 4096"); return 0; }
    return 1;
}

/* Securely obtain a fd for a path-backed segment: create it exclusively
 * (O_CREAT|O_EXCL|O_NOFOLLOW at `mode`, default 0600 = owner-only), or, if it
 * already exists, attach to it (O_RDWR|O_NOFOLLOW, no O_CREAT). O_EXCL blocks a
 * pre-seeded or hard-linked file and O_NOFOLLOW a symlink swap, so a local
 * attacker can no longer redirect or poison the backing store through the path.
 * Cross-user sharing is opt-in via a wider `mode` (e.g. 0660); the caller still
 * validates the file's contents via tk_validate_header. */
static int tk_secure_open(const char *path, mode_t mode, char *errbuf) {
    for (int attempt = 0; attempt < 100; attempt++) {
        int fd = open(path, O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, mode);
        if (fd >= 0) { (void)fchmod(fd, mode); return fd; }   /* exact mode: umask narrowed the O_EXCL create */
        if (errno != EEXIST) { TK_ERR("create %s: %s", path, strerror(errno)); return -1; }
        fd = open(path, O_RDWR|O_NOFOLLOW|O_CLOEXEC);
        if (fd >= 0) return fd;
        if (errno == ENOENT) continue;   /* creator unlinked between our two opens; retry */
        TK_ERR("open %s: %s", path, strerror(errno));  /* ELOOP => symlink rejected */
        return -1;
    }
    TK_ERR("open %s: create/attach kept racing", path);
    return -1;
}

static TkHandle *tk_create(const char *path, uint64_t capacity, uint64_t key_size, uint32_t tkmode, double alpha, mode_t mode, char *errbuf) {
    if (!tk_validate_args(capacity, key_size, errbuf)) return NULL;

    uint64_t total = tk_total_size((uint32_t)capacity, (uint32_t)key_size);
    int anonymous = (path == NULL);
    int fd = -1;
    size_t map_size;
    void *base;

    if (anonymous) {
        map_size = (size_t)total;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) { TK_ERR("mmap: %s", strerror(errno)); return NULL; }
    } else {
        fd = tk_secure_open(path, mode, errbuf);
        if (fd < 0) return NULL;
        if (flock(fd, LOCK_EX) < 0) { TK_ERR("flock: %s", strerror(errno)); close(fd); return NULL; }
        struct stat st;
        if (fstat(fd, &st) < 0) { TK_ERR("fstat: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        int is_new = (st.st_size == 0);
        if (!is_new && (uint64_t)st.st_size < sizeof(TkHeader)) {
            TK_ERR("%s: file too small (%lld)", path, (long long)st.st_size);
            flock(fd, LOCK_UN); close(fd); return NULL;
        }
        if (is_new && ftruncate(fd, (off_t)total) < 0) {
            TK_ERR("ftruncate: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL;
        }
        map_size = is_new ? (size_t)total : (size_t)st.st_size;
        base = mmap(NULL, map_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) { TK_ERR("mmap: %s", strerror(errno)); flock(fd, LOCK_UN); close(fd); return NULL; }
        if (!is_new) {
            if (!tk_validate_header((TkHeader *)base, (uint64_t)st.st_size)) {
                TK_ERR("invalid top-k table file"); munmap(base, map_size); flock(fd, LOCK_UN); close(fd); return NULL;
            }
            flock(fd, LOCK_UN); close(fd);
            return tk_setup(base, map_size, path, -1);
        }
    }
    tk_init_header(base, (uint32_t)capacity, (uint32_t)key_size, tkmode, alpha, total);
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
    return tk_setup(base, map_size, path, -1);
}

static TkHandle *tk_create_memfd(const char *name, uint64_t capacity, uint64_t key_size, uint32_t tkmode, double alpha, char *errbuf) {
    if (!tk_validate_args(capacity, key_size, errbuf)) return NULL;

    uint64_t total = tk_total_size((uint32_t)capacity, (uint32_t)key_size);
    int fd = memfd_create(name ? name : "topk", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd < 0) { TK_ERR("memfd_create: %s", strerror(errno)); return NULL; }
    if (ftruncate(fd, (off_t)total) < 0) {
        TK_ERR("ftruncate: %s", strerror(errno)); close(fd); return NULL;
    }
    (void)fcntl(fd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW);
    void *base = mmap(NULL, (size_t)total, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { TK_ERR("mmap: %s", strerror(errno)); close(fd); return NULL; }
    tk_init_header(base, (uint32_t)capacity, (uint32_t)key_size, tkmode, alpha, total);
    return tk_setup(base, (size_t)total, NULL, fd);
}

static TkHandle *tk_open_fd(int fd, char *errbuf) {
    if (errbuf) errbuf[0] = '\0';
    struct stat st;
    if (fstat(fd, &st) < 0) { TK_ERR("fstat: %s", strerror(errno)); return NULL; }
    if ((uint64_t)st.st_size < sizeof(TkHeader)) { TK_ERR("too small"); return NULL; }
    size_t ms = (size_t)st.st_size;
    void *base = mmap(NULL, ms, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { TK_ERR("mmap: %s", strerror(errno)); return NULL; }
    if (!tk_validate_header((TkHeader *)base, (uint64_t)st.st_size)) {
        TK_ERR("invalid top-k table"); munmap(base, ms); return NULL;
    }
    int myfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (myfd < 0) { TK_ERR("fcntl: %s", strerror(errno)); munmap(base, ms); return NULL; }
    return tk_setup(base, ms, NULL, myfd);
}

static void tk_destroy(TkHandle *h) {
    if (!h) return;
    /* Release our reader slot on clean teardown (else short-lived-reader churn
     * exhausts the slot table); skip if a lock is still held (subcount>0). */
    if (h->reader_slots && h->my_slot_idx != UINT32_MAX && h->cached_pid &&
        h->cached_fork_gen == __atomic_load_n(&tk_fork_gen, __ATOMIC_RELAXED) &&
        __atomic_load_n(&h->reader_slots[h->my_slot_idx].subcount, __ATOMIC_ACQUIRE) == 0) {
        uint32_t expected = h->cached_pid;
        __atomic_compare_exchange_n(&h->reader_slots[h->my_slot_idx].pid,
                &expected, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
    }
    if (h->backing_fd >= 0) close(h->backing_fd);
    if (h->base) munmap(h->base, h->mmap_size);
    free(h->path);
    free(h);
}

static inline int tk_msync(TkHandle *h) {
    if (!h || !h->base) return 0;
    return msync(h->base, h->mmap_size, MS_SYNC);
}

/* ================================================================
 * Space-Saving operations (callers hold the lock)
 *
 * A fixed set of m counters monitors the heaviest keys.  Observing a key that
 * is already monitored bumps its count; observing a new key when a counter is
 * free fills it; observing a new key when full evicts the minimum-count counter,
 * inheriting its count as the new key's error bound.  A min-heap (over counts)
 * finds the eviction victim in O(log m); an intrusive hash index (chaining via
 * slot->hnext) tests membership in O(1).  Keys are compared by their first
 * key_size bytes (longer keys are truncated on the way in).
 * ================================================================ */

static inline uint64_t tk_hash(const void *key, size_t len) {
    return XXH3_64bits(key, len);
}

/* ---- forward-decay helpers (decayed mode stores a double's bits in count/error) ---- */
static inline double   tk_w_get(uint64_t bits) { double d;   memcpy(&d, &bits, sizeof d); return d; }
static inline uint64_t tk_w_bits(double d)     { uint64_t b; memcpy(&b, &d,   sizeof b); return b; }
/* forward-decay factor g(now) = exp(alpha*(now - landmark)); >= 1, == 1 just after a rescale */
static inline double tk_g(TkHandle *h) {
    return exp(h->alpha * (h->hdr->now - h->hdr->landmark));
}
/* mode-aware count comparisons for the min-heap */
static inline int tk_slot_lt(TkHandle *h, const TkSlot *a, const TkSlot *b) {
    if (h->mode == TK_MODE_DECAYED) return tk_w_get(a->count) < tk_w_get(b->count);
    return a->count < b->count;
}
static inline int tk_slot_le(TkHandle *h, const TkSlot *a, const TkSlot *b) {
    if (h->mode == TK_MODE_DECAYED) return tk_w_get(a->count) <= tk_w_get(b->count);
    return a->count <= b->count;
}

/* ---- min-heap over slot counts (heap[] holds slot indices) ---- */

static inline void tk_heap_swap(TkHandle *h, uint32_t *heap, uint64_t a, uint64_t b) {
    uint32_t sa = heap[a], sb = heap[b];
    heap[a] = sb; heap[b] = sa;
    if (TK_SLOT_OK(h, sa)) tk_slot(h, sa)->heap_pos = (uint32_t)b;
    if (TK_SLOT_OK(h, sb)) tk_slot(h, sb)->heap_pos = (uint32_t)a;
}

static void tk_sift_up(TkHandle *h, uint32_t *heap, uint64_t pos) {
    while (pos > 0) {
        uint64_t parent = (pos - 1) / 2;
        uint32_t sp = heap[parent], sc = heap[pos];
        if (!TK_SLOT_OK(h, sp) || !TK_SLOT_OK(h, sc)) break;      /* Layer B */
        if (tk_slot_le(h, tk_slot(h, sp), tk_slot(h, sc))) break;
        tk_heap_swap(h, heap, parent, pos);
        pos = parent;
    }
}

static void tk_sift_down(TkHandle *h, uint32_t *heap, uint64_t size, uint64_t pos) {
    for (;;) {
        uint64_t l = 2 * pos + 1, r = 2 * pos + 2, smallest = pos;
        if (l < size) {
            uint32_t sl = heap[l], ss = heap[smallest];
            if (TK_SLOT_OK(h, sl) && TK_SLOT_OK(h, ss) &&
                tk_slot_lt(h, tk_slot(h, sl), tk_slot(h, ss))) smallest = l;
        }
        if (r < size) {
            uint32_t sr = heap[r], ss = heap[smallest];
            if (TK_SLOT_OK(h, sr) && TK_SLOT_OK(h, ss) &&
                tk_slot_lt(h, tk_slot(h, sr), tk_slot(h, ss))) smallest = r;
        }
        if (smallest == pos) break;
        tk_heap_swap(h, heap, pos, smallest);
        pos = smallest;
    }
}

/* current heap size, clamped to the capacity (Layer B: hdr->used is peer-writable) */
static inline uint64_t tk_heap_size(TkHandle *h) {
    uint64_t used = h->hdr->used;
    return (used > h->capacity) ? h->capacity : used;
}

/* ---- intrusive hash index (chaining via slot->hnext) ---- */

/* find the slot monitoring `key` (klen already truncated), or TK_NIL */
static uint32_t tk_hash_find(TkHandle *h, const void *key, uint32_t klen, uint64_t hv) {
    uint32_t *buckets = tk_buckets(h);
    uint32_t s = buckets[hv & h->hash_mask];
    uint64_t guard = 0;
    while (s != TK_NIL && TK_SLOT_OK(h, s) && guard++ <= (uint64_t)h->capacity) {
        TkSlot *sl = tk_slot(h, s);
        uint32_t kl = sl->key_len; if (kl > h->key_size) kl = h->key_size;     /* Layer B */
        if (kl == klen && memcmp(tk_slot_key(sl), key, klen) == 0) return s;
        s = sl->hnext;
    }
    return TK_NIL;
}

/* prepend slot s (key already stored) to its hash chain */
static void tk_hash_insert(TkHandle *h, uint32_t s, uint64_t hv) {
    uint32_t *buckets = tk_buckets(h);
    uint64_t b = hv & h->hash_mask;
    tk_slot(h, s)->hnext = buckets[b];
    buckets[b] = s;
}

/* unlink slot s from its hash chain (hv = hash of its stored key) */
static void tk_hash_remove(TkHandle *h, uint32_t s, uint64_t hv) {
    uint32_t *buckets = tk_buckets(h);
    uint64_t b = hv & h->hash_mask;
    uint32_t cur = buckets[b];
    if (cur == s) { buckets[b] = tk_slot(h, s)->hnext; return; }
    uint64_t guard = 0;
    while (cur != TK_NIL && TK_SLOT_OK(h, cur) && guard++ <= (uint64_t)h->capacity) {
        TkSlot *c = tk_slot(h, cur);
        if (c->hnext == s) { c->hnext = tk_slot(h, s)->hnext; return; }
        cur = c->hnext;
    }
}

/* store (truncated) key bytes into slot s; returns the stored length */
static inline uint32_t tk_store_key(TkHandle *h, uint32_t s, const void *key, size_t len) {
    uint32_t klen = (len > h->key_size) ? h->key_size : (uint32_t)len;
    TkSlot *sl = tk_slot(h, s);
    memcpy(tk_slot_key(sl), key, klen);
    sl->key_len = klen;
    return klen;
}

/* Rescale all weights by 1/g and advance the landmark to now, keeping the
 * forward-decay weights bounded (decayed mode; caller holds the write lock). */
static void tk_rescale(TkHandle *h) {
    double g = tk_g(h);
    if (!(g > 1.0)) { h->hdr->landmark = h->hdr->now; return; }   /* g <= 1 or NaN: nothing to rescale */
    double inv = isfinite(g) ? 1.0 / g : 0.0;   /* g == +Inf: a huge time gap fully decayed all old weight -> 0 */
    uint64_t used = tk_heap_size(h);
    uint64_t smax = tk_slots_max(h);
    if (used > smax) used = smax;                     /* Layer B */
    for (uint64_t i = 0; i < used; i++) {
        TkSlot *sl = tk_slot(h, i);
        sl->count = tk_w_bits(tk_w_get(sl->count) * inv);
        sl->error = tk_w_bits(tk_w_get(sl->error) * inv);
    }
    h->hdr->landmark = h->hdr->now;                   /* g(now) == 1 again */
}

/* Observe one item; returns the item's estimated count after this observation.
 * (caller holds the write lock) */
static uint64_t tk_observe_locked(TkHandle *h, const void *item, size_t len, int has_ts, double ts) {
    if (h->capacity == 0) return 0;                        /* Layer B: unusable mapping */
    uint32_t klen = (len > h->key_size) ? h->key_size : (uint32_t)len;
    uint64_t hv = tk_hash(item, klen);
    h->hdr->seen++;

    double g = 1.0;                                        /* increment weight (1 in plain mode) */
    if (h->mode == TK_MODE_DECAYED) {
        /* advance the monotonic clock: to ts if given and larger, else one tick */
        if (has_ts) { if (ts > h->hdr->now) h->hdr->now = ts; }
        else h->hdr->now += 1.0;
        if (h->alpha * (h->hdr->now - h->hdr->landmark) > TK_RESCALE_LN)
            tk_rescale(h);                                 /* advance the landmark to keep g bounded */
        g = tk_g(h);
    }

    uint32_t s = tk_hash_find(h, item, klen, hv);
    uint32_t *heap = tk_heap(h);
    if (s != TK_NIL) {                                     /* already monitored: bump */
        TkSlot *sl = tk_slot(h, s);
        if (h->mode == TK_MODE_DECAYED) sl->count = tk_w_bits(tk_w_get(sl->count) + g);
        else                            sl->count++;
        uint64_t size = tk_heap_size(h);
        uint64_t pos = sl->heap_pos;
        if (pos < size) tk_sift_down(h, heap, size, pos); /* count rose -> sink it */
        return sl->count;
    }

    if (h->hdr->used < h->capacity) {                     /* warm-up: fill a fresh slot */
        uint32_t ns = (uint32_t)h->hdr->used;
        TkSlot *sl = tk_slot(h, ns);
        tk_store_key(h, ns, item, len);
        sl->count = (h->mode == TK_MODE_DECAYED) ? tk_w_bits(g)   : 1;
        sl->error = (h->mode == TK_MODE_DECAYED) ? tk_w_bits(0.0) : 0;
        sl->hnext = TK_NIL;
        sl->heap_pos = ns;
        heap[ns] = ns;
        tk_hash_insert(h, ns, hv);
        h->hdr->used++;
        tk_sift_up(h, heap, ns);
        return sl->count;
    }

    /* full: evict the minimum-count slot (heap root) */
    uint32_t victim = heap[0];
    if (!TK_SLOT_OK(h, victim)) return 0;                 /* Layer B: corrupt heap root */
    TkSlot *sl = tk_slot(h, victim);
    uint32_t old_kl = sl->key_len; if (old_kl > h->key_size) old_kl = h->key_size;
    uint64_t old_hv = tk_hash(tk_slot_key(sl), old_kl);
    tk_hash_remove(h, victim, old_hv);                    /* drop the evicted key */
    tk_store_key(h, victim, item, len);
    if (h->mode == TK_MODE_DECAYED) {
        double min_w = tk_w_get(sl->count);               /* victim's weight (heap minimum) */
        sl->error = tk_w_bits(min_w);                     /* over-estimate bound */
        sl->count = tk_w_bits(min_w + g);
    } else {
        uint64_t min_count = sl->count;
        sl->error = min_count;
        sl->count = min_count + 1;
    }
    tk_hash_insert(h, victim, hv);
    tk_sift_down(h, heap, h->capacity, 0);                /* root's count rose -> sink it */
    return sl->count;
}

/* estimated count of `item` (0 if not monitored); *err_out gets the over-estimate
 * bound if non-NULL.  (caller holds a lock) */
static uint64_t tk_estimate_locked(TkHandle *h, const void *item, size_t len, uint64_t *err_out) {
    uint32_t klen = (len > h->key_size) ? h->key_size : (uint32_t)len;
    uint64_t hv = tk_hash(item, klen);
    uint32_t s = tk_hash_find(h, item, klen, hv);
    if (s == TK_NIL) { if (err_out) *err_out = 0; return 0; }
    TkSlot *sl = tk_slot(h, s);
    if (err_out) *err_out = sl->error;
    return sl->count;
}

/* reset to an empty table (caller holds the write lock) */
static inline void tk_clear_locked(TkHandle *h) {
    h->hdr->used = 0;
    h->hdr->seen = 0;
    uint64_t nb = h->hash_mask + 1;
    memset(tk_buckets(h), 0xFF, (size_t)(nb * sizeof(uint32_t)));   /* all buckets -> TK_NIL */
}

#endif /* TK_H */
