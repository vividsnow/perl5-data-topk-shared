#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "ppport.h"
#include "topk.h"

#define EXTRACT(sv) \
    if (!sv_isobject(sv) || !sv_derived_from(sv, "Data::TopK::Shared")) \
        croak("Expected a Data::TopK::Shared object"); \
    TkHandle *h = INT2PTR(TkHandle*, SvIV(SvRV(sv))); \
    if (!h) croak("Attempted to use a destroyed Data::TopK::Shared object")

#define MAKE_OBJ(class, handle) \
    SV *obj = newSViv(PTR2IV(handle)); \
    SV *ref = newRV_noinc(obj); \
    sv_bless(ref, gv_stashpv(class, GV_ADD)); \
    RETVAL = ref

/* one snapshotted monitored entry, for sorting top() results.
 * off is a byte offset into the key blob (used * key_size can exceed 4 GiB), so
 * it must be 64-bit; klen <= key_size <= 4096 fits 32-bit. */
typedef struct { uint64_t count, error, off; uint32_t klen; } TkEnt;

/* sort by count descending, then by error ascending (tighter estimate first) */
static int tk_ent_cmp(const void *a, const void *b) {
    const TkEnt *x = (const TkEnt *)a, *y = (const TkEnt *)b;
    if (x->count > y->count) return -1;
    if (x->count < y->count) return  1;
    if (x->error < y->error) return -1;
    if (x->error > y->error) return  1;
    return 0;
}

MODULE = Data::TopK::Shared  PACKAGE = Data::TopK::Shared

PROTOTYPES: DISABLE

SV *
new(class, path = &PL_sv_undef, capacity = 0, key_size = 256, ...)
    const char *class
    SV *path
    UV capacity
    UV key_size
  PREINIT:
    char errbuf[TK_ERR_BUFLEN];
  CODE:
    const char *p = (SvGETMAGIC(path), SvOK(path)) ? SvPV_nolen(path) : NULL;
    if (capacity < 1)
        croak("Data::TopK::Shared->new: capacity must be >= 1");
    /* Optional 5th arg: file mode for a newly-created file-backed segment
     * (default 0600, owner-only). Pass e.g. 0660 for cross-user sharing. */
    mode_t mode = (items > 4 && (SvGETMAGIC(ST(4)), SvOK(ST(4)))) ? (mode_t)SvUV(ST(4)) : 0600;
    TkHandle *h = tk_create(p, (uint64_t)capacity, (uint64_t)key_size, mode, errbuf);
    if (!h) croak("Data::TopK::Shared->new: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

SV *
new_memfd(class, name = &PL_sv_undef, capacity = 0, key_size = 256)
    const char *class
    SV *name
    UV capacity
    UV key_size
  PREINIT:
    char errbuf[TK_ERR_BUFLEN];
  CODE:
    const char *nm = (SvGETMAGIC(name), SvOK(name)) ? SvPV_nolen(name) : NULL;   /* undef -> default label */
    if (capacity < 1)
        croak("Data::TopK::Shared->new_memfd: capacity must be >= 1");
    TkHandle *h = tk_create_memfd(nm, (uint64_t)capacity, (uint64_t)key_size, errbuf);
    if (!h) croak("Data::TopK::Shared->new_memfd: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

SV *
new_from_fd(class, fd)
    const char *class
    int fd
  PREINIT:
    char errbuf[TK_ERR_BUFLEN];
  CODE:
    TkHandle *h = tk_open_fd(fd, errbuf);
    if (!h) croak("Data::TopK::Shared->new_from_fd: %s", errbuf);
    MAKE_OBJ(class, h);
  OUTPUT:
    RETVAL

void
DESTROY(self)
    SV *self
  CODE:
    if (sv_isobject(self) && sv_derived_from(self, "Data::TopK::Shared")) {
        TkHandle *h = INT2PTR(TkHandle*, SvIV(SvRV(self)));
        if (h) { sv_setiv(SvRV(self), 0); tk_destroy(h); }   /* null first: activates EXTRACT's use-after-destroy croak + makes a double DESTROY a no-op */
    }

UV
add(self, item)
    SV *self
    SV *item
  PREINIT:
    EXTRACT(self);
    STRLEN n;
    const char *s;
  CODE:
    s = SvPVbyte(item, n);                 /* may croak (wide char) -- BEFORE the lock */
    tk_rwlock_wrlock(h);
    RETVAL = (UV)tk_observe_locked(h, s, n);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    tk_rwlock_wrunlock(h);
  OUTPUT:
    RETVAL

UV
add_many(self, items)
    SV *self
    SV *items
  PREINIT:
    EXTRACT(self);
    AV *av;
    IV  top;
    UV  processed = 0;
  CODE:
    if (!SvROK(items) || SvTYPE(SvRV(items)) != SVt_PVAV)
        croak("Data::TopK::Shared->add_many: expected an array reference");
    av = (AV *)SvRV(items);
    top = av_len(av);                     /* last index, -1 if empty */
    {
        STRLEN cnt = (top >= 0) ? (STRLEN)(top + 1) : 0, i;
        const char **ps = NULL; STRLEN *ls = NULL;
        if (cnt) {                                       /* resolve all bytes BEFORE locking */
            Newx(ps, cnt, const char *); SAVEFREEPV(ps); /* freed on return OR unwind */
            Newx(ls, cnt, STRLEN);       SAVEFREEPV(ls);
            for (i = 0; i < cnt; i++) {                  /* a croak here holds NO lock; SAVEFREEPV cleans up */
                SV **el = av_fetch(av, (SSize_t)i, 0);
                if (el && *el) ps[i] = SvPVbyte(*el, ls[i]);
                else { ps[i] = ""; ls[i] = 0; }
            }
        }
        tk_rwlock_wrlock(h);                             /* locked region: NO croak-capable calls */
        for (i = 0; i < cnt; i++) { tk_observe_locked(h, ps[i], ls[i]); processed++; }
        __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);  /* a call always counts, even an empty batch */
        tk_rwlock_wrunlock(h);
    }
    RETVAL = processed;
  OUTPUT:
    RETVAL

UV
estimate(self, item)
    SV *self
    SV *item
  PREINIT:
    EXTRACT(self);
    STRLEN n;
    const char *s;
  CODE:
    s = SvPVbyte(item, n);                 /* may croak (wide char) -- BEFORE the lock */
    tk_rwlock_rdlock(h);
    RETVAL = (UV)tk_estimate_locked(h, s, n, NULL);
    tk_rwlock_rdunlock(h);
  OUTPUT:
    RETVAL

UV
error(self, item)
    SV *self
    SV *item
  PREINIT:
    EXTRACT(self);
    STRLEN n;
    const char *s;
    uint64_t err = 0;
  CODE:
    s = SvPVbyte(item, n);                 /* may croak (wide char) -- BEFORE the lock */
    tk_rwlock_rdlock(h);
    (void)tk_estimate_locked(h, s, n, &err);
    tk_rwlock_rdunlock(h);
    RETVAL = (UV)err;                       /* 0 for an untracked key or an exact count */
  OUTPUT:
    RETVAL

void
top(self, k = 0)
    SV *self
    UV k
  PREINIT:
    EXTRACT(self);
  PPCODE:
    {
        TkEnt *ents = NULL;
        char  *keys = NULL;
        uint64_t used, ks = h->key_size, i, cap = h->capacity;
        /* Allocate worst-case snapshot buffers (all `capacity` slots) BEFORE the
           lock: Newx can croak on OOM, and under the read lock that longjmp would
           strand the lock.  `used` (below) is clamped to capacity, so these fit. */
        if (cap) {
            Newx(ents, (size_t)cap, TkEnt);          SAVEFREEPV(ents);
            Newx(keys, (size_t)(cap * ks), char);    SAVEFREEPV(keys);
        }
        /* Snapshot every monitored (count, error, key) under the read lock; the
           copied key bytes cannot then be evicted out from under us, and all
           Perl-value building happens after the unlock. */
        tk_rwlock_rdlock(h);
        used = tk_heap_size(h);            /* == used, clamped to capacity (Layer B) */
        for (i = 0; i < used; i++) {
            TkSlot *sl = tk_slot(h, i);
            uint32_t kl = sl->key_len; if (kl > h->key_size) kl = h->key_size;  /* Layer B */
            ents[i].count = sl->count;
            ents[i].error = sl->error;
            ents[i].off   = i * ks;        /* 64-bit byte offset into the key blob */
            ents[i].klen  = kl;
            memcpy(keys + ents[i].off, tk_slot_key(sl), kl);
        }
        tk_rwlock_rdunlock(h);

        if (used) {
            qsort(ents, (size_t)used, sizeof(TkEnt), tk_ent_cmp);
            uint64_t want = (k == 0 || k > used) ? used : (uint64_t)k;
            EXTEND(SP, (SSize_t)want);
            for (i = 0; i < want; i++) {
                HV *hv = newHV();
                hv_stores(hv, "key",   newSVpvn(keys + ents[i].off, ents[i].klen));
                hv_stores(hv, "count", newSVuv((UV)ents[i].count));
                hv_stores(hv, "error", newSVuv((UV)ents[i].error));
                PUSHs(sv_2mortal(newRV_noinc((SV *)hv)));
            }
        }
    }

void
clear(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    tk_rwlock_wrlock(h);
    tk_clear_locked(h);
    __atomic_fetch_add(&h->hdr->stat_ops, 1, __ATOMIC_RELAXED);
    tk_rwlock_wrunlock(h);

UV
capacity(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = (UV)h->hdr->capacity;
  OUTPUT:
    RETVAL

UV
key_size(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = (UV)h->hdr->key_size;
  OUTPUT:
    RETVAL

UV
tracked(self)
    SV *self
  PREINIT:
    EXTRACT(self);
    UV n;
  CODE:
    tk_rwlock_rdlock(h);
    n = (UV)tk_heap_size(h);
    tk_rwlock_rdunlock(h);
    RETVAL = n;
  OUTPUT:
    RETVAL

UV
seen(self)
    SV *self
  PREINIT:
    EXTRACT(self);
    UV n;
  CODE:
    tk_rwlock_rdlock(h);
    n = (UV)h->hdr->seen;
    tk_rwlock_rdunlock(h);
    RETVAL = n;
  OUTPUT:
    RETVAL

SV *
stats(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    {
        uint64_t used, seen, ops;
        uint32_t cap, ks;
        tk_rwlock_rdlock(h);
        used = tk_heap_size(h);
        seen = h->hdr->seen;
        cap  = h->hdr->capacity;
        ks   = h->hdr->key_size;
        ops  = h->hdr->stat_ops;
        tk_rwlock_rdunlock(h);

        HV *hv = newHV();
        hv_stores(hv, "capacity",  newSVuv((UV)cap));
        hv_stores(hv, "key_size",  newSVuv((UV)ks));
        hv_stores(hv, "tracked",   newSVuv((UV)used));
        hv_stores(hv, "seen",      newSVuv((UV)seen));
        hv_stores(hv, "ops",       newSVuv((UV)ops));
        hv_stores(hv, "mmap_size", newSVuv((UV)h->mmap_size));
        RETVAL = newRV_noinc((SV *)hv);
    }
  OUTPUT:
    RETVAL

SV *
path(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = h->path ? newSVpv(h->path, 0) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
memfd(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    RETVAL = h->backing_fd;
  OUTPUT:
    RETVAL

void
sync(self)
    SV *self
  PREINIT:
    EXTRACT(self);
  CODE:
    if (tk_msync(h) != 0) croak("sync: %s", strerror(errno));

void
unlink(self, ...)
    SV *self
  CODE:
    if (sv_isobject(self) && sv_derived_from(self, "Data::TopK::Shared")) {
        TkHandle *h = INT2PTR(TkHandle*, SvIV(SvRV(self)));
        if (h && h->path) unlink(h->path);
    } else if (items >= 2 && (SvGETMAGIC(ST(1)), SvOK(ST(1)))) {
        unlink(SvPV_nolen(ST(1)));
    }
