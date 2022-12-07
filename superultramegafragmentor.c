/*
 * Module to fragment memory artificially according to some profile.
 */

#include <linux/module.h>
#include <linux/gfp.h>
#include <linux/vmalloc.h>
#include <linux/prandom.h>
#include <linux/shrinker.h>
#include <linux/mm.h>
#include <linux/proc_fs.h>

MODULE_AUTHOR("Mark Mansi");
MODULE_LICENSE("Dual MIT/GPL");

////////////////////////////////////////////////////////////////////////////////
// State.

#define FLAGS_BUDDY (1 << 0)
#define FLAGS_FILE (1 << 1)
#define FLAGS_ANON (1 << 2)
#define FLAGS_ANON_THP (1 << 3)
#define FLAGS_NONE (1 << 4)
#define FLAGS_PINNED (1 << 5)

#define GFP_FLAGS (__GFP_NORETRY | __GFP_HIGHMEM | \
    __GFP_MOVABLE | __GFP_IO | __GFP_FS | __GFP_THISNODE)

// In the MP, we need to represent probabilities using integers (because
// floating point is not allowed in the kernel). To do this we represent a
// probability p as `p * MP_GRANULARITY`.
#define MP_GRANULARITY 1000

// The profile graph.
//
// `struct profile` is the main data structure. It has a list of nodes and a
// list of edges. All of the edges are kept in the `profile->edges` field in a
// big array. All edges that originate from the same node are contiguous in the
// array (e.g., all outgoing edges from node A, then all outgoing edges from
// node B, and so on). There is no guarantee that they are sorted, though.

struct profile_node {
    u64 npages;
    u64 flags;

    // The index of the first outgoing edge from this node in the list of edges.
    u64 first_edge_idx;

    // The total sum of probabilities of all outgoing edges. This should be
    // close to `MP_GRANULARITY`, but because of rounding/truncation when
    // converting to integers, it may be slightly less.
    u64 edge_total;
};

struct profile_edge {
    u64 from;
    u64 to;

    // an int from 0 to MP_GRANYLARITY representing a probability that this
    // edge is taken.
    u64 prob;
};

struct profile {
    // Array of nodes.
    struct profile_node *nodes;
    u64 nnodes;

    // Array of edges.
    struct profile_edge *edges;
    u64 nedges;
};

// Pages on a single NUMA node of a single type.
struct sumf_page_pool {
    u64 npages; // always base pages, even if the pages in `pages` are high-order...
    struct list_head pages;
};

// Different types of memory usage. Used for indexing sumf_node_pools.pools.
enum SUMF_LIST {
    SUMF_PINNED = 0,
    SUMF_ANON = 1,
    SUMF_ANON_THP = 2,
    SUMF_FILE = 3,

    SUMF_NTYPES,
};

// A collection of pools of different types but all on the same NUMA node.
struct sumf_node_pools {
    struct sumf_page_pool pools[SUMF_NTYPES];
};

// All pages taken from the kernel for sumf.
static struct sumf_node_pools captured_pages[MAX_NUMNODES];
static DEFINE_SPINLOCK(allocation_lock); // locks captured_pages

static u64 sumf_node_count_pages(struct sumf_node_pools *pools) {
    u64 total = 0;
    int ty;
    for (ty = 0; ty < SUMF_NTYPES; ++ty) {
        total += pools->pools[ty].npages;
    }
    return total;
}

// Given a node, find the index of that node.
#define profile_node_idx(profile, node) ((node) - ((profile)->nodes))

// Iterate over all outgoing edges from `node`.
// profile: struct profile *
// node: struct profile_node *
// n: struct profile_node *
// e: struct profile_edge *
//
// At each iteration, `e` will be an edge that connects `node` to a neighbor.
#define profile_for_each_edge(profile, node, e) \
    for(e = (profile)->edges + (node)->first_edge_idx; \
        e < ((profile)->edges + (profile)->nedges) && \
        e->from == profile_node_idx(profile, node); \
        ++e)

////////////////////////////////////////////////////////////////////////////////
// Configuration.

// When flipped to 1, start fragmenting; then, reset to 0.
// When flipped to 2, free any held memory; then, reset to 0.
static int trigger = 0;
static int trigger_set(const char *val, const struct kernel_param *kp);
static const struct kernel_param_ops trigger_ops = {
    .set = trigger_set,
    .get = param_get_int,
};
module_param_cb(trigger, &trigger_ops, &trigger, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH);

// How much memory to allocate in units of 4KB pages.
static u64 npages = 0;
module_param(npages, ullong, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH);

// Whether to enable the shrinker or not.
static bool enable_shrinker = 0;
module_param(enable_shrinker, bool, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH);

// Stats about how many pages were shrunk.
static u64 stats_nshrunk = 0;
module_param(stats_nshrunk, ullong, S_IRUSR | S_IRGRP | S_IROTH);

// Stats about how many pages were actually allocated initially.
static u64 stats_nactual_alloc = 0;
module_param(stats_nactual_alloc, ullong, S_IRUSR | S_IRGRP | S_IROTH);

// Stats about how many pages were freed initially.
static u64 stats_nfree_initially = 0;
module_param(stats_nfree_initially, ullong, S_IRUSR | S_IRGRP | S_IROTH);

// Stats about how many cycles are spent in the shrinker.
static u64 stats_shrinker_cycles = 0;
module_param(stats_shrinker_cycles, ullong, S_IRUSR | S_IRGRP | S_IROTH);

// Profile to fragment memory with.
#define PROFILE_STR_MAX 2097152
static char profile_str[PROFILE_STR_MAX];
static size_t profile_str_n = 0;
static struct profile *profile = NULL;
static struct proc_dir_entry *ent;

static ssize_t profile_write(struct file *file, const char __user *ubuf,
                             size_t count, loff_t *ppos)
{
    size_t max_writen = min(PROFILE_STR_MAX - profile_str_n, count);
    ssize_t ret;

    if ((ret = copy_from_user(&profile_str[*ppos], ubuf, max_writen))) {
        return ret;
    }

    *ppos += max_writen;
    profile_str_n = *ppos;

    return count;
}

static ssize_t profile_read(struct file *file, char __user *ubuf,
                            size_t count, loff_t *ppos)
{
    size_t max_readn = min(profile_str_n - ((size_t)*ppos), count);
    ssize_t ret;

    if (*ppos >= profile_str_n) {
        return 0;
    }

    if ((ret = copy_to_user(ubuf, &profile_str[*ppos], max_readn))) {
        return ret;
    }

    *ppos += max_readn;

    return max_readn;
}

static const struct proc_ops profile_ops = {
    .proc_read = profile_read,
    .proc_write = profile_write,
};

////////////////////////////////////////////////////////////////////////////////
// Parsing and building the profile graph.

// Parse the first u64 in the string `*cursor` and place it in `*res`. Then,
// advance `*cursor` past the parsed integer. The new value of `*cursor` will
// point to the first character after the integer (e.g., `;` or NULL).
//
// Integers are assumed to be delimited by a single ' ' (space) character. The
// search will not attempt to look beyond `end`, though it may access
// characters beyond `end` if not NULL terminated.
//
// Returns 0 if successful. Returns -1 if there is no token. Otherwise, returns
// the return value of `kstrtoull`.
static int profile_parse_u64(
        const char **cursor, const char *end,
        unsigned int base, u64 *res, char delimiter)
{
    const char *tok_end;
    char tmp[40];
    int ret;

    tok_end = strchr(*cursor, delimiter);
    if (tok_end == NULL || tok_end > end) tok_end = end;
    if (*cursor >= tok_end) return -1;

    strncpy(tmp, *cursor, tok_end - *cursor);
    tmp[tok_end - *cursor] = 0;

    ret = kstrtoull(tmp, base, res);
    if (ret != 0) return ret;

    *cursor = tok_end + 1;

    return 0;
}

// Attempts to parse the contents of `profile_str` and generate output to
// `profile`. Returns an errno in case of error.
//
// Format for input:
// - All one line with nodes ended by `;`, including the last one.
// - No trailing or separating whitespace except as below.
// - The first node specified should be the source node.
// - Each node has the following format:
//      S|F (T P)+;
//
//   All elements are u64, space separated.
//   S: number of 4KB pages.
//   F: flags (in hex).
//   (T P)+: one or more outgoing edges.
//      T: the index of the node on the other side of the edge.
//      P: a u64 from 0 to MP_GRANULARITY which is the probability of taking
//         the edge. The sum of outgoing edges should be ~MP_GRANULARITY.
//      Don't actually write the ( ) + characters... I was using regex notation.
//
//  Example: "1|0 0 50 1 50;3|0 0 1000;"
//
//        +---+
//     .5 |   |  .5
//        +->[0]--->[1]
//            ^      |
//            +------+
//               1
//
static int profile_parse(void) {
    u64 nnodes = 0;
    u64 nedges = 0;
    int i, j;
    int ret = 0;
    const char *cursor = profile_str;
    char *line_end = NULL;
    struct profile_node *node;
    struct profile_edge *edge;
    u64 discard, ecount = 0, prob_total, node_nedges;

    // Free any existing profile.
    if (profile) {
        vfree(profile->nodes);
        vfree(profile->edges);
        vfree(profile);
        profile = NULL;
    }

    // Count the number of nodes.
    for (i = 0; profile_str[i] != 0; ++i) {
        if (profile_str[i] == ';') ++nnodes;
    }

    // Error: found no ';'...
    if (nnodes == 0) {
        printk(KERN_ERR "frag: no ';' found...\n");
        ret = -EINVAL;
        goto err_out;
    }

    //printk(KERN_WARNING "frag: %llu\n", nnodes);

    profile = vzalloc(sizeof(struct profile));
    if (!profile) {
        ret = -ENOMEM;
        goto err_out;
    }

    profile->nnodes = nnodes;
    profile->nodes = vzalloc(sizeof(struct profile_node) * nnodes);
    if (!profile->nodes) {
        ret = -ENOMEM;
        goto err_out;
    }

    // Work our way through the list of nodes. Parse the node size and flags
    // and count the number of edges so we can parse edges next.
    i = 0;
    while ((line_end = strchr(cursor, ';'))) {
        node = &profile->nodes[i++];

        // Parse size
        ret = profile_parse_u64(&cursor, line_end, 10, &node->npages, '|');
        if (ret == -1) {
            printk(KERN_ERR "frag: expected size, found end of line\n");
            ret = -EINVAL;
            goto err_out;
        } else if (ret != 0) {
            printk(KERN_ERR "frag: error parsing size\n");
            goto err_out;
        }

        // Parse flags
        ret = profile_parse_u64(&cursor, line_end, 16, &node->flags, ' ');
        if (ret == -1) {
            printk(KERN_ERR "frag: expected flags, found end of line\n");
            ret = -EINVAL;
            goto err_out;
        } else if (ret != 0) {
            printk(KERN_ERR "frag: error parsing flags\n");
            goto err_out;
        }

        // We haven't allocated yet, but we do know the index.
        node->first_edge_idx = nedges;

        // Count edges...
        ecount = 0;
        while ((ret = profile_parse_u64(&cursor, line_end, 10, &discard, ' ')) == 0) {
            //printk(KERN_WARNING "frag: discard=%llu\n", discard);
            ecount++;
        }
        if (ecount % 2 != 0) {
            printk(KERN_ERR "frag: edge missing probability ecount=%llu\n", ecount);
            if (ret == -1) ret = -EINVAL;
            goto err_out;
        }
        node_nedges = ecount / 2;
        nedges += node_nedges;

        // printk(KERN_WARNING "frag: nedges so far %llu\n", nedges);

        if (node_nedges == 0) {
            printk(KERN_ERR "frag: must have at least one outgoing edge\n");
            goto err_out;
        }
    }

    // Parse edges.
    profile->nedges = nedges;
    profile->edges = vzalloc(sizeof(struct profile_edge) * nedges);
    if (!profile->edges) {
        ret = -ENOMEM;
        goto err_out;
    }

    cursor = profile_str;
    i = 0; // node idx.
    j = 0; // edge idx.
    while ((line_end = strchr(cursor, ';'))) {
        node = &profile->nodes[i++];

        // Already did these.
        ret = profile_parse_u64(&cursor, line_end, 10, &discard, '|');
        BUG_ON(ret != 0);
        ret = profile_parse_u64(&cursor, line_end, 16, &discard, ' ');
        BUG_ON(ret != 0);

        prob_total = 0;

        // Parse edges.
        while(cursor < line_end) {
            edge = &profile->edges[j++];
            edge->from = profile_node_idx(profile, node);

            ret = profile_parse_u64(&cursor, line_end, 10, &edge->to, ' ');
            if (ret == -1) {
                printk(KERN_ERR "frag: expected dest node, found end of line\n");
                ret = -EINVAL;
                goto err_out;
            } else if (ret != 0) {
                printk(KERN_ERR "frag: error parsing dest node\n");
                goto err_out;
            }

            ret = profile_parse_u64(&cursor, line_end, 10, &edge->prob, ' ');
            if (ret == -1) {
                printk(KERN_ERR "frag: expected prob, found end of line\n");
                ret = -EINVAL;
                goto err_out;
            } else if (ret != 0) {
                printk(KERN_ERR "frag: error parsing prob\n");
                goto err_out;
            }

            prob_total += edge->prob;
        }

        if (prob_total > MP_GRANULARITY || prob_total < (MP_GRANULARITY * 2 / 3)) {
            printk(KERN_ERR "frag: probabilities add to %llu != %d\n",
                    prob_total, MP_GRANULARITY);
            goto err_out;
        }

        node->edge_total = prob_total;
    }

    return 0;

err_out:
    if (profile) {
        if (profile->nodes) vfree(profile->nodes);
        if (profile->edges) vfree(profile->edges);
        vfree(profile);
    }

    profile = NULL;

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// The Meat.

#define ALLOC_ORDER (HPAGE_SHIFT - PAGE_SHIFT)

// Get the `n`th element of the list, or the last element if there are fewer
// than `n` elements left. If the list is empty, return null.
static struct list_head *list_nth(struct list_head *head, u64 n) {
    struct list_head *elem = head->next;

    if (list_empty(head)) return NULL;

    while (n-- > 0 && !list_is_last(elem, head)) {
        elem = elem->next;
    }

    return elem;
}

static u64 list_count(struct list_head *list) {
    u64 count = 0;
    struct list_head *it;
    list_for_each(it, list) {
        count += 1;
    }
    return count;
}

struct allocated_unsplit_pages {
    // Any unsplit (order ALLOC_ORDER) pages we have allocated. Initially, they
    // are all in this list.
    struct list_head unsplit;

    // The any pages that has already been split.
    struct list_head split;
    u64 nsplit;
};

static void init_allocated_unsplit_pages(struct allocated_unsplit_pages *aup) {
    INIT_LIST_HEAD(&aup->unsplit);
    INIT_LIST_HEAD(&aup->split);
    aup->nsplit = 0;
}

// Take `n` base pages worth of unsplit pages, rounding down to the next unsplit page.
// The pages are returned unsplit.
//
// If there are not `n` pages remaining, all remaining pages are returned.
//
// Returns the amount of pages taken.
static u64 take_unsplit_pages(u64 n,
        struct allocated_unsplit_pages *aup, struct list_head *assigned)
{
    const u64 nhighorder = n >> ALLOC_ORDER;
    LIST_HEAD(tmp);
    struct list_head *nth_entry;
    u64 ret;

    if (list_empty(&aup->unsplit)) {
        return 0;
    }

    nth_entry = list_nth(&aup->unsplit, nhighorder - 1);
    list_cut_position(&tmp, &aup->unsplit, nth_entry);
    ret = list_count(&tmp) << ALLOC_ORDER;
    list_splice_tail(&tmp, assigned->prev);
    return ret;
}

// Take `n` base pages already split. Or if there are not `n` pages
// remaining, take all remaining pages. Return the number of pages taken.
static u64 take_and_split_pages(u64 n,
        struct allocated_unsplit_pages *aup, struct list_head *assigned)
{
    u64 nassigned = 0, nsplit;
    struct list_head *nth_entry, *next_unsplit;
    LIST_HEAD(tmp);
    struct page *pages;
    int i;

    while (true) {
        // Take as many pages as we can that are already split.
        nsplit = min(aup->nsplit, n - nassigned);
        if (nsplit > 0) {
            nth_entry = list_nth(&aup->split, nsplit - 1);
            list_cut_position(&tmp, &aup->split, nth_entry);
            list_splice_tail_init(&tmp, assigned->prev);
            nassigned += nsplit;
            aup->nsplit -= nsplit;
        }

        // Check if we are done.
        if (nassigned == n) {
            return nassigned;
        }

        // There are no more split pages; split a new unsplit page.
        BUG_ON(!list_empty(&aup->split));
        if (list_empty(&aup->unsplit)) {
            // Out of pages...
            return nassigned;
        }

        // Split...
        next_unsplit = aup->unsplit.next;
        list_del(next_unsplit);
        pages = list_entry(next_unsplit, struct page, lru);
        split_page(pages, ALLOC_ORDER);
        for (i = 0; i < (1 << ALLOC_ORDER); ++i) {
            list_add_tail(&pages[i].lru, &aup->split);
            aup->nsplit += 1;
        }
    }
}

static bool aup_empty(struct allocated_unsplit_pages *aup) {
    return list_empty(&aup->unsplit) && (aup->nsplit == 0);
}

// Allocate >=N pages with as much contiguity as possible, and add them to the
// given list of pages without splitting them. Update `n` with actual number
// allocated.
static int alloc_npages(int nid, u64 *n, struct allocated_unsplit_pages *aup) {
    u64 alloced = 0;
    struct page *pages;

    while (alloced < *n) {
        pages = alloc_pages_node(nid, GFP_FLAGS, ALLOC_ORDER);
        if (!pages) {
            return -ENOMEM;
        } else {
            list_add_tail(&pages->lru, &aup->unsplit);
            alloced += 1 << ALLOC_ORDER;
        }

        // Let the scheduler know we haven't deadlocked.
        if (alloced % (1 << 12) == 0) {
            cond_resched();
        }
    }

    *n = alloced;

    return 0;
}

static void set_page_flags(struct list_head *list, bool lru,
        bool private, bool private2, bool ownerpriv, bool reserved)
{
    struct page *page;
    list_for_each_entry(page, list, lru) {
        if (lru) SetPageLRU(page);
        if (private) SetPagePrivate(page);
        if (private2) SetPagePrivate2(page);
        if (ownerpriv) SetPageOwnerPriv1(page);
        if (reserved) SetPageReserved(page);
    }
}

// Free the first page of the list or do nothing if the list is empty.
static inline void free_first_page(struct list_head *list, u64 order) {
    if (!list_empty(list)) {
        // remove from list.
        struct page *p = list_first_entry(list, struct page, lru);
        list_del_init(&p->lru);

        // clear any flags we may have set before we release the page to the
        // buddy allocator.
        ClearPageLRU(p);
        ClearPagePrivate(p);
        ClearPagePrivate2(p);
        ClearPageOwnerPriv1(p);
        ClearPageReserved(p);

        // free the page back to the buddy allocator.
        __free_pages(p, order);
    }
}

// Free all pages on the given list.
static void free_all_pages(struct list_head *list, u64 order) {
    while (!list_empty(list)) {
        free_first_page(list, order);
    }
}

static void free_aup(struct allocated_unsplit_pages *aup) {
    free_all_pages(&aup->split, 0);
    aup->nsplit = 0;
    free_all_pages(&aup->unsplit, ALLOC_ORDER);
}

static u64 free_pool(struct sumf_page_pool *pool, u64 order) {
    u64 npages = pool->npages;
    free_all_pages(&pool->pages, order);
    pool->npages = 0;
    return npages;
}

static u64 free_node_pools(struct sumf_node_pools *pools) {
    u64 npages = 0;
    int ty;
    for (ty = 0; ty < SUMF_NTYPES; ++ty) {
        npages += free_pool(&pools->pools[ty],
                ty == SUMF_ANON_THP ? ALLOC_ORDER : 0);
    }
    return npages;
}

static void free_memory(void) {
    u64 total = 0;
    int nid;

    for (nid = 0; nid < MAX_NUMNODES; ++nid) {
        total += free_node_pools(&captured_pages[nid]);
    }

    printk(KERN_WARNING "frag: Freed %llu pages.\n", total);
}

#define CACHE_GRANULARITY 1000
#define CACHE_MEM_PREALLOC_N (256 << 10)

struct linked_list_cache {
    // The first element in the cache array.
    struct list_head **cache;
    // The number of elements in the cache.
    u64 length;
};

static inline void list_mk_cache(
        struct linked_list_cache *cache,
        struct list_head *head, u64 n)
{
    // Preallocating the array as a static here allows us to avoid OOM
    // situations due to trying to allocate the cache after we have already
    // called alloc_npages.
    static struct list_head* cache_mem_prealloc[CACHE_MEM_PREALLOC_N];

    struct list_head *elem;
    u64 i = 0;

    cache->length = n / CACHE_GRANULARITY;
    BUG_ON(cache->length > CACHE_MEM_PREALLOC_N);
    cache->cache = cache_mem_prealloc;

    list_for_each(elem, head) {
        if (i % CACHE_GRANULARITY == 0) {
            cache->cache[i / CACHE_GRANULARITY] = elem;
        }

        i += 1;
    }
}

static struct list_head *list_nth_with_cache(
        struct linked_list_cache *cache,
        struct list_head *head, u64 n)
{
    u64 cachen = n / CACHE_GRANULARITY;
    u64 remainder = n % CACHE_GRANULARITY;

    return list_nth(cache->cache[cachen]->prev, remainder);
}

// Randomize the given list.
static void pool_randomize(struct sumf_page_pool *pool, u64 order) {
    struct list_head *head = &pool->pages;
    struct list_head *curr = head->next, *prev = head, *second;
    u32 i = 0, rand;
    struct linked_list_cache cache = { };
    u64 cachen, npages;

    BUG_ON(pool->npages % (1 << order) != 0);
    npages = pool->npages >> order;

    if (list_empty(head)) {
        BUG_ON(pool->npages > 0);
        return;
    }

    // Initialize cache...
    list_mk_cache(&cache, head, npages);

    while (!list_is_last(curr, head)) {
        // Pick another random element from the rest of the list.
        rand = prandom_u32() % (npages - i - 1);
        second = list_nth_with_cache(&cache, head, i + 1 + rand);

        // Swap them to randomize the list.
        list_swap(curr, second);

        // Update the cache...
        if (i % CACHE_GRANULARITY == 0) {
            cachen = ((u64)i) / CACHE_GRANULARITY;
            cache.cache[cachen] = second;
        }
        if ((rand + i + 1) % CACHE_GRANULARITY == 0) {
            cachen = ((u64)(rand + i + 1)) / CACHE_GRANULARITY;
            cache.cache[cachen] = curr;
        }

        if (npages < 100 || i % (npages / 100) == 0) {
            printk(KERN_ERR "frag: rand %d %lld%%\n", i, i * 100 / npages);
        }

        // Let the scheduler know we haven't deadlocked...
        if (i % 500 == 0) {
            cond_resched();
        }

        // Move to the next element.
        prev = prev->next;
        curr = prev->next;
        i += 1;
    }
}

// Take a random step in the profile MP from the `current` node.
static struct profile_node *rand_mp_step(struct profile_node *current_node) {
    struct profile_edge *selected_edge;
    struct profile_node *next_node;

    u64 rand = (get_random_u64() % current_node->edge_total) + 1;
    u64 edge_idx = current_node->first_edge_idx;
    u64 walk = profile->edges[edge_idx].prob;

    while (walk < rand) {
       edge_idx += 1;
       walk += profile->edges[edge_idx].prob;
    }

    selected_edge = &profile->edges[edge_idx];
    next_node =  &profile->nodes[selected_edge->to];

    if (unlikely(
            selected_edge->from != profile_node_idx(profile, current_node) ||
            selected_edge->to != profile_node_idx(profile, next_node)))
    {
        pr_err("pco: current_node=%lu next_node=%lu edge=%llu->%llu\n",
                profile_node_idx(profile, current_node),
                profile_node_idx(profile, next_node),
                selected_edge->from,
                selected_edge->to
                );
        BUG();
    }

    return next_node;
}

// Allocate the memory. Then, do a random walk over the markov process until we
// have fragmented all memory we allocated.
static int do_fragment(int nid, u64 npages) {
    int ret;
    struct profile_node *current_node;
    struct allocated_unsplit_pages aup;
    struct sumf_page_pool *pools = captured_pages[nid].pools;
    struct list_head pages_to_free;

    init_allocated_unsplit_pages(&aup);

    // Sanity check...
    if (!profile) {
        printk(KERN_ERR "frag: no profile\n");
        ret = -EINVAL;
        goto free_and_error;
    }
    if (npages == 0) {
        printk(KERN_ERR "frag: must set npages\n");
        ret = -EINVAL;
        goto free_and_error;
    }

    // Allocate the requested amount of memory. If the system was relatively
    // unfragmented before (e.g., after a fresh reboot), then we can expect the
    // pages in the list to be fairly contiguous and in order.
    ret = alloc_npages(nid, &npages, &aup);
    if (ret != 0) {
        printk(KERN_ERR "frag: error allocating: %d\n", ret);
        return ret;
    }
    printk(KERN_WARNING "frag: done allocating %llu pages\n", npages);
    stats_nactual_alloc = npages;

    // Fragment the memory by doing a random walk over the profile.
    // Start at node 0 and then walk randomly.
    current_node = &profile->nodes[0];
    while (!aup_empty(&aup)) {
        // Based on current_node, we want to remove `npages` pages from
        // `allocated_pages` and set them to the purpose indicated by `flags`.
        u64 pages = current_node->npages;

        switch (current_node->flags) {
            case FLAGS_NONE:
            case FLAGS_PINNED:
                pages = take_and_split_pages(pages, &aup, pools[SUMF_PINNED].pages.prev);
                pools[SUMF_PINNED].npages += pages;
                //printk(KERN_WARNING "frag: P %llu %llu\n",
                //        pools[SUMF_PINNED].npages,
                //        list_count(&pools[SUMF_PINNED].pages));
                break;

            case FLAGS_FILE:
                pages = take_and_split_pages(pages, &aup, pools[SUMF_FILE].pages.prev);
                pools[SUMF_FILE].npages += pages;
                //printk(KERN_WARNING "frag: F %llu %llu\n",
                //        pools[SUMF_FILE].npages,
                //        list_count(&pools[SUMF_FILE].pages));
                break;

            case FLAGS_ANON:
                pages = take_and_split_pages(pages, &aup, pools[SUMF_ANON].pages.prev);
                pools[SUMF_ANON].npages += pages;
                //printk(KERN_WARNING "frag: A %llu %llu\n",
                //        pools[SUMF_ANON].npages,
                //        list_count(&pools[SUMF_ANON].pages));
                break;

            case FLAGS_ANON_THP:
                pages = take_unsplit_pages(pages, &aup, pools[SUMF_ANON_THP].pages.prev);
                pools[SUMF_ANON_THP].npages += pages;
                //printk(KERN_WARNING "frag: T %llu %llu\n",
                //        pools[SUMF_ANON_THP].npages,
                //        list_count(&pools[SUMF_ANON_THP].pages) << ALLOC_ORDER);
                break;

            case FLAGS_BUDDY:
                INIT_LIST_HEAD(&pages_to_free);
                take_and_split_pages(pages, &aup, &pages_to_free);
                free_all_pages(&pages_to_free, 0);
                stats_nfree_initially += pages;
                //printk(KERN_WARNING "frag: B %llu\n", pages);
                break;

            default:
                ret = -EINVAL;
                printk(KERN_ERR "frag: unrecognized flags: %llx\n", current_node->flags);
                goto free_and_error;
        }

        current_node = rand_mp_step(current_node);
    }

    // Randomize the lists. (We don't randomize pinned pages, though because
    // they are not reclaimable).
    //
    // We also set some flags on each page to indicate which category the page
    // is in when we use the kpfsnapshot tool. The set of flags for each
    // category is arbitrary -- they just need to be different.
    printk(KERN_WARNING "frag: Randomize file pages. pages=%llu\n", pools[SUMF_FILE].npages);
    pool_randomize(&pools[SUMF_FILE], 0);
    set_page_flags(&pools[SUMF_FILE].pages, /*lru*/true, /*priv*/false,
                                /*priv2*/false, /*opriv*/true, /*rsvd*/true);
    printk(KERN_WARNING "frag: Randomize anon pages. pages=%llu\n", pools[SUMF_ANON].npages);
    pool_randomize(&pools[SUMF_ANON], 0);
    set_page_flags(&pools[SUMF_ANON].pages, /*lru*/false, /*priv*/true,
                                /*priv2*/false, /*opriv*/true, /*rsvd*/true);
    printk(KERN_WARNING "frag: Randomize anon thp pages. pages=%llu\n", pools[SUMF_ANON_THP].npages);
    pool_randomize(&pools[SUMF_ANON_THP], ALLOC_ORDER);
    set_page_flags(&pools[SUMF_ANON_THP].pages, /*lru*/false, /*priv*/true,
                                    /*priv2*/true, /*opriv*/true, /*rsvd*/true);
    printk(KERN_WARNING "frag: Marking pinned pages. pages=%llu\n", pools[SUMF_PINNED].npages);
    set_page_flags(&pools[SUMF_PINNED].pages, /*lru*/false, /*priv*/false,
                                  /*priv2*/false, /*opriv*/true, /*rsvd*/true);

    printk(KERN_WARNING "frag: The deed is done. (node %d) pages=%llu\n", nid, npages);

    return 0;

free_and_error:
    free_aup(&aup);
    free_memory();
    return ret;
}

static int trigger_set(const char *val, const struct kernel_param *kp) {
    int nnodes, nid;
    u64 split_npages;

    // Parse value.
    int ret = param_set_int(val, kp);
    if (ret != 0) return ret;

    // Parse profile.
    if ((ret = profile_parse()) != 0) {
        return ret;
    }

    spin_lock(&allocation_lock);

    // Trigger fragmentation.
    if (trigger == 1) {
        // Assume all nodes are the same size and split the quota of pages
        // equally among them.
        nnodes = num_online_nodes();
        split_npages = npages / nnodes;

        for_each_online_node(nid) {
            ret = do_fragment(nid, split_npages);
            if (ret != 0) {
                printk(KERN_WARNING
                    "frag: unable to fragment all nodes. nid=%d\n", nid);
                break;
            }
        }
    } else if (trigger == 2) {
        free_memory();
    } else {
        ret = -EINVAL;
    }

    spin_unlock(&allocation_lock);

    // Reset trigger.
    trigger = 0;

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Shrinker for memory reclamation.

static unsigned long
frag_shrink_count(struct shrinker *shrink, struct shrink_control *sc) {
    unsigned long ret;
    u64 start_tsc = rdtsc();

    ret = enable_shrinker
        ? sumf_node_count_pages(&captured_pages[sc->nid])
        : 0;

    stats_shrinker_cycles += rdtsc() - start_tsc;

    return ret;
}

// Free a random subset of `nr_to_scan` pages.
static unsigned long
frag_shrink_scan(struct shrinker *shrink, struct shrink_control *sc) {
    unsigned long freed = 0;
    struct sumf_page_pool *pools;
    u64 start_tsc = rdtsc();

    pools = captured_pages[sc->nid].pools;

    //printk(KERN_WARNING "frag: shrinking...\n");

    // Avoid deadlocks where we are trying to allocate and shrink at the same
    // time. We should only enable the shrinker after we are done allocating
    // and fragmenting memory.
    if (!enable_shrinker) goto out;

    spin_lock(&allocation_lock);

    // The free lists are already randomized, so we can free a random subset by
    // freeing pages from the head of the list.

    // Figure out what we will reclaim. Initially we want to reclaim file
    // memory and single anon pages. Then we will break apart THP pages.
    while ((pools[SUMF_FILE].npages > 0
                || pools[SUMF_ANON].npages > 0
                || pools[SUMF_ANON_THP].npages > 0)
            && freed < sc->nr_to_scan)
    {
        switch (prandom_u32() % 2) {
            case 0: // file
                if (pools[SUMF_FILE].npages > 0) {
                    free_first_page(&pools[SUMF_FILE].pages, 0);
                    pools[SUMF_FILE].npages -= 1;
                    freed += 1;
                    break;
                }
                // fall through
                // if we don't have file pages.

            case 1: // anon
                if (pools[SUMF_ANON].npages > 0) {
                    free_first_page(&pools[SUMF_ANON].pages, 0);
                    pools[SUMF_ANON].npages -= 1;
                    freed += 1;
                    break;
                }
                // fall through
                // if we don't have single anon pages.

            case 2: // anon thp -- will never happen on it's own. Only happens
                    // by fall-through.
                if (pools[SUMF_ANON_THP].npages > 0) {
                    free_first_page(&pools[SUMF_ANON_THP].pages, ALLOC_ORDER);
                    pools[SUMF_ANON_THP].npages -= 1 << ALLOC_ORDER;
                    freed += 1 << ALLOC_ORDER;
                }
                break;

            default:
                BUG();
        }
    }

    spin_unlock(&allocation_lock);

    stats_nshrunk += freed;

out:
    stats_shrinker_cycles += rdtsc() - start_tsc;

    return freed;
}

static struct shrinker frag_shrinker = {
    .count_objects = frag_shrink_count,
    .scan_objects = frag_shrink_scan,
    .flags = SHRINKER_NUMA_AWARE,
};

////////////////////////////////////////////////////////////////////////////////
// Module init/exit.

static int mod_init(void) {
    int ret, nid, ty;

    printk(KERN_WARNING "frag: Init.\n");

    // Init captured_pages
    memset(&captured_pages, 0, sizeof(captured_pages));
    for (nid = 0; nid < MAX_NUMNODES; ++nid) {
        for (ty = 0; ty < SUMF_NTYPES; ++ty) {
            INIT_LIST_HEAD(&captured_pages[nid].pools[ty].pages);
        }
    }

    // Init sysfs interface.
    memset(profile_str, 0, PROFILE_STR_MAX);
    ent = proc_create("sumf", 0660, NULL, &profile_ops);

    // Register shrinker to return memory.
    ret = register_shrinker(&frag_shrinker, "superultramegafragmentor");
    if (ret) return ret;

    return 0;
}
module_init(mod_init);

static void mod_exit(void) {
    unregister_shrinker(&frag_shrinker);

    proc_remove(ent);

    spin_lock(&allocation_lock);
    free_memory();
    spin_unlock(&allocation_lock);
    printk(KERN_WARNING "frag: Exit.\n");
}
module_exit(mod_exit);
