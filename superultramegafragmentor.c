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

#define GFP_FLAGS (__GFP_NORETRY | __GFP_HIGHMEM | __GFP_MOVABLE | __GFP_IO | __GFP_FS)

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
    u64 order;
    u64 flags;

    // The index of the first outgoing edge from this node in the list of edges.
    u64 first_edge_idx;
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

static u64 npages_total = 0;
static u64 npages_pinned = 0;
static u64 npages_anon = 0;
static u64 npages_anon_thp = 0;
static u64 npages_file = 0;
static struct list_head pages_pinned = LIST_HEAD_INIT(pages_pinned);
static struct list_head pages_anon = LIST_HEAD_INIT(pages_anon);
static struct list_head pages_anon_thp = LIST_HEAD_INIT(pages_anon_thp);
static struct list_head pages_file = LIST_HEAD_INIT(pages_file);
static DEFINE_SPINLOCK(allocation_lock); // locks the lists above...

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

// Profile to fragment memory with.
#define PROFILE_STR_MAX 8192
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
        const char **cursor, const char *end, unsigned int base, u64 *res)
{
    const char *tok_end;
    char tmp[40];
    int ret;

    tok_end = strchr(*cursor, ' ');
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
//      S F (T P)+;
//
//   All elements are u64, space separated.
//   S: order of the allocation (log(npages)).
//   F: flags (in hex).
//   (T P)+: one or more outgoing edges.
//      T: the index of the node on the other side of the edge.
//      P: a u64 from 0 to MP_GRANULARITY which is the probability of taking
//         the edge. The sum of outgoing edges should be ~MP_GRANULARITY.
//      Don't actually write the ( ) + characters... I was using regex notation.
//
//  Example: "1 0 0 50 1 50;3 0 0 1000;"
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
        ret = profile_parse_u64(&cursor, line_end, 10, &node->order);
        if (ret == -1) {
            printk(KERN_ERR "frag: expected size, found end of line\n");
            ret = -EINVAL;
            goto err_out;
        } else if (ret != 0) {
            printk(KERN_ERR "frag: error parsing size\n");
            goto err_out;
        }

        // Parse flags
        ret = profile_parse_u64(&cursor, line_end, 16, &node->flags);
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
        while ((ret = profile_parse_u64(&cursor, line_end, 10, &discard)) == 0) {
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
        ret = profile_parse_u64(&cursor, line_end, 10, &discard);
        BUG_ON(ret != 0);
        ret = profile_parse_u64(&cursor, line_end, 16, &discard);
        BUG_ON(ret != 0);

        prob_total = 0;

        // Parse edges.
        while(cursor < line_end) {
            edge = &profile->edges[j++];
            edge->from = profile_node_idx(profile, node);

            ret = profile_parse_u64(&cursor, line_end, 10, &edge->to);
            if (ret == -1) {
                printk(KERN_ERR "frag: expected dest node, found end of line\n");
                ret = -EINVAL;
                goto err_out;
            } else if (ret != 0) {
                printk(KERN_ERR "frag: error parsing dest node\n");
                goto err_out;
            }

            ret = profile_parse_u64(&cursor, line_end, 10, &edge->prob);
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

        if (prob_total != 100) {
            printk(KERN_ERR "frag: probabilities add to %llu != 100\n", prob_total);
            goto err_out;
        }
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

// Allocate N pages with as much contiguity as possible, and add them to the
// given list of pages.
static int alloc_npages(u64 n, struct list_head *list) {
    u64 alloced = 0;
    struct page *pages;
    u64 current_order;
    int i;

    while (alloced < n) {
        current_order = min(MAX_ORDER - 1, ilog2(n - alloced));
        pages = alloc_pages(GFP_FLAGS, current_order);
        split_page(pages, current_order);

        if (!pages) {
            return -ENOMEM;
        } else {
            for (i = 0; i < (1 << current_order); ++i) {
                list_add_tail(&pages[i].lru, list);
            }

            alloced += 1 << current_order;
        }

        // Let the scheduler know we haven't deadlocked.
        if (alloced % (1 << 12) == 0) {
            cond_resched();
        }
    }

    npages_total += alloced;

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
static inline void free_first_page(struct list_head *list) {
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
        __free_pages(p, 0);
    }
}

// Free all pages on the given list.
static void free_all_pages(struct list_head *list) {
    while (!list_empty(list)) {
        free_first_page(list);
    }
}

#define CACHE_GRANULARITY 1000

struct linked_list_cache {
    // The first element in the cache array.
    struct list_head **cache;
    // The number of elements in the cache.
    u64 length;
};

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

static inline void list_mk_cache(
        struct linked_list_cache *cache,
        struct list_head *head, u64 n)
{
    struct list_head *elem;
    u64 i = 0;

    printk(KERN_WARNING "frag: building cache...\n");

    cache->length = n / CACHE_GRANULARITY;
    cache->cache = vmalloc(cache->length * sizeof(struct page*));
    BUG_ON(cache->cache == NULL); // Hopefully won't happen often.

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
static void list_randomize(struct list_head *head, u64 npages) {
    struct list_head *curr = head->next, *prev = head, *second;
    u32 i = 0, rand;
    struct linked_list_cache cache = { };
    u64 cachen;

    if (list_empty(head)) {
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

        if (i % 1000 == 0) {
            printk(KERN_ERR "frag: rand %p %d\n", head, i);
            cond_resched();
        }

        // Move to the next element.
        prev = prev->next;
        curr = prev->next;
        i += 1;
    }
}

u64 list_count(struct list_head *head) {
    u64 count = 0;
    struct list_head *elem;

    list_for_each(elem, head) {
        count += 1;
    }

    return count;
}

// Allocate the memory. Then, do a random walk over the markov process until we
// have fragmented all memory we allocated.
static int do_fragment(void) {
    int ret;
    struct profile_node *current_node, *n;
    struct profile_edge *e;
    u32 rand, walk;
    u64 pages_remaining = npages;
    struct list_head allocated_pages = LIST_HEAD_INIT(allocated_pages);

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
    ret = alloc_npages(npages, &allocated_pages);
    if (ret != 0) {
        printk(KERN_ERR "frag: error allocating: %d\n", ret);
        return ret;
    }
    printk(KERN_WARNING "frag: done allocating %llu pages\n", npages);

    // Fragment the memory by doing a random walk over the profile.
    // Start at node 0 and then walk randomly.
    current_node = &profile->nodes[0];
    while (!list_empty(&allocated_pages)) {
        // Based on current_node, we want to remove `order` pages from
        // `allocated_pages` and set them to the purpose indicated by `flags`.
        struct list_head assigned_pages = LIST_HEAD_INIT(assigned_pages);
        u64 pages = min(pages_remaining, 1ull << current_node->order);
        struct list_head *nth_entry = list_nth(&allocated_pages, pages - 1);
        list_cut_position(&assigned_pages, &allocated_pages, nth_entry);

        switch (current_node->flags) {
            case FLAGS_NONE:
            case FLAGS_PINNED:
                list_splice_tail(&assigned_pages, pages_pinned.prev);
                npages_pinned += pages;
                break;

            case FLAGS_FILE:
                list_splice_tail(&assigned_pages, pages_file.prev);
                npages_file += pages;
                break;

            case FLAGS_ANON:
                list_splice_tail(&assigned_pages, pages_anon.prev);
                npages_anon += pages;
                break;

            case FLAGS_ANON_THP:
                // TODO: make these actually be huge page aligned
                list_splice_tail(&assigned_pages, pages_anon_thp.prev);
                npages_anon_thp += pages;
                break;

            case FLAGS_BUDDY:
                free_all_pages(&assigned_pages);
                break;

            default:
                free_all_pages(&assigned_pages);
                ret = -EINVAL;
                printk(KERN_ERR "frag: unrecognized flags: %llx\n", current_node->flags);
                goto free_and_error;
        }

        pages_remaining -= pages;

        // Then, randomly walk to a neighbor.
        rand = prandom_u32() % 100;
        walk = 0;
        profile_for_each_edge(profile, current_node, e) {
            n = &profile->nodes[e->to];
            walk += e->prob;
            if (rand < walk) break;
        }
        //printk(KERN_WARNING "frag: rand=%u walk=%u\n", rand, walk);
        current_node = n;
    }

    // Randomize the lists. (We don't randomize pinned pages, though because
    // they are not reclaimable).
    //
    // We also set some flags on each page to indicate which category the page
    // is in when we use the kpfsnapshot tool. The set of flags for each
    // category is arbitrary -- they just need to be different.
    printk(KERN_WARNING "frag: Randomize file pages. pages=%llu\n", npages_file);
    list_randomize(&pages_file, npages_file);
    set_page_flags(&pages_file, /*lru*/true, /*priv*/false,
                                /*priv2*/false, /*opriv*/true, /*rsvd*/true);
    printk(KERN_WARNING "frag: Randomize anon pages. pages=%llu\n", npages_anon);
    list_randomize(&pages_anon, npages_anon);
    set_page_flags(&pages_anon, /*lru*/false, /*priv*/true,
                                /*priv2*/false, /*opriv*/true, /*rsvd*/true);
    printk(KERN_WARNING "frag: Randomize anon thp pages. pages=%llu\n", npages_anon_thp);
    list_randomize(&pages_anon_thp, npages_anon_thp);
    set_page_flags(&pages_anon_thp, /*lru*/false, /*priv*/true,
                                    /*priv2*/true, /*opriv*/true, /*rsvd*/true);
    printk(KERN_WARNING "frag: Marking pinned pages. pages=%llu\n", npages_pinned);
    set_page_flags(&pages_pinned, /*lru*/false, /*priv*/false,
                                  /*priv2*/false, /*opriv*/true, /*rsvd*/true);

    printk(KERN_WARNING "frag: The deed is done. pages=%llu\n", npages);

    return 0;

free_and_error:
    free_all_pages(&allocated_pages);
    return ret;
}

static void free_memory(void) {
    free_all_pages(&pages_anon);
    free_all_pages(&pages_anon_thp);
    free_all_pages(&pages_file);
    free_all_pages(&pages_pinned);

    printk(KERN_WARNING "frag: Freed %llu pages.\n", npages_total);

    npages_total = 0;
    npages_anon = 0;
    npages_anon_thp = 0;
    npages_file = 0;
    npages_pinned = 0;
}

static int trigger_set(const char *val, const struct kernel_param *kp) {
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
        ret = do_fragment();
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
    return enable_shrinker ? npages_total : 0;
}

// Free a random subset of `nr_to_scan` pages.
static unsigned long
frag_shrink_scan(struct shrinker *shrink, struct shrink_control *sc) {
    unsigned long freed = 0;

    printk(KERN_WARNING "frag: shrinking...\n");

    // Avoid deadlocks where we are trying to allocate and shrink at the same
    // time. We should only enable the shrinker after we are done allocating
    // and fragmenting memory.
    if (!enable_shrinker) return 0;

    spin_lock(&allocation_lock);

    // The free lists are already randomized, so we can free a random subset by
    // freeing pages from the head of the list.

    // Figure out what we will reclaim. Initially we want to reclaim file
    // memory and single anon pages. Then we will break apart THP pages.
    while ((npages_file > 0 || npages_anon > 0 || npages_anon_thp > 0)
            && freed < sc->nr_to_scan)
    {
        switch (prandom_u32() % 2) {
            case 0: // file
                if (npages_file > 0) {
                    free_first_page(&pages_file);
                    npages_file -= 1;
                    freed += 1;
                    break;
                }
                // fall through
                // if we don't have file pages.

            case 1: // anon
                if (npages_anon > 0) {
                    free_first_page(&pages_anon);
                    npages_anon -= 1;
                    freed += 1;
                    break;
                }
                // fall through
                // if we don't have single anon pages.

            case 2: // anon thp -- will never happen on it's own. Only happens
                    // by fall-through.
                if (npages_anon_thp > 0) {
                    free_first_page(&pages_anon_thp);
                    npages_anon_thp -= 1;
                    freed += 1;
                }
                break;

            default:
                BUG();
        }
    }

    spin_unlock(&allocation_lock);

    return freed;
}

static struct shrinker frag_shrinker = {
    .count_objects = frag_shrink_count,
    .scan_objects = frag_shrink_scan,
};

////////////////////////////////////////////////////////////////////////////////
// Module init/exit.

static int mod_init(void) {
    int ret;

    printk(KERN_WARNING "frag: Init.\n");

    memset(profile_str, 0, PROFILE_STR_MAX);
    ent = proc_create("sumf", 0660, NULL, &profile_ops);

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
