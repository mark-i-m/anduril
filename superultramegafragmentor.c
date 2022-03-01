/*
 * Module to fragment memory artificially according to some profile.
 */

#include <linux/module.h>
#include <linux/gfp.h>
#include <linux/vmalloc.h>
#include <linux/prandom.h>

MODULE_AUTHOR("Mark Mansi");
MODULE_LICENSE("Dual MIT/GPL");

////////////////////////////////////////////////////////////////////////////////
// State.

// Represents a single allocation.
struct allocation {
    // The first allocated page.
    struct page *pages;

    // Order of the allocation.
    u64 order;

    // Pointer to the next one.
    struct allocation *next;
};

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
    u64 nedges;
};

struct profile_edge {
    u64 from;
    u64 to;

    // an int from 0 to 100 representing a probability that this edge is taken.
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


// The whole set of `allocation`s as a singly-linked list.
static struct allocation *allocations = NULL;

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

// Profile to fragment memory with.
static struct profile *profile = NULL;
static int profile_set(const char *val, const struct kernel_param *kp);
static int profile_get(char *buffer, const struct kernel_param *kp);
static const struct kernel_param_ops profile_ops = {
    .set = profile_set,
    .get = profile_get,
};
module_param_cb(profile, &profile_ops, &profile, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH);

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

// Format for input:
// - All one line with nodes ended by `;`, including the last one.
// - No trailing or separating whitespace except as below.
// - The first node specified should be the source node.
// - Each node has the following format:
//      S F (T P)+;
//
//   All elements are u64, space separated.
//   S: size (in 4KB pages).
//   F: flags (in hex).
//   (T P)+: one or more outgoing edges.
//      T: the index of the node on the other side of the edge.
//      P: a u64 from 0 to 100 which is the probability of taking the edge.
//      Don't actually write the ( ) + characters... I was using regex notation.
//
//  Example: "1 0 0 50 1 50;3 0 0 100;"
//
//        +---+
//     .5 |   |  .5
//        +->[0]--->[1]
//            ^      |
//            +------+
//               1
//
static int profile_set(const char *val, const struct kernel_param *kp) {
    u64 nnodes = 0;
    u64 nedges = 0;
    int i, j;
    int ret = 0;
    const char *cursor = val;
    char *line_end = NULL;
    struct profile_node *node;
    struct profile_edge *edge;
    u64 discard, ecount = 0, prob_total;

    // Free any existing profile.
    if (profile) {
        vfree(profile->nodes);
        vfree(profile->edges);
        vfree(profile);
        profile = NULL;
    }

    printk(KERN_WARNING "frag: \"%s\"\n", val);

    // Count the number of nodes.
    for (i = 0; val[i] != 0; ++i) {
        if (val[i] == ';') ++nnodes;
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
        ret = profile_parse_u64(&cursor, line_end, 10, &node->npages);
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
        node->nedges = ecount / 2;
        nedges += node->nedges;

        // printk(KERN_WARNING "frag: nedges so far %llu\n", nedges);

        if (node->nedges == 0) {
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

    cursor = val;
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

        if (prob_total > 100) {
            printk(KERN_ERR "frag: probabilities add to %llu > 100\n", prob_total);
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

static int profile_get(char *buffer, const struct kernel_param *kp) {
    u64 i;
    int len;
    struct profile_node *node;
    struct profile_edge *e;

    // No profile loaded.
    if (profile == NULL) {
        return sprintf(buffer, "NULL\n");
    }

    // Profile loaded. Print a list of nodes first.
    for (i = 0; i < profile->nnodes; ++i) {
        node = &profile->nodes[i];

        // printk(KERN_WARNING "frag: %llu %llu %llu\n", i, profile->nnodes, profile->nedges);

        len += sprintf(buffer + len,
                "%llu: size=%llu flags=%llx edges=",
                i, node->npages, node->flags);

        buffer[len] = 0;

        profile_for_each_edge(profile, node, e) {
            len += sprintf(buffer + len,
                    "(%llu %llu %llu) ",
                    e->from, e->to, e->prob);
        }

        buffer[len] = 0;

        len += sprintf(buffer + len, "\n");

        buffer[len] = 0;
    }

    return len;
}

////////////////////////////////////////////////////////////////////////////////
// The Meat.

// Do a random walk over the markov process until we have allocated the needed
// amount of memory.
static int do_fragment(void) {
    u64 pages_sofar = 0;
    struct profile_node *current_node, *n;
    struct profile_edge *e;
    struct allocation *alloc;
    u32 rand, walk;

    if (!profile) {
        printk(KERN_ERR "frag: no profile\n");
        return -EINVAL;
    }

    // Start at node 0 and then walk randomly.
    current_node = &profile->nodes[0];

    while(pages_sofar < npages) {
        printk(KERN_WARNING "frag: random node %lu\n",
                profile_node_idx(profile, current_node));

        // Do the allocation at current node.
        alloc = vzalloc(sizeof(struct allocation));
        if (!alloc) {
            return -ENOMEM;
        }
        // TODO: npages is not the same as order...
        alloc->order = current_node->npages;
        // TODO: flags
        alloc->pages = alloc_pages(GFP_KERNEL, alloc->order);
        if (!alloc->pages) {
            return -ENOMEM;
        }
        alloc->next = allocations;
        allocations = alloc;

        // TODO: figure out order vs npages...
        pages_sofar += 1 << alloc->order;

        // Then, randomly walk to a neighbor.
        rand = prandom_u32() % 100;
        walk = 0;
        profile_for_each_edge(profile, current_node, e) {
            n = &profile->nodes[e->to];
            walk += e->prob;
            if (rand < walk) break;
        }
        printk(KERN_WARNING "frag: rand=%u walk=%u\n", rand, walk);
        current_node = n;
    }
    
    printk(KERN_WARNING "frag: The deed is done.\n");

    return 0;
}

static int trigger_set(const char *val, const struct kernel_param *kp) {
    // Parse value.
    int ret = param_set_int(val, kp);
    if (ret != 0) return ret;

    // Trigger fragmentation.
    ret = do_fragment();

    // Reset trigger.
    trigger = 0;

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// Module init/exit.

static int mod_init(void) {
    printk(KERN_WARNING "frag: Init.\n");
    return 0;
}
module_init(mod_init);

static void mod_exit(void) {
    struct allocation *alloc;

    // Free all allocations.
    while (allocations) {
        alloc = allocations;
        allocations = alloc->next;

        printk(KERN_WARNING "frag: free(pages%p, size=%llu)\n",
                alloc->pages, alloc->order);

        __free_pages(alloc->pages, alloc->order);
        vfree(alloc);
    }

    printk(KERN_WARNING "frag: Exit.\n");
}
module_exit(mod_exit);
