# Given a group G, construct a random generating sequence. This is likely to be shorter than a generating sequence
# found in some other way, and hence more efficient for constructing Cayley graphs.

def random_generating_sequence(G):
    l = []
    while G.subgroup(l).order() != G.order():
        l.append(G.random_element())
    return l

## CHANGES START ##
# Given a vector in $F^n$, compute the subgroup of G stabilizing this subspace for the *right* action.
def vec_stab_subspace(v, G):
    Ggap = G._gap_()
    vgap = libgap.Vector([c._gap_() for c in v])
    S = Ggap.Stabilizer(vgap)
    return G.subgroup(S.GeneratorsOfGroup())
## CHANGES END ##

# Given a generalized Cayley graph for a group G acting on a set of vertices, return a list of connected
# component representatives not containing any vertices listed in `exclude` or any vertices 
# for which `forbid` returns True, and a dictionary computing these representatives to 
# arbitrary vertices in their components.

def group_retract(G, vertices, edges, exclude=[], forbid=None):
    vertices = list(vertices)
    # Add dummy edges to link all excluded vertices.
    edges2 = [(exclude[0], exclude[i]) for i in range(1, len(exclude))]
    # Construct the digraph.
    Gamma = DiGraph([vertices, edges + edges2], loops=True, format='vertices_and_edges')
    # Check that we did not implicitly add any vertices.
    assert set(Gamma.vertices()) == set(vertices)
    # Compute connected components.
    conn = Gamma.connected_components(sort=False)
    forbidden_verts = []
    reps = []
    # Remove all components containing an excluded or forbidden vertex.
    for l in conn:
        if (exclude and exclude[0] in set(l)) or (forbid and forbid(l[0])):
            forbidden_verts += l
        else:
            reps.append(l[0])
    forbidden_verts = set(forbidden_verts)
    # Compute the retract on the remaining components.
    d = {M: (M, G(1)) for M in reps}
    queue = reps[:]
    while queue:
        M = queue.pop(0)
        assert M in d
        for (_, M1, g) in Gamma.outgoing_edge_iterator(M):
            if M1 not in d:
                queue.append(M1)
                d[M1] = (d[M][0], g*d[M][1])
        for (M1, _, g) in Gamma.incoming_edge_iterator(M):
            if M1 not in d:
                queue.append(M1)
                d[M1] = (d[M][0], ~g*d[M][1])
    assert all(M in d or M in forbidden_verts for M in vertices)
    return reps, d, forbidden_verts

# The data structure for orbit lookup trees of depth $n$:
# - The tree is a dictionary `tree` indexed by levels $0, \dots, n$.
# - Each level `tree[k]` is a dictionary keyed by a subspace represented as a matrix of rows, identified with nodes of the tree.
# - Each value `tree[k][U]` is a dictionary with the following keys:
#  - `gpel` (for $U$ eligible): a pair $(U', g)$ such that $U'$ is a green node and $g(U') = U$.
#  - `stab` (for $U$ green) the stabilizer of $U$.
#  - `retract` (for $U$ green and $k<n$): a dictionary whose value at $y \in S \setminus rowspan(U)$ (resp $y \in S/U$) is an element $g \in G_U$ such that $U \cup \{g^{-1}(y)\}$ (resp. $\pi_U^{-1}(g^{-1}(y))$) is a red or green node.

# Use an enhanced $n$-orbit tree to identify an orbit representative for the action of the group $G$ on $k$-subspaces.
# mats[:-1] should be in reduced row echelon form
def orbit_rep_from_tree_subspace(G, tree, mats, apply_group_elem, optimized_rep, find_green=True):
    n = mats.nrows()
    if n not in tree:
        raise ValueError("Tree not computed")
    if n == 0:
        return mats, optimized_rep(G(1))
    mats0 = mats[:-1]
    y = mats[-1]
    if mats0 in tree[n-1] and 'gpel' not in tree[n-1][mats0]: # Truncation is an ineligible node
        return None, None
    if mats0 in tree[n-1] and 'stab' in tree[n-1][mats0]: # Truncation is a green node
        assert 'retract' in tree[n-1][mats0]
        g0 = optimized_rep(G(1))
        mats0pan = mats0.row_space()
        assert y not in mats0pan
    else: # Truncation needs to be resolved
        mats0, g0 = orbit_rep_from_tree_subspace(G, tree, mats0, apply_group_elem, optimized_rep)
        mats0pan = mats0.row_space()
        if mats0 is None:
            return None, None
        assert 'gpel' in tree[n-1][mats0]
        assert 'retract' in tree[n-1][mats0]
        y = apply_group_elem(~g0, y)
    z, g1 = tree[n-1][mats0]['retract'][y]
    assert z not in mats0pan
    mats1 = mats0.stack(z).rref()
    mats1.set_immutable()
    if not find_green:
        return mats1, g0*g1
    if 'gpel' not in tree[n][mats1]:
        return None, None
    mats2, g2 = tree[n][mats1]['gpel']
    assert 'gpel' in tree[n][mats2]
    return mats2, g0*g1*g2

def subspace_binomial(n, k, q=2):
    """
    The number of k-subspaces of an n-dimensional space over F_q.
    """
    return prod(q^n - q^i for i in range(k)) // prod(q^k - q^i for i in range(k))

# Given an orbit lookup tree at depth $n$ (for the action of a finite group $G$ on a finite set $S$), extend it in place
# to depth $n+1$. For $n=0$, pass for `tree` an empty dict and it will be initialized correctly.
#
# The argument `methods` is a dictionary containing functions as specified:
# - `apply_group_elem`: given a pair $(g, x) \in G \times S$, returns $g(x)$.
# - `stabilizer`: given $x \in S$, returns a group whose intersection with $G$ (in some ambient group) is $G_x$.
# - `optimized_rep` (optional): given an element $g \in G$, return an optimized representation of $g$.

def tprint(verbose, msg, t0):
    if verbose:
        print(msg + f" ({cputime() - t0:.2f})")

def extend_orbit_tree_subspace(G, V, tree, methods, verbose=True, terminate=False):
    t0 = cputime()
    S = [matrix(v) for v in V if v]
    for M in S:
        M.set_immutable()
    apply_group_elem = methods['apply_group_elem']
    stabilizer = methods['stabilizer']
    optimized_rep = methods['optimized_rep'] if 'optimized_rep' in methods else lambda g: g
    forbid = methods['forbid'] if 'forbid' in methods else (lambda x, easy=False: False)
    if not tree: # Initialize
        S0 = matrix(V.base_field(), 0, V.dimension())
        S0.set_immutable()
        tree[0] = {S0: {'gpel': (S0, optimized_rep(G(1))), 'stab': []}}
    n = max(tree.keys())
    tprint(verbose, "Current level: {}".format(n), t0)
    tree[n+1] = {}
    for mats in tree[n]:
        if 'stab' in tree[n][mats]: # Green node
            # Compute the stabilizer of mats (deferred from the previous level).
            ## CHANGES START ##
            matspan = mats.row_space()
            vertices = [M for M in S if M not in matspan]
            tprint(verbose, "%s vertices" % len(vertices), t0)
            edges = []
            if n == 0:
                G1 = G0 = G
                gens = G.gens()
            else:
                # It's not clear to me that this is the right stabilizer now....
                parent = mats[:-1]
                endgen = mats[-1]
                staber = stabilizer(endgen, G)
                tprint(verbose, "Found stabilizer of order %s" % staber.order(), t0)
                G0 = tree[n-1][parent]['stab'].intersection(staber)
                tprint(verbose, "Found G0 of order %s" % G0.order(), t0)
                G1 = G.subgroup(list(G0.gens()) + tree[n][mats]['stab'])
                tprint(verbose, "Found G1 of order %s" % G1.order(), t0)
                if G1.order() == 1: # Early abort
                    tprint(verbose, "Trivial stabilizer", t0)
                    tmp = vertices
                    d = {M: (M, optimized_rep(G(1))) for M in vertices}
                else: # Construct the group retract under this green node.
                    gens = [optimized_rep(g) for g in random_generating_sequence(G1)]
                    assert all(apply_group_elem(g, M) in matspan for g in gens for M in mats)
                    tprint(verbose, "Number of generators: {}".format(len(gens)), t0)
                    G1 = G.subgroup(gens) # Is this secretly intersecting with G? We may not have that the previous G1 is contained in G
                    tprint(verbose, "G1 found", t0)
            if G1.order() != 1:
                for cnt, M in enumerate(vertices):
                    tprint(verbose and cnt and cnt%512==0, "Edges progress: %s" % cnt, t0)
                    for g in gens:
                        edges.append((M, apply_group_elem(g, M), g))
                #edges = [(M, apply_group_elem(g, M), g) for M in vertices for g in gens]
                ## CHANGES END ##
                tprint(verbose, "Edges computed", t0)
                tmp, d, _ = group_retract(G, vertices, edges)
                tprint(verbose, "Retract computed", t0)
            tree[n][mats]['stab'] = G1
            tree[n][mats]['retract'] = d
            for M in tmp:
                if M in matspan:
                    raise ValueError("Found repeated entry in tuple")
                mats1 = mats.stack(M)
                mats1.set_immutable()
                tree[n+1][mats1] = {}
    # If no forbidden vertices, check the orbit-stabilizer formula.
    if 'forbid' not in methods:
        N = G.order()
        ## CHANGED binomial TO subspace_binomial ##
        if not sum(N // tree[n][mats]['stab'].order() for mats in tree[n] if 'stab' in tree[n][mats]) == subspace_binomial(V.dimension(), n):
            raise RuntimeError("Error in orbit-stabilizer formula")
    tprint(verbose, "Number of new nodes: {}".format(len(tree[n+1])), t0)
    edges = []
    exclude = []
    ## CHANGES START ##
    if n > 0:
        # when n = 0 this process doesn't lead to any new edges
        dualspace = F^(n+1)
        dbasis = dualspace.basis()
        def oddify(v):
            for i in range(len(v)):
                if v[i] != 0:
                    return dbasis[i]
        splittings = [dualspace.hom(matrix(d).T).kernel().basis_matrix().stack(oddify(d)) for d in dualspace if d]
        # splittings is a list of matrices Y so that the top n-1 rows of Y*M runs over the n-1 dimensional subspaces of the row space of M, and the last row gives a vector not in their span.
        for mats in tree[n+1]:
            # mats should already be rrefed
            if forbid(mats, easy=True):
                exclude.append(mats)
            else:
                # We need to enumerate all of the subspaces of one less dimension that could lead do this one
                for Y in splittings:
                    i = Y * mats
                    i.set_immutable()
                    mats1, g1 = orbit_rep_from_tree_subspace(G, tree, i, apply_group_elem, optimized_rep, find_green=False)
                    if mats1 is None:
                        exclude.append(mats)
                    elif mats != mats1:
                        edges.append((mats1, mats, g1))
    ## CHANGES END ##
    tprint(verbose, "Edges computed", t0)
    tmp, d, forbidden_verts = group_retract(G, tree[n+1].keys(), edges, exclude, forbid)
    tprint(verbose, "Retract computed", t0)
    for mats in d:
        tree[n+1][mats]['gpel'] = d[mats]
    # Mark green nodes.
    for mats in tmp:
        tree[n+1][mats]['stab'] = []
    assert all('stab' in tree[n+1][tree[n+1][mats]['gpel'][0]]
                   for mats in tree[n+1] if 'gpel' in tree[n+1][mats])
    # Defer the computation of stabilizers, recording some elements read off of the graph.
    if terminate:
        tprint(verbose, "Stabilizer generators not computed", t0)
    else:
        for e in edges:
            if e[0] not in forbidden_verts:
                mats1, g1 = d[e[0]]
                mats2, g2 = d[e[1]]
                assert mats1 == mats2
                g = ~g2*e[2]*g1
                if g != G(1):
                    tree[n+1][mats1]['stab'].append(g)
        tprint(verbose, "Stabilizer generators found", t0)
    tprint(verbose, "Number of new green nodes: {}".format(
        sum(1 for mats in tree[n+1]
            if 'stab' in tree[n+1][mats])), t0)
    tprint(verbose, "New level: {}".format(max(tree.keys())), t0)

# Build an orbit lookup tree to depth n. By default, we do not record stabilizer generators at depth n,
# so the result cannot be used to extend to depth n+1.

def build_orbit_tree_subspace(G, V, n, methods, verbose=True, terminate=True):
    apply_group_elem = methods['apply_group_elem']
    optimized_rep = methods['optimized_rep'] if 'optimized_rep' in methods else lambda g: g
    tree = {}
    for i in range(n):
        extend_orbit_tree_subspace(G, V, tree, methods, verbose=verbose, terminate=(terminate and (i == n-1)))
    return tree

# Return a list of green nodes at depth $n$.
def green_nodes(tree, n):
    return [mats for mats in tree[n] if 'stab' in tree[n][mats]]
