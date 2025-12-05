from itertools import combinations, permutations

# check if list of clauses is an sts
def check_if_cnf_sts(clauses) -> bool:
    # convert to positive variable IDs only
    triples = [set(abs(x) for x in clause) for clause in clauses]
    
    # all triples must have size 3
    if not all(len(t) == 3 for t in triples):
        return False
    
    # get universe of elements
    elements = set().union(*triples)
    v = len(elements)
    
    # check STS necessary condition: v ≡ 1 or 3 (mod 6)
    if v % 6 not in (1, 3):
        return False
    
    # count pair occurrences
    pair_counts = {}
    for triple in triples:
        for a, b in combinations(triple, 2):
            pair = tuple(sorted((a, b)))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    # every pair from the universe must appear exactly once
    for a, b in combinations(elements, 2):
        pair = (a, b)
        if pair_counts.get(pair, 0) != 1:
            return False
    
    return True


# pull in all variables and ensure all are positive values
def read_from_file(file_path: str):
    clauses = []
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('p'):
                continue

            nums = line.split()
            nums = [abs(int(x)) for x in nums if x != '0']
            if len(nums) != 3:
                raise ValueError(f'All clauses must have 3 literals. The following file does not follow this: {file_path}')
            clauses.append(set(nums))
    return clauses


# pull in all variables without care for positive values
def read_from_file_raw(file_path: str):
    clauses = []
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('p'):
                continue

            nums = line.split()
            nums = [int(x) for x in nums if x != '0']
            if len(nums) != 3:
                raise ValueError(f'All clauses must have 3 literals. The following file does not follow this: {file_path}')
            clauses.append(nums)
    return clauses


# take in negated clauses from read_from_file_raw and transfer into all positive clauses for config counting
def negated_clause_to_all_pos(clauses):
    clause_copy = [row[:] for row in clauses]
    for clause in clause_copy:
        clause = [abs(lit) for lit in clause]
    order = max(max(clause) for clause in clause_copy)
    
    for i in range(len(clauses)):
        for j in range(len(clauses[0])):
            if clauses[i][j] < 0:
                clauses[i][j] = clauses[i][j] + 2 * abs(clauses[i][j]) + order
        #clauses[i] = set(clauses[i])

    return clauses


# pull in sts from a file and return as a list of sets that represent the clauses
def cnf_to_sts(file_path: str):
    data = read_from_file(file_path=file_path)
    if not check_if_cnf_sts(data):
        raise ValueError(f'The following file is not an STS generated 3-SAT: {file_path}')
    
    return data


# count number of negated literals in cnf file
def negated_lit_count(file_path: str) -> int:
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('c') or line.startswith('p') or line.startswith('%'):
                continue
            
            tokens = line.strip().split()

            for token in tokens:
                try:
                    num = int(token)
                    if num < 0:
                        count += 1
                except ValueError:
                    continue
    return count


# create lookup table structure
def sts_lookup_table(clauses):
    num_literals = max(max(clause) for clause in clauses)
    lookup_table = []
    for i in range(num_literals):
        lookup_table.append([0] * num_literals)
        lookup_table[i][i] = i + 1

    for clause in clauses:
        a = clause.pop()
        b = clause.pop()
        c = clause.pop()

        lookup_table[a-1][b-1] = c
        lookup_table[b-1][a-1] = c
        lookup_table[a-1][c-1] = b
        lookup_table[c-1][a-1] = b
        lookup_table[b-1][c-1] = a
        lookup_table[c-1][b-1] = a

    return lookup_table


# package read_from_file_raw, negated_clause_to_all_pos, and sts_lookup_table
def sts3sat_lookup_table_from_file(file_path):
    clauses = read_from_file_raw(file_path)
    clauses = negated_clause_to_all_pos(clauses)
    return sts_lookup_table(clauses)


# count pasch configurations using lookup table rather than list of clauses
def count_pasch_configurations_lt(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for a in range(1,order+1):
        for b in range(a+1,order+1):
            for f in range(b+1,order+1):
                if lookup_table[a-1][b-1] == f: # skip if a,b,f create 1 block/clause
                    break

                e = lookup_table[a-1][b-1]
                c = lookup_table[a-1][f-1]
                if e == 0 or c == 0:
                    break
                is_pasch = lookup_table[e-1][f-1] == lookup_table[b-1][c-1] and lookup_table[e-1][f-1] != 0

                if is_pasch:
                    d = lookup_table[b-1][c-1]
                    config = {
                        frozenset({a,b,e}),
                        frozenset({a,f,c}),
                        frozenset({b,c,d}),
                        frozenset({d,e,f})}
                    if config not in found_configs:
                        found_configs.append(config)

    return len(found_configs)


# count mitre configurations in an sts given a lookup table
def count_mitre_configurations(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for a in range(1,order+1):
        for c in range(a+1,order+1):
            for e in range(c+1,order+1):
                if lookup_table[a-1][c-1] == e:
                    break 

                f = lookup_table[a-1][c-1]
                g = lookup_table[e-1][f-1]
                b = lookup_table[a-1][g-1]
                if f == 0 or g == 0 or b == 0:
                    break
                is_mitre = lookup_table[g-1][c-1] == lookup_table[b-1][e-1] and lookup_table[g-1][c-1] != 0

                if is_mitre:
                    d = lookup_table[b-1][e-1]
                    config = {
                        frozenset({a,c,f}),
                        frozenset({a,b,g}),
                        frozenset({b,d,e}),
                        frozenset({c,d,g}),
                        frozenset({e,f,g}),
                    }
                    if config not in found_configs:
                        found_configs.append(config)
    
    return len(found_configs)


# count fano line configurations in an sts given a lookup table
def count_fano_line_configurations(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for c in range(1,order+1):
        for e in range(c+1,order+1):
            for f in range(e+1,order+1):
                if lookup_table[c-1][e-1] == f:
                    break

                a = lookup_table[e-1][f-1]
                b = lookup_table[c-1][e-1]
                d = lookup_table[c-1][f-1]
                if a == 0 or b == 0 or d == 0:
                    break
                is_fano = lookup_table[a-1][c-1] == lookup_table[b-1][f-1] and lookup_table[a-1][c-1] == lookup_table[d-1][e-1] and lookup_table[a-1][c-1] != 0

                if is_fano:
                    g = lookup_table[a-1][c-1]
                    config = {
                        frozenset({a,c,g}),
                        frozenset({a,e,f}),
                        frozenset({b,c,e}),
                        frozenset({b,f,g}),
                        frozenset({c,d,f}),
                        frozenset({d,e,g})
                    }
                    if config not in found_configs:
                        found_configs.append(config)
    return len(found_configs)


# count hexagon configurations in an sts given a lookup table
def count_hexagon_configurations(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for b in range(1,order+1):
        for c in range(b+1,order+1):
            for d in range(c+1,order+1):
                if lookup_table[b-1][c-1] == d:
                    break

                a = lookup_table[b-1][c-1]
                h = lookup_table[b-1][d-1]
                e = lookup_table[a-1][d-1]
                g = lookup_table[c-1][h-1]
                if a == 0 or h == 0 or e == 0 or g == 0:
                    break
                is_hex = lookup_table[a-1][g-1] == lookup_table[e-1][h-1] and lookup_table[a-1][g-1] != 0

                if is_hex:
                    f = lookup_table[a-1][g-1]
                    config = {
                        frozenset({a,b,c}),
                        frozenset({e,h,f}),
                        frozenset({b,d,h}),
                        frozenset({a,d,e}),
                        frozenset({a,g,f}),
                        frozenset({c,g,h})
                    }
                    if config not in found_configs:
                        found_configs.append(config)
    return len(found_configs)


# count crown configurations in an sts given a lookup table
def count_crown_configurations(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for e in range(1,order+1):
        for f in range(e+1,order+1):
            for g in range(f+1,order+1):
                if lookup_table[e-1][f-1] == g:
                    break

                a = lookup_table[f-1][g-1]
                d = lookup_table[e-1][g-1]
                h = lookup_table[e-1][f-1]
                b = lookup_table[d-1][f-1]
                if a == 0 or d == 0 or h == 0 or b == 0:
                    break
                is_crown = lookup_table[a-1][b-1] == lookup_table[g-1][h-1] and lookup_table[a-1][b-1] != 0

                if is_crown:
                    c = lookup_table[a-1][b-1]
                    config = {
                        frozenset({a,b,c}),
                        frozenset({a,f,g}),
                        frozenset({b,d,f}),
                        frozenset({c,h,g}),
                        frozenset({d,e,g}),
                        frozenset({e,f,h})
                    }
                    if config not in found_configs:
                        found_configs.append(config)
    return len(found_configs)


# count prism configurations in an sts given a lookup table
def count_prism_configurations(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for a in range(1,order+1):
        for b in range(a+1,order+1):  
            for d in range(b+1,order+1):
                if lookup_table[a-1][b-1] == d:
                    break
                for f in range(d+1,order+1):
                    if lookup_table[a-1][b-1] == f or lookup_table[a-1][d-1] == f or lookup_table[b-1][d-1] == f:
                        break
                    
                    c = lookup_table[b-1][f-1]
                    e = lookup_table[a-1][b-1]
                    g = lookup_table[a-1][d-1]
                    i = lookup_table[f-1][g-1]
                    if c == 0 or e == 0 or g == 0 or i == 0:
                        break
                    is_prism = lookup_table[c-1][e-1] == lookup_table[d-1][i-1] and lookup_table[c-1][e-1] != 0

                    if is_prism:
                        h = lookup_table[c-1][e-1]
                        config = {
                            frozenset({a,b,e}),
                            frozenset({e,c,h}),
                            frozenset({h,i,d}),
                            frozenset({d,g,a}),
                            frozenset({b,c,f}),
                            frozenset({f,i,g})
                        }
                        if config not in found_configs:
                            found_configs.append(config)
    return len(found_configs)


# count grid configurations in an sts given a lookup table
def count_grid_configurations(lookup_table):
    found_configs = []
    order = len(lookup_table)

    for a in range(1,order+1):
        for b in range(a+1,order+1):
            for c in range(b+1,order+1):
                if lookup_table[a-1][b-1] == c:
                    break
                for e in range(c+1,order+1):
                    if lookup_table[a-1][b-1] == e or lookup_table[a-1][c-1] == e or lookup_table[b-1][c-1] == e:
                        break

                    d = lookup_table[a-1][b-1]
                    h = lookup_table[c-1][e-1]
                    g = lookup_table[a-1][e-1]
                    f = lookup_table[b-1][c-1]
                    if d == 0 or h == 0 or g == 0 or f == 0:
                        break
                    is_grid = lookup_table[d-1][h-1] == lookup_table[g-1][f-1] and lookup_table[d-1][h-1]

                    if is_grid:
                        i = lookup_table[d-1][h-1]
                        config = {
                            frozenset({a,b,d}),
                            frozenset({e,c,h}),
                            frozenset({g,f,i}),
                            frozenset({a,e,g}),
                            frozenset({b,c,f}),
                            frozenset({d,h,i})
                        }
                        if config not in found_configs:
                            found_configs.append(config)
    return len(found_configs)


# count pasch configurations in 3-SAT file WITHOUT lookup table structure
def count_pasch_configurations(clauses):
    """
    Count Pasch configurations in an STS given as a list (or iterable)
    of 3-element sets. Returns an integer count.
    Time complexity: O(v^3) where v = number of points.
    """
    # normalize blocks to frozensets for hashing/lookup
    blocks = [frozenset(b) for b in clauses]
    if any(len(b) != 3 for b in blocks):
        raise ValueError("All blocks must be 3-element sets")

    blocks_set = set(blocks)
    elements = set().union(*blocks) if blocks else set()

    # build map: unordered pair -> the unique block containing that pair (if any)
    pair_to_block = {}
    for blk in blocks:
        for p in combinations(blk, 2):
            key = tuple(sorted(p))
            pair_to_block[key] = blk

    seen = set()   # to avoid double-counting same Pasch
    count = 0

    # iterate over actual blocks (≈ O(v^2)), then over ordered choices inside the block (constant),
    # then over d in remaining points (O(v)) => overall O(v^3)
    for triple in blocks_set:
        # choose which element of the triple acts as 'a' and order the remaining as (b,c)
        for a in triple:
            others = [x for x in triple if x != a]
            for b, c in permutations(others, 2):
                for d in elements - triple:
                    # block containing pair (a,d) --> gives e
                    key_ad = tuple(sorted((a, d)))
                    blk_ad = pair_to_block.get(key_ad)
                    if not blk_ad:
                        continue
                    e_candidates = set(blk_ad) - {a, d}
                    if len(e_candidates) != 1:
                        continue
                    e = next(iter(e_candidates))

                    # block containing pair (b,d) --> gives f
                    key_bd = tuple(sorted((b, d)))
                    blk_bd = pair_to_block.get(key_bd)
                    if not blk_bd:
                        continue
                    f_candidates = set(blk_bd) - {b, d}
                    if len(f_candidates) != 1:
                        continue
                    f = next(iter(f_candidates))

                    # check final required block {f, c, e}
                    if frozenset([f, c, e]) not in blocks_set:
                        continue

                    pasch_blocks = frozenset([
                        triple,
                        frozenset([a, d, e]),
                        frozenset([f, b, d]),
                        frozenset([f, c, e]),
                    ])

                    if pasch_blocks not in seen:
                        seen.add(pasch_blocks)
                        count += 1

    return count
