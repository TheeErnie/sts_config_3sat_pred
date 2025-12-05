import sys
import re
import random
import math

# creates random seed for scrambling
def generate_random_seed(order: int) -> int:
    max_seed = 2 ** int(3 * order * (order - 1) / 6)
    return random.randint(0,max_seed)


# creates random seed for scrambling with the binary representation having the an equal number of 1 digits as the num_negations
def generate_num_negations_seed(order: int, num_negations: int) -> int:
    max_seed = 2 ** int(3 * order * (order - 1) / 6)
    
    # helper to ensure uniform distribution of literal negations
    def count_with_limit(n, bits, ones):
        if ones < 0:
            return 0
        if bits == 0:
            return 1 if ones == 0 else 0

        msb = 1 << (bits - 1)
        if n < msb:
            # top bit is 0, so we can't set it
            return count_with_limit(n, bits - 1, ones)
        else:
            # two options: put 0 in top bit, or put 1
            without = math.comb(bits - 1, ones)  # top=0, free placement below
            with1  = count_with_limit(n - msb, bits - 1, ones - 1)
            return without + with1
        
    # grab random seed from 0 to n with x number of negations
    bits = max_seed.bit_length()
    result = 0
    remaining_ones = num_negations
    remaining_n = max_seed

    for b in range(bits, 0, -1):
        msb = 1 << (b - 1)
        if remaining_n < msb:
            # must place 0
            continue
        # Option 1: place 0 here
        count0 = math.comb(b - 1, remaining_ones)
        # Option 2: place 1 here
        count1 = count_with_limit(remaining_n - msb, b - 1, remaining_ones - 1)
        total = count0 + count1
        if total == 0:
            break
        choice = random.randrange(total)
        if choice < count0:
            # put 0, nothing added to result
            remaining_n = msb - 1  # now limit shrinks to all 0s below
        else:
            # put 1
            result |= msb
            remaining_ones -= 1
            remaining_n -= msb

    return result


# reads in .cnf file and returns the file header with an of the clauses
def parse_doc(read): # read in file
    data = []
    header = read.readline().rstrip() # split header from data

    for readline in read: # add each 3sat and tear off the end line 0
        line = [int(item) for item in readline.rstrip().split(" ")]
        line.pop()
        data.append([item for item in line])
    return header, data
# output sample : ('p cnf 9 12', [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [2, 6, 7], [1, 5, 9], [3, 4, 8], [2, 4, 9], [3, 5, 7], [1, 6, 8]])


# scrambles the literals by adding negations based on the input seed and returns list of clauses
def scramble_literals(data, seed, meta): # swap signs of literals based on seed
    bin_seed_str = bin(int(seed))[2:] # convert seed to binary
    meta_split = meta.split()
    num_literals = 3 * int(meta_split[3])

    max_seed = 2 ** (3 * int(meta_split[3])) - 1 # ensure no bad seeds
    if int(seed) > max_seed or int(seed) < 0:
        raise RuntimeError(f"Seed: {seed} exceeds expected value of {max_seed}")
    
    while len(bin_seed_str) < num_literals: # pad the seed to be the same length as the number of literals
        bin_seed_str = "0" + bin_seed_str

    scrambled_literals = data
    # turn all literals that align with a 1 in the binary representation of the seed negative
    # *** UPDATE: No longer assumes all literals start positive
    # ***         Now wipes literal negations before altering
    for i in range(len(bin_seed_str)): 
        if bin_seed_str[i] == "1":
            scrambled_literals[i // 3][i % 3] = abs(scrambled_literals[i // 3][i % 3]) * -1
        else:
            scrambled_literals[i // 3][i % 3] = abs(scrambled_literals[i // 3][i % 3])

    return scrambled_literals


# writes the new data to the given file with prebuilt utility to skip the output file path by giving a system number
def write_scrambled(meta, data, system_number, write_file_path=None): # write scrambled 3sat to file
    order = int(next(x for x in meta.split() if x.isdigit()))

    if write_file_path == None:
        write_file_path = f"/work/luduslab/sts_3sat/sts3sat_conversion/sat_problems/order_{order}/sat3_order{order}_scrambled_no{system_number}.cnf"

    with open(write_file_path, "w") as file:
        file.write(f"{meta}\n") # write header
        for line in data:       # write lines
            file.write(" ".join(map(str,line + ["0"])) + "\n")


# scrambles negation of literals from base_file_path and sends new .cnf to write_file_path
def packaged_scrambler(base_file_path: str, seed: int, write_file_path=None):
    if write_file_path == None:
        write_file_path = base_file_path
    with open(base_file_path, 'r') as f:
        header, data = parse_doc(f)
        scrambled_literals = scramble_literals(data, seed, header)
        match = re.search(r"_no(d+)", base_file_path)
        if match:
            system_number = int(match.group(1))
        else: 
            system_number = None

        write_scrambled(header, scrambled_literals, system_number, write_file_path=write_file_path)


# packaged scrambler that takes in the additional parameter of a specific proportion of literals to be negated in place of seed
def proportional_packaged_scrambler(in_file_path: str, out_file_path: str, pos_neg_literal_prop: float):
    with open(in_file_path, 'r') as f:
        header, data = parse_doc(f)

        # get number of literals to negate
        order = int(header.split()[2])
        num_literals = 3 * order * (order - 1) / 6
        num_negations = int(num_literals * pos_neg_literal_prop)

        # scramble the literals according to pos_neg_literal_prop value
        seed = generate_num_negations_seed(order=order, num_negations=num_negations)
        scrambled_literals = scramble_literals(data=data, seed=seed, meta=header)

        write_scrambled(meta=header, data=scrambled_literals, system_number=None, write_file_path=out_file_path)

    return

if __name__ == "__main__":
    if len(sys.argv) == 3:
        with open(sys.argv[1], "r") as f:
            header, data = parse_doc(f)
            scrambled_literals = scramble_literals(data, sys.argv[2], header)
            match = re.search(r"_no(\d+)", sys.argv[1])
            if match:
                system_number = int(match.group(1))
            else:
                raise RuntimeError("File name misread. Expecting file to end with /'_nox.cnf/' where x is an integer")
            write_scrambled(header, scrambled_literals, system_number)
    else:
        print("Please provide base file and seed number")