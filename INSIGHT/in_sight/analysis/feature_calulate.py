def compute_amplification_error_rate(con, phasing_block):
    # Check if phasing_block is unusable ('()' or '')
    if phasing_block in ["()", ""]:
        return None
    
    # Parsing phasing block to numeric
    # order is: j1, k2, j2, k1
    phasing_block = phasing_block.strip("()")
    phasing_block_num = list(map(float, phasing_block.split(",")))
    
    # Extract values in the order j-h1, k-h2, j-h2, k-h1
    j1 = phasing_block_num[0]
    k2 = phasing_block_num[1]
    j2 = phasing_block_num[2]
    k1 = phasing_block_num[3]
    
    # Initialize result
    result = None
    
    # Compute based on the condition
    if con == 'j1k2':
        # Compute equation is j2 / (k2 + j2)
        if (k2 + j2) != 0:
            result = j2 / (k2 + j2)
        else:
            result = None
    elif con == 'j2k1':
        # Compute equation is j1 / (k1 + j1)
        if (k1 + j1) != 0:
            result = j1 / (k1 + j1)
        else:
            result = None
    
    return result

# Example usage:
# f_df = {
#     'con_1': ['j1k2', 'j2k1'],
#     'phasing_block_1': ['(1,2,3,4)', '(5,6,7,8)']
# }

# print(compute_amplification_error_rate(f_df['con_1'][0], f_df['phasing_block_1'][0]))

def compute_phasing_error(con, phasing_block):
    # Check if phasing_block is unusable ('()' or '')
    if phasing_block in ["()", ""]:
        return None
    
    # Parsing phasing block to numeric
    # order is: j1, k2, j2, k1
    phasing_block = phasing_block.strip("()")
    phasing_block_num = list(map(float, phasing_block.split(",")))
    
    # Extract values in the order j-h1, k-h2, j-h2, k-h1
    j1 = phasing_block_num[0]
    k2 = phasing_block_num[1]
    j2 = phasing_block_num[2]
    k1 = phasing_block_num[3]
    
    # Initialize result
    result = None

    # Compute based on the condition
    k = k1+k2
    if con == 'j1k2':
        # compute equation is k-h1/k
        if k != 0:
            result = k1 / k
        else:
            result = None
    elif con == 'j2k1':
        # compute equation is k-h2/k
        if k != 0:
            result = k2 / k
        else:
            result = None

    return result

# Example usage:
# print(compute_accurate_phasing('j1k2', '(1,2,3,4)'))
# print(compute_accurate_phasing('j2k1', '(1,2,3,4)'))
