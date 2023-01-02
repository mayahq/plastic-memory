import numpy as np
arr = np.array([
    [1, 2, 3], 
    [0, 0, 0],
    [5, 6, 7]
], dtype=np.float)

# row_sums = arr.sum(axis=1, keepdims=True)
# #print(row_sums)
# new_matrix = arr / row_sums[:, np.newaxis]
# print(new_matrix)
a_norm = np.nan_to_num((arr/arr.sum(axis=1)[:,None]).round(7))
print(a_norm)