import copy


def get_subchains_from_indices(indices, x):
    subchains = []

    for i, sublist in enumerate(indices):
        subchains.append([])

        for j, k in enumerate(indices[i]):
            subchains[i].append(x[indices[i][j]])

    return subchains


def glue_subchains(indices, subchains, numDOF):
    x = numDOF * [0.0]

    for i in range(len(subchains)):
        for j in range(len(subchains[i])):
            x[indices[i][j]] = subchains[i][j]

    return x


def inject_subchain(x, x_subchain_idx, indices, subchains, numDOF):
    ret_subchains = copy.deepcopy(subchains)
    ret_subchains[x_subchain_idx] = x
    return glue_subchains(indices, ret_subchains, numDOF)


def inject_state(x, subchain, subchain_indices):
    x_ret = copy.deepcopy(x)
    for i, s in enumerate(subchain_indices):
        x_ret[s] = subchain[i]
    return x_ret


if __name__ == "__main__":
    x = [0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    indices = [[0, 2], [1, 3, 4], [8, 6, 7, 5]]
    subchains = [[0.0, 1.0], [2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]

    print(x)
