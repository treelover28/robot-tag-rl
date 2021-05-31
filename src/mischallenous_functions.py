
def _get_permutations(current_list_index, lists, prefix, k, states_accumulator):
    """ Helper function to generate all permutations of length n with replacements
    using elements in choices list. The prefix is the current partial permutation
    and k indicates how many more elements we still need to add to current partial permuation
    to finish it.

    Args:
        choices: list of elements to create permuation from
        prefix: contains the partial permuation
        k: (int) number of element left to add to current partial permutation to get it up to desireable length
        states_accumulator ([type]): list to accumulate / store the states.
    """
    if (k == 0):
        states_accumulator.append(prefix)
        return
    
    list_to_select_from = lists[current_list_index]
   
    for i in range(len(list_to_select_from)):
        new_prefix = (prefix + [list_to_select_from[i]])
        _get_permutations(current_list_index + 1, lists, new_prefix, k-1, states_accumulator)



