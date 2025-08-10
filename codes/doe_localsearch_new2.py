

doe_localsearch = {
    'long': {
        'local_search_doe': 'long',
        'local_search_num_bitflips': 1,
        'local_search_maxiter': None,
        'local_search_maxepoch': 1000,
        'local_search_maxfevals': 2**15
    },
    'fast': {
        'local_search_doe': 'fast',
        'local_search_num_bitflips': 1,
        'local_search_maxiter': None,
        'local_search_maxepoch': 1000,
        'local_search_maxfevals_per_variable': 2
    },
    'vns': {
        'local_search_doe': 'vns_k3',
        "local_search_vns_k_max": 3,
        "local_search_num_bitflips": 1
    },


    'qaoa_p1': {
        'ansatz': 'QAOA',
        'ansatz_params': {'reps': 1}, 
        'local_search_doe': 'vns_k3', 
        "local_search_vns_k_max": 3,
        "local_search_num_bitflips": 1
    }
}

