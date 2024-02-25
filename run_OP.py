from optim_process_mobo_new import optim_process
# initialization for scheme
SCHEME = "teno6_dv"
scheme_setup = {
    "teno6_dv": {
        "para1_name": "eta_eno",
        "para1_range": [0.5, 1],
        "para2_name": "eta_v",
        "para2_range": [0, 1],
        "para3_name": "ducros_cutoff",
        "para3_range": [0, 1],
    },
}
# optim initialization
noise = 1e-3
n_init = 20
n_iter = 100
pareto = optim_process(scheme_setup, noise, SCHEME, n_init=n_init, n_iter=n_iter, normalization=False, DEBUG=False)