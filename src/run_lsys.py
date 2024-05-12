from dpp import sweep, run_lsys_search

if __name__ == "__main__":
    sweep("./configs/mcmc-lsys.yaml", run_lsys_search)
