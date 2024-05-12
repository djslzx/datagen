from dpp import sweep, run_ant_search_from_conf

if __name__ == "__main__":
    sweep("./configs/mcmc-ant.yaml", run_ant_search_from_conf)


