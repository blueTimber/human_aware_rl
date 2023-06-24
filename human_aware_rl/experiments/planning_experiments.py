from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS
from overcooked_ai.overcooked_ai_py.agents.agent import CoupledPlanningAgent, CoupledPlanningPair
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import EmbeddedPlanningAgent, AgentPair

from overcooked_ai_py.utils import mean_and_std_err

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved


def get_delivery_horizon(layout):
    if layout == "simple" or layout == "random1":
        return 2
    if layout == "random0":
        return 2
    return 3

def P_BC_evaluation_for_layout(ae, layout, best_bc_models, num_games, delivery_horizons, omit_dict, set_history, seen_buffer, return_best, bc_stochastic, dist_heur):

    if layout in delivery_horizons:
        delivery_horizon = delivery_horizons[layout]
    else:
        delivery_horizon = get_delivery_horizon(layout)
    print("Delivery horizon for layout {}: {}".format(layout, delivery_horizon))

    layout_p_bc_eval = {"scores": {}, "states": {}}

    #######################
    # P_BC_test + BC_test #
    #######################
    p_bc_test_bc_name = 'P_BC_test+BC_test_0'
    bc_p_bc_test_name = 'P_BC_test+BC_test_1'

    # Prepare BC_test
    test_model_name = best_bc_models["test"][layout]
    agent_bc_test, _ = get_bc_agent_from_saved(test_model_name)
    agent_bc_test.stochastic = bc_stochastic
    agent_bc_test.logging_level = 0
    
    # Prepare P_BC_test (making another copy of BC_test just to be embedded in P_BC)
    agent_bc_test_embedded, _ = get_bc_agent_from_saved(test_model_name)
    agent_bc_test_embedded.stochastic = False
    agent_bc_test_embedded.logging_level = 0
    p_bc_test = EmbeddedPlanningAgent(agent_bc_test_embedded, agent_bc_test_embedded.mlp, ae.env, 
                                      delivery_horizon=delivery_horizon, set_history=set_history, seen_buffer=seen_buffer, return_best=return_best,
                                      dist_heur=dist_heur,
                                      logging_level=0)
    p_bc_test.debug = True

    print("Drop locations: ", agent_bc_test_embedded.mlp.ml_action_manager.counter_drop)
    print("Pickup locations: ", agent_bc_test_embedded.mlp.ml_action_manager.counter_pickup)
    print("Valid counters: ", agent_bc_test_embedded.mlp.ml_action_manager.motion_planner.counter_goals)
    
    # Execute runs
    if p_bc_test_bc_name not in omit_dict:
        ap_training = AgentPair(p_bc_test, agent_bc_test)
        data0 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
        layout_p_bc_eval["scores"][p_bc_test_bc_name] = mean_and_std_err(data0['ep_returns'])
        layout_p_bc_eval["states"][p_bc_test_bc_name] = mean_and_std_err(data0['ep_states'])

    if bc_p_bc_test_name not in omit_dict:
        ap_training = AgentPair(agent_bc_test, p_bc_test)
        data1 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
        layout_p_bc_eval["scores"][bc_p_bc_test_name] = mean_and_std_err(data1['ep_returns'])
        layout_p_bc_eval["states"][bc_p_bc_test_name] = mean_and_std_err(data1['ep_states'])

    if p_bc_test_bc_name not in omit_dict and bc_p_bc_test_name not in omit_dict:
        print("P_BC_test + BC_test", mean_and_std_err(data0['ep_returns']), mean_and_std_err(data1['ep_returns']))
        print("States", mean_and_std_err(data0['ep_states']), mean_and_std_err(data1['ep_states']))


    ########################
    # P_BC_train + BC_test #
    ########################
    p_bc_train_bc_name = 'P_BC_train+BC_test_0'
    bc_p_bc_train_name = 'P_BC_train+BC_test_1'

    # Prepare P_BC_train
    train_model_name = best_bc_models["train"][layout]
    agent_bc_train_embedded, _ = get_bc_agent_from_saved(train_model_name)
    agent_bc_train_embedded.stochastic = False
    p_bc_train = EmbeddedPlanningAgent(agent_bc_train_embedded, agent_bc_train_embedded.mlp, ae.env, 
                                       delivery_horizon=delivery_horizon, set_history=set_history, seen_buffer=seen_buffer, return_best=return_best,
                                       dist_heur=dist_heur)
    p_bc_train.debug = True
    
    # Execute runs
    if p_bc_train_bc_name not in omit_dict:
        ap_testing = AgentPair(p_bc_train, agent_bc_test)
        data0 = ae.evaluate_agent_pair(ap_testing, num_games=num_games, display=True)
        layout_p_bc_eval["scores"][p_bc_train_bc_name] = mean_and_std_err(data0['ep_returns'])
        layout_p_bc_eval["states"][p_bc_train_bc_name] = mean_and_std_err(data0['ep_states'])
    
    if bc_p_bc_train_name not in omit_dict:
        ap_testing = AgentPair(agent_bc_test, p_bc_train)
        data1 = ae.evaluate_agent_pair(ap_testing, num_games=num_games, display=True)
        layout_p_bc_eval["scores"][bc_p_bc_train_name] = mean_and_std_err(data1['ep_returns'])
        layout_p_bc_eval["states"][bc_p_bc_train_name] = mean_and_std_err(data1['ep_states'])

    if p_bc_train_bc_name not in omit_dict and bc_p_bc_train_name not in omit_dict:
        print("P_BC_train + BC_test", mean_and_std_err(data0['ep_returns']), mean_and_std_err(data1['ep_returns']))
        print("States", mean_and_std_err(data0['ep_states']), mean_and_std_err(data1['ep_states']))

    return layout_p_bc_eval

def P_BC_evaluation(best_bc_models, layouts, num_games=1, horizon=400, counter_dict={}, omit_dict={}, delivery_horizons={}, set_history=False, seen_buffer=1, 
                    return_best=False, bc_stochastic=False, dist_heur=False):

    p_bc_evaluation = {}

    for layout in layouts:
        mdp_params = {"layout_name": layout}
        env_params = {"horizon": horizon}
        
        mlp_params = NO_COUNTERS_PARAMS
        if layout in counter_dict:
            mlp_params['counter_goals'] = counter_dict[layout]
            mlp_params['counter_drop'] = counter_dict[layout]
            mlp_params['counter_pickup'] = counter_dict[layout]

        ae = AgentEvaluator(mdp_params, env_params, mlp_params=mlp_params)
        p_bc_evaluation[layout] = P_BC_evaluation_for_layout(ae, layout, 
                                                             best_bc_models, num_games=num_games, delivery_horizons=delivery_horizons, 
                                                             omit_dict=omit_dict, set_history=set_history, seen_buffer=seen_buffer, 
                                                             return_best=return_best, bc_stochastic=bc_stochastic, dist_heur=dist_heur)
    
    return p_bc_evaluation


def CP_evaluation(best_bc_models, layouts, num_games=1, horizon=400, counter_dict={}, omit_dict={}, delivery_horizons={}, dist_heur=False, debug=False):

    cp_evaluation = {}

    for layout in layouts:
        mdp_params = {"layout_name": layout}
        env_params = {"horizon": horizon}

        mlp_params = NO_COUNTERS_PARAMS
        if layout in counter_dict:
            mlp_params['counter_goals'] = counter_dict[layout]
            mlp_params['counter_drop'] = counter_dict[layout]
            mlp_params['counter_pickup'] = counter_dict[layout]

        ae = AgentEvaluator(mdp_params, env_params, mlp_params=mlp_params)
        cp_evaluation[layout] = CP_evaluation_for_layout(ae, layout, best_bc_models, num_games, delivery_horizons, omit_dict, dist_heur, debug=debug)
    
    return cp_evaluation

def CP_evaluation_for_layout(ae, layout, best_bc_models, num_games, delivery_horizons, omit_dict, dist_heur, debug=False):

    if layout in delivery_horizons:
        delivery_horizon = delivery_horizons[layout]
    else:
        delivery_horizon = get_delivery_horizon(layout)
    print("Delivery horizon for layout {}: {}".format(layout, delivery_horizon))

    layout_cp_eval = {"scores": {}, "states": {}}

    print("Drop locations: ", ae.mlp.ml_action_manager.counter_drop)
    print("Pickup locations: ", ae.mlp.ml_action_manager.counter_pickup)
    print("Valid counters: ", ae.mlp.ml_action_manager.motion_planner.counter_goals)

    ###########
    # CP + CP #
    ###########
    cp_pair_name = "CP+CP"

    # Prepare CP
    cp = CoupledPlanningAgent(ae.mlp, delivery_horizon=delivery_horizon, dist_heur=dist_heur, debug=debug)
    cp.env = ae.env

    if cp_pair_name not in omit_dict:
        print(cp_pair_name)
        # Prepare CP pair
        cp_pair = CoupledPlanningPair(cp, debug=debug)
        
        # Execute runs
        data = ae.evaluate_agent_pair(cp_pair, num_games=num_games, display=True)
        layout_cp_eval["scores"][cp_pair_name] = data['ep_returns'][0]
        layout_cp_eval["states"][cp_pair_name] = mean_and_std_err(data['ep_states'])
        print("CP + CP", data['ep_returns'][0])
        print("States", mean_and_std_err(data['ep_states']))

    ################
    # CP + BC_test #
    ################
    cp_bc_name = 'CP+BC_test_0'
    bc_cp_name = 'CP+BC_test_1'

    # Prepare BC_test
    test_model_name = best_bc_models["test"][layout]
    agent_bc_test, _ = get_bc_agent_from_saved(test_model_name)
    agent_bc_test.stochastic = False
    
    # Execute runs
    if cp_bc_name not in omit_dict:
        print(cp_bc_name)
        ap_training = AgentPair(cp, agent_bc_test)
        data0 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
        layout_cp_eval["scores"][cp_bc_name] = mean_and_std_err(data0['ep_returns'])
        layout_cp_eval["states"][cp_bc_name] = mean_and_std_err(data0['ep_states'])

    if bc_cp_name not in omit_dict:
        print(bc_cp_name)
        ap_training = AgentPair(agent_bc_test, cp)
        data1 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
        layout_cp_eval["scores"][bc_cp_name] = mean_and_std_err(data1['ep_returns'])
        layout_cp_eval["states"][bc_cp_name] = mean_and_std_err(data1['ep_states'])

    if cp_bc_name not in omit_dict and bc_cp_name not in omit_dict:
        print("CP + BC_test", mean_and_std_err(data0['ep_returns']), mean_and_std_err(data1['ep_returns']))
        print("States", mean_and_std_err(data0['ep_states']), mean_and_std_err(data1['ep_states']))

    return layout_cp_eval