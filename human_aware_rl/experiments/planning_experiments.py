from overcooked_ai.overcooked_ai_py.agents.agent import CoupledPlanningAgent, CoupledPlanningPair
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import EmbeddedPlanningAgent, AgentPair

from overcooked_ai_py.utils import mean_and_std_err

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved


def get_delivery_horizon(layout):
    if layout == "simple" or layout == "random1":
        return 2
    return 3

def P_BC_evaluation_for_layout(ae, layout, best_bc_models, num_games):

    delivery_horizon = get_delivery_horizon(layout)
    print("Delivery horizon for layout {}: {}".format(layout, delivery_horizon))

    layout_p_bc_eval = {}

    #######################
    # P_BC_test + BC_test #
    #######################

    # Prepare BC_test
    test_model_name = best_bc_models["test"][layout]
    agent_bc_test, _ = get_bc_agent_from_saved(test_model_name)
    agent_bc_test.stochastic = False
    
    # Prepare P_BC_test (making another copy of BC_test just to be embedded in P_BC)
    agent_bc_test_embedded, _ = get_bc_agent_from_saved(test_model_name)
    agent_bc_test_embedded.stochastic = False
    p_bc_test = EmbeddedPlanningAgent(agent_bc_test_embedded, agent_bc_test_embedded.mlp, delivery_horizon)
    p_bc_test.env = ae.env
    p_bc_test.debug = True
    
    # Execute runs
    ap_training = AgentPair(p_bc_test, agent_bc_test)
    data0 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
    layout_p_bc_eval['P_BC_test+BC_test_0'] = mean_and_std_err(data0['ep_returns'])

    ap_training = AgentPair(agent_bc_test, p_bc_test)
    data1 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
    layout_p_bc_eval['P_BC_test+BC_test_1'] = mean_and_std_err(data1['ep_returns'])
    print("P_BC_test + BC_test", mean_and_std_err(data0['ep_returns']), mean_and_std_err(data1['ep_returns']))


    ########################
    # P_BC_train + BC_test #
    ########################

    # Prepare P_BC_train
    train_model_name = best_bc_models["train"][layout]
    agent_bc_train_embedded, _ = get_bc_agent_from_saved(train_model_name)
    agent_bc_train_embedded.stochastic = False
    p_bc_train = EmbeddedPlanningAgent(agent_bc_train_embedded, agent_bc_train_embedded.mlp, delivery_horizon)
    p_bc_train.env = ae.env
    p_bc_train.debug = True
    
    # Execute runs
    ap_testing = AgentPair(p_bc_train, agent_bc_test)
    data0 = ae.evaluate_agent_pair(ap_testing, num_games=num_games, display=True)
    layout_p_bc_eval['P_BC_train+BC_test_0'] = mean_and_std_err(data0['ep_returns'])
    
    ap_testing = AgentPair(agent_bc_test, p_bc_train)
    data1 = ae.evaluate_agent_pair(ap_testing, num_games=num_games, display=True)
    layout_p_bc_eval['P_BC_train+BC_test_1'] = mean_and_std_err(data1['ep_returns'])
    print("P_BC_train + BC_test", mean_and_std_err(data0['ep_returns']), mean_and_std_err(data1['ep_returns']))

    return layout_p_bc_eval

def P_BC_evaluation(best_bc_models, num_games=1):

    p_bc_evaluation = {}

    layouts = ['simple', 'unident_s']

    for layout in layouts:
        mdp_params = {"layout_name": layout}
        env_params = {"horizon": 400}
        ae = AgentEvaluator(mdp_params, env_params)
        p_bc_evaluation[layout] = P_BC_evaluation_for_layout(ae, layout, best_bc_models, num_games)
    
    return p_bc_evaluation


def CP_evaluation(best_bc_models, num_games=1):

    cp_evaluation = {}

    layouts = ['simple', 'unident_s']

    for layout in layouts:
        mdp_params = {"layout_name": layout}
        env_params = {"horizon": 400}
        ae = AgentEvaluator(mdp_params, env_params)
        cp_evaluation[layout] = CP_evaluation_for_layout(ae, layout, best_bc_models, num_games)
    
    return cp_evaluation

def CP_evaluation_for_layout(ae, layout, best_bc_models, num_games):

    delivery_horizon = get_delivery_horizon(layout)
    print("Delivery horizon for layout {}: {}".format(layout, delivery_horizon))

    layout_p_bc_eval = {}

    ################
    # CP + BC_test #
    ################

    # Prepare BC_test
    test_model_name = best_bc_models["test"][layout]
    agent_bc_test, _ = get_bc_agent_from_saved(test_model_name)
    agent_bc_test.stochastic = False
    
    # Prepare CP
    cp = CoupledPlanningAgent(ae.mlp, delivery_horizon=delivery_horizon)
    cp.env = ae.env
    cp.debug = True
    
    # Execute runs
    ap_training = AgentPair(cp, agent_bc_test)
    data0 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
    layout_p_bc_eval['CP+BC_test_0'] = mean_and_std_err(data0['ep_returns'])

    ap_training = AgentPair(agent_bc_test, cp)
    data1 = ae.evaluate_agent_pair(ap_training, num_games=num_games, display=True)
    layout_p_bc_eval['CP+BC_test_1'] = mean_and_std_err(data1['ep_returns'])
    print("CP + BC_test", mean_and_std_err(data0['ep_returns']), mean_and_std_err(data1['ep_returns']))


    ###########
    # CP + CP #
    ###########

    # Prepare CP pair
    cp_pair = CoupledPlanningPair(cp)
    
    # Execute runs
    data = ae.evaluate_agent_pair(cp_pair, num_games=num_games, display=True)
    layout_p_bc_eval['CP+CP'] = data['ep_returns'][0]
    print("CP + CP", data['ep_returns'][0])

    return layout_p_bc_eval