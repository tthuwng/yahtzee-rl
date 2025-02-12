import gradio as gr
import torch
import numpy as np

from env import YahtzeeEnv, IDX_TO_ACTION
from encoder import StateEncoder
from dqn import YahtzeeAgent

# We'll load a saved checkpoint of the agent
# Provide a path or load from default
AGENT_PATH = "models/yahtzee_run_best.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = YahtzeeAgent(
    state_size=22,  # default if not using opponent_value
    action_size=46,
    device=device,
    num_envs=1
)
try:
    agent.load(AGENT_PATH)
    print(f"Loaded agent from {AGENT_PATH}")
except:
    print("Warning: could not load agent checkpoint.")

# Prepare environment and encoder for demonstration
demo_env = YahtzeeEnv()
demo_encoder = StateEncoder(use_opponent_value=False)

def simulate_game_interface():
    """
    Runs a full game with the agent, returns text logs of each turn's dice and action.
    """
    text_log = []
    state = demo_env.reset()
    done = False
    agent.eval()
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    turn = 1
    while not done:
        text_log.append(f"=== Turn {turn} ===\n{demo_env.render()}")
        vec = demo_encoder.encode(state)
        valid_actions = demo_env.get_valid_actions()
        if not valid_actions:
            break
        action_idx = agent.select_action(vec, valid_actions)
        state, reward, done, info = demo_env.step(action_idx)
        act_type = info["action_type"]
        if act_type == "SCORE":
            text_log.append(f"Scored: {reward} points")
            turn += 1
    text_log.append("=== Game Over ===")
    text_log.append(demo_env.render())
    agent.epsilon = old_eps
    agent.train()
    return "\n".join(text_log)

def calculate_q_values(dice_str, rolls_left, score_dict_str):
    """
    Let user specify dice, rolls_left, and partial score sheet to see Q-values for each action.
    dice_str: e.g. "1,2,3,4,0" (0 for unrolled dice)
    score_dict_str: e.g. "ONES=3, TWOS=None, THREES=None, ..."

    We'll parse them, build a GameState, encode it, and show sorted Q-values for valid actions.
    """
    # parse dice
    try:
        dice_vals = [int(x.strip()) for x in dice_str.split(",")]
        if len(dice_vals) != 5:
            return "Please provide exactly 5 dice values."
    except:
        return "Failed parsing dice. Make sure it's comma-separated 5 integers."
    
    # parse score dict
    score_sheet = {}
    try:
        # example: "ONES=3, TWOS=None, THREES=None, FOURS=None, FIVES=None, SIXES=None,
        #           THREE_OF_A_KIND=None, FOUR_OF_A_KIND=None, FULL_HOUSE=None,
        #           SMALL_STRAIGHT=None, LARGE_STRAIGHT=None, YAHTZEE=None, CHANCE=None"
        from env import YahtzeeCategory
        mapping = {cat.name:cat for cat in YahtzeeCategory}
        entries = [x.strip() for x in score_dict_str.split(",")]
        for e in entries:
            if "=" in e:
                cat_str, val = e.split("=")
                cat_str = cat_str.strip()
                val = val.strip()
                cat = mapping.get(cat_str, None)
                if cat is None:
                    score_sheet_str = ", ".join([c.name for c in YahtzeeCategory])
                    return f"Invalid category: {cat_str}. Must be one of: {score_sheet_str}"
                if val.lower() == "none":
                    score_sheet[cat] = None
                else:
                    score_sheet[cat] = int(val)
    except Exception as e:
        return f"Error parsing score dict: {e}"

    from env import GameState
    state = GameState(
        current_dice=np.array(dice_vals, dtype=int),
        rolls_left=int(rolls_left),
        score_sheet=score_sheet
    )
    vec = demo_encoder.encode(state)
    valid_actions = demo_env.get_valid_actions()
    # But we need the environment to have the same state to get the correct valid_actions
    # We'll do a quick hack: we'll store the state in demo_env
    old_state = demo_env.state
    demo_env.state = state
    valid_indices = demo_env.get_valid_actions()
    # revert environment
    demo_env.state = old_state

    q_values = agent.get_q_values(vec)
    # mask invalid
    mask = np.full(agent.action_size, -1e9)
    mask[valid_indices] = 0
    q_values += mask
    # Sort
    ranked = sorted([(i, float(q_values[i])) for i in valid_indices], key=lambda x: x[1], reverse=True)
    lines = []
    for rank, (idx, val) in enumerate(ranked):
        action = IDX_TO_ACTION[idx]
        lines.append(f"{rank+1}. {action.kind.name} {action.data if action.data else ''} => Q={val:.2f}")
    return "\n".join(lines)

with gr.Blocks() as demo:
    gr.Markdown("# Yahtzee RL Demo\nWelcome to Yahtzee agent UI. We have two modes below.")
    
    with gr.Tab("Simulation Mode"):
        sim_btn = gr.Button("Run a full game with the RL Agent")
        sim_output = gr.Textbox(lines=25, label="Game Progress & Final Score")
        sim_btn.click(fn=simulate_game_interface, outputs=sim_output)
    
    with gr.Tab("Calculation Mode"):
        gr.Markdown("Provide partial game state to see Q-values.")
        dice_in = gr.Textbox(label="Dice (comma-separated, e.g. '1,2,3,4,0')")
        rolls_in = gr.Number(value=3, label="Rolls Left")
        score_in = gr.Textbox(label="Score Sheet (e.g. 'ONES=None, TWOS=None, ...')")
        calc_btn = gr.Button("Get Q-Values")
        calc_out = gr.Textbox(lines=15, label="Valid Actions & Q-Values")
        calc_btn.click(fn=calculate_q_values, inputs=[dice_in, rolls_in, score_in], outputs=calc_out)

demo.launch()