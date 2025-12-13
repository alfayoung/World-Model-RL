## Real-World Digital-Twin Double-Q Learning

The DSRL repository can be naturally extended to a **real-world digital-twin, double-Q** setup, where:

- `env_real` is the main LIBERO simulator (or real robot) used for performance evaluation and the primary SAC updates.
- `env_twin` is a second LIBERO instance (digital twin) used for inexpensive lookahead and safety/feasibility checks from the same initial state as `env_real` at the beginning of each rollout.

### Core idea

1. Maintain two critics:
  - `Q_real(s, a)` trained on real (or primary sim) data `D_real` using the standard PixelSAC loss.
  - `Q_twin(s, a)` trained on twin data `D_twin` using the same SAC-style Bellman update but with `r_twin` and `P_twin` (TODO! `P_twin` can be useful when risk and safety are considered).

2. At **beginning of each real environment rollout** (see `collect_traj` in `examples/train_utils_sim.py`):

  - Observe the current real observation `o_real`.
  - Initialize / reset `env_twin` to the same **task configuration** as `env_real` (for pure simulation, this can be the same initial state and random seed; for real robots, this is an approximate digital twin of the current scene).

  - Use the pretrained diffusion policy π₀ plus the SAC noise policy (the composed policy π_total) to generate **K candidate trajectories** by running π_total in `env_twin` with **K different RNG seeds** starting from `o_twin = o_real`:

    - For each seed `k`, roll out π_total in `env_twin` in a **closed-loop** manner for the full episode, obtaining:
      - a sequence of actions and observations
      - the discounted twin return
        $$
        G_k = \sum_{t=0}^{T_k-1} \gamma^t r_{\text{twin}, t}^{(k)},
        $$
      - and logging all encountered one-step transitions into `D_twin`.

    - Optionally use the twin critic as a stabilizer by combining Monte Carlo and value-based estimates, e.g.
      $$
      \text{score}_k = (1 - \beta)\, G_k + \beta \, \min_j Q_{\text{twin}, j}\!\bigl(s_H^{(k)}, a_H^{(k)}\bigr),
      $$
      where `β` is a schedule that can increase over training as `Q_twin` becomes more accurate, and `(s_H^{(k)}, a_H^{(k)})` is the terminal (or horizon-H) state–action pair of the k-th twin rollout.  
      (Using a tail value at the end of the twin rollout is equivalent to an MPC-style finite-horizon return with a value tail.)

  - Select the candidate with the highest score `k*` and **only execute its first action / noise on the real environment**:
    - Let `(s_0^{(k*)}, a_0^{(k*)})` be the first state–action pair of the selected twin rollout.
    - In `env_real`, apply the corresponding composed policy action
      $$
      a_{\text{real}} = a_0^{(k*)}
      $$
      and step the real environment once to obtain `(s_{\text{real}}', r_{\text{real}}, \text{done})`.
    - Log this transition into `D_real`.

  - At the next real time step, repeat the whole procedure with the new real observation (i.e., **receding-horizon replanning**).

3. Learning objectives:
  - Real critic and actor (unchanged SAC):
    - `L_SAC(D_real)` is the standard PixelSAC loss already implemented in `jaxrl2/agents/pixel_sac`, trained on `D_real` collected from `env_real`.
  - Twin critic (auxiliary):
    - Define `Q_twin(s,a)` as the soft Q-function under `P_twin, r_twin`, with target:
      $$
      y_{\text{twin}} = r_{\text{twin}} + \gamma (1 - \text{done}) \, \mathbb{E}_{a' \sim \pi(\cdot|s')}
      \left[\min_j Q_{\text{twin}, j}(s', a') - \alpha \log \pi(a'|s')\right].
      $$
    - Train `Q_twin` by minimizing:
      $$
      L_{\text{critic}}^{\text{twin}} = \mathbb{E}_{(s,a,r_{\text{twin}},s',\text{done})\sim D_{\text{twin}}, j}
      \left(Q_{\text{twin}, j}(s,a) - y_{\text{twin}}\right)^2.
      $$
    - In practice, this can be implemented as a **second critic network** (same pixel encoder architecture as `Q_real`, separate parameters and target critic) trained purely on `D_twin`.

### Behavioral effect

- The twin critic and twin simulator **prune infeasible or unsafe action sequences** before they ever reach `env_real`, reducing unsafe or clearly suboptimal rollouts.
- Early in training, behavior can be driven primarily by Monte Carlo returns from `env_twin` (β ≈ 0); as `Q_twin` improves, its influence can be increased (larger β) to stabilize ranking of noisy twin rollouts and to allow shorter twin horizons by using the value function as a tail approximation.
- The real critic still defines the main objective on `D_real`, while the twin provides a structured prior and safety filter that can reduce sample complexity and improve safety during training.

### Algorithm summary

```
Algorithm: Twin-Guided Seed-Based Receding-Horizon Planning with Double Q
-------------------------------------------------------------------------------

Input:
    Pretrained diffusion policy π₀
    Pixel-SAC actor π and real critics {Q_real,j}
    Twin critics {Q_twin,j}
    Replay buffers D_real, D_twin
    Real environment env_real, twin environment env_twin
    Planning horizon H, number of seeds K
    Discount γ, mixing coefficient β

For each real episode do
    # ------------------------------------------------------------
    # 1. Initialize real env and synchronize twin
    # ------------------------------------------------------------
    s₀ ← env_real.reset()
    Synchronize env_twin ← env_real

    # ------------------------------------------------------------
    # 2. Generate K candidate trajectories in twin
    # ------------------------------------------------------------
    for k = 1 … K do
        Reset env_twin to s₀ with seed k
        s₀^(k) ← s₀
        G_k ← 0

        for h = 0 … H−1 do
            Sample action:
                a_h^(k) ~ π_total(· | s_h^(k), seed = k)
                where π_total = π₀ ⊕ π
            (s_{h+1}^(k), r_h^(k), done) ← env_twin.step(a_h^(k))

            Store (s_h^(k), a_h^(k), r_h^(k), s_{h+1}^(k), done) into D_twin
            G_k ← G_k + γ^h r_h^(k)
            if done then break
        end for

        # finite-horizon rollout + value tail
        score_k ← (1−β)·G_k + β·min_j Q_twin,j(s_H^(k), π(s_H^(k)))
    end for

    # ------------------------------------------------------------
    # 3. Select best seed
    # ------------------------------------------------------------
    k* ← argmax_k score_k

    # ------------------------------------------------------------
    # 4. Closed-loop real execution using seed k*
    # ------------------------------------------------------------
    t ← 0
    done_real ← False
    while not done_real do
        Sample action:
            a_t ~ π_total(· | s_t, seed = k*)
        (s_{t+1}, r_t, done_real) ← env_real.step(a_t)

        Store (s_t, a_t, r_t, s_{t+1}, done_real) into D_real
        s_t ← s_{t+1}
    end while

    # ------------------------------------------------------------
    # 5. Update real critics and actor (Pixel-SAC)
    # ------------------------------------------------------------
    Update {Q_real,j}, actor π using L_SAC(D_real)

    # ------------------------------------------------------------
    # 6. Update twin critics
    # ------------------------------------------------------------
    for (s, a, r, s', done) ~ D_twin do
        Sample a' ~ π(·|s')
        y_twin ← r + γ(1−done) · (min_j Q_twin,j(s', a') − α log π(a'|s'))
        Minimize (Q_twin,j(s,a) − y_twin)² for all j
    end for

end for

Output:
    Trained actor π, real critics Q_real, and twin critics Q_twin.
-------------------------------------------------------------------------------
```

## VLAC Curriculum Learning

### Core idea

To further improve training efficiency beyond the Real-World Digital-Twin Double-Q Learning framework, we propose a curriculum learning strategy driven by the value predictions of a **Vision-Language-Action-Critic (VLAC)** model. The VLAC value function is defined as:
$$
\text{VLAC} : (o_t, o_{t+1}, \text{language}) \rightarrow \delta_t,
$$
which evaluates whether the transition from $o_t$ to $o_{t+1}$ progresses toward the language-specified task objective.

At a high level, the VLAC model acts as a **failure detector**: whenever the agent begins drifting away from task completion, VLAC produces persistently negative values. This signal is exploited to automatically identify problematic states in the real environment and convert them into targeted curriculum training tasks inside a digital twin. The procedure is repeated iteratively as follows:

1. Roll out the current policy in the real environment

  Execute the current DSRL policy $\pi$ in the real environment and record the trajectory:
  $$
  \tau = \{(o_0, a_0, o_1), (o_1, a_1, o_2), \dots, (o_{T-1}, a_{T-1}, o_T)\}.
  $$

2. Evaluate VLAC values for each step 

  For every transition in the trajectory, compute the VLAC-predicted value:
  $$
  \delta_t = \text{VLAC}(o_t, o_{t+1}, \text{language}).
  $$
  Persistently negative $\delta_t$ values indicate that the agent is moving away from the desired goal.

3. Detect the earliest failure window

  Identify the **first time step** $t^*$ where VLAC outputs **$N$ consecutive negative values**:
  $$
  \delta_{t^*}, \delta_{t^*+1}, \dots, \delta_{t^*+N-1} < 0.
  $$
  The corresponding state $o_{t^*}$ marks the onset of meaningful task degradation or behavioral drift.

4. Reconstruct this failure state in a digital twin  

  Recreate the environment state corresponding to step $t^*$ inside a digital twin simulator. This reconstruction should ideally be **fully automatic**, relying on logged observations and environment metadata. The digital twin provides an inexpensive, resettable environment for focused policy improvement.

5. Domain-randomized RL finetuning from the identified failure state

  Starting from the reconstructed state, apply **domain-randomized reinforcement learning** to finetune the DSRL policy:
  - Reset the digital twin repeatedly to the failure state.
  - Randomize object pose parameters to make the policy robust.
  - Continue RL training until the policy consistently succeeds from this state.

### Behavioral effect

By iterating this pipeline, the policy progressively learns to handle **all failure modes** encountered across real-world interactions. Only a **small number of real rollouts** is required to diagnose where the agent struggles; all intensive learning happens safely and cheaply in the domain-randomized twin. Ultimately, the resulting policy becomes capable of solving the full task end-to-end with high robustness.

### Algorithm summary

```
Algorithm: VLAC-Guided Curriculum Learning with Digital Twin
--------------------------------------------------------------------------------

Input:
    DSRL policy π_θ
    VLAC value model f_VLAC(o_t, o_{t+1}, language) → δ_t
    Natural language instruction L
    Real environment env_real
    Twin environment env_twin

    Hyperparameters:
        N_neg              # number of consecutive negative steps
        epsilon            # negativity threshold, e.g. δ < -epsilon
        max_curriculum_iters
        max_real_rollouts_per_iter
        max_finetune_episodes_per_iter
        target_success_rate
        finetune_horizon H_c
        DR_distribution p(φ)  # domain randomization distribution

Initialize:
    θ ← initial policy parameters (e.g., pretrained DSRL policy)
    Initialize RL optimizer, replay buffer B for twin training

for iter = 1 ... max_curriculum_iters do
    # ------------------------------------------------------------------------
    # 1. Collect real trajectories and detect first "failure window"
    # ------------------------------------------------------------------------
    failure_found ← False
    s_star ← None

    for m = 1 ... max_real_rollouts_per_iter do
        o_0 ← env_real.reset()
        trajectory ← []  # store (o_t, a_t, r_t)

        done_real ← False
        while not done_real do
            a_t ~ π_θ(· | o_t)          # current policy, maybe with exploration
            (o_{t+1}, r_t, done_real, info) ← env_real.step(a_t)
            Append (o_t, a_t, r_t, o_{t+1}) to trajectory
            o_t ← o_{t+1}
        end while

        T ← length(trajectory)

        # -----------------------------
        # 1.1 Compute VLAC scores δ_t
        # -----------------------------
        for t = 0 ... T-2 do
            (o_t, a_t, r_t, o_{t+1}) ← trajectory[t]
            δ_t ← f_VLAC(o_t, o_{t+1}, L)
            Store δ_t
        end for

        # ------------------------------------------------------
        # 1.2 Find first negative window of length N_neg
        # ------------------------------------------------------
        for t = 0 ... T-1-N_neg do
            if δ_t < -epsilon
               and δ_{t+1} < -epsilon
               ...
               and δ_{t+N_neg-1} < -epsilon then

                # first failure window found
                t_star ← t
                failure_found ← True

                # reconstruct the underlying state s_star
                s_star ← reconstruct_state_from_real_log(trajectory, t_star)
                break
            end if
        end for

        if failure_found then
            break  # stop collecting more real rollouts in this iteration
        end if
    end for

    # ------------------------------------------------------------------------
    # 2. If no failure found in all real rollouts, stop curriculum
    # ------------------------------------------------------------------------
    if not failure_found then
        print("No more failure windows detected. Curriculum finished.")
        break
    end if

    # ------------------------------------------------------------------------
    # 3. Curriculum finetuning from s_star in twin env with domain randomization
    # ------------------------------------------------------------------------
    finetune_episode_count ← 0

    while finetune_episode_count < max_finetune_episodes_per_iter do
        finetune_episode_count ← finetune_episode_count + 1

        # 3.1 Sample domain parameters and reset twin
        φ ~ p(φ)   # sample domain randomization parameters
        env_twin.apply_domain_params(φ)
        env_twin.reset_to_state(s_star)

        # 3.2 Rollout in twin for at most H_c steps and collect data
        o_0_twin ← env_twin.get_observation()
        done_twin ← False
        step_count ← 0
        success_flag ← False

        while not done_twin and step_count < H_c do
            a_t ~ π_θ(· | o_twin)    # current policy, possibly with exploration
            (o_{t+1}_twin, r_t_twin, done_twin, info_twin) ← env_twin.step(a_t)

            # Store transition into twin replay buffer B
            Store (o_twin, a_t, r_t_twin, o_{t+1}_twin, done_twin) in B

            # Define success via info_twin["success"] or terminal reward
            if done_twin and info_twin["success"] == True then
                success_flag ← True
            end if

            o_twin ← o_{t+1}_twin
            step_count ← step_count + 1
        end while

        # 3.3 RL update in twin using data in B (e.g., SAC / DSRL update)
        for update_step in 1 ... K_updates_per_episode do
            Sample minibatch from B
            Update θ and critic parameters via RL loss
        end for

        # 3.4 Periodically evaluate success rate from s_star under current π_θ
        if finetune_episode_count % EVAL_INTERVAL == 0 then
            success_rate ← EvaluatePolicyFromState(env_twin, s_star, π_θ,
                                                   DR_distribution, H_c)
            if success_rate ≥ target_success_rate then
                print("Reached target success rate from s_star. Stop finetuning.")
                break
            end if
        end if

    end while

    # After finetuning, go back to next curriculum iteration (iter+1)
    # The updated policy π_θ will now be used in real env for failure mining.

end for

Output:
    Updated DSRL policy π_θ that can recover from previously failing states
    discovered by VLAC in the real world.
--------------------------------------------------------------------------------
```

## Comparison of Double-Q Learning and VLAC Curriculum Learning

**Similarities:**
1. Both methods utilize a digital twin of the real-world environment.
2. Both approaches aim to enhance RL training efficiency by incorporating the digital twin.

**Differences:**
1. Double-Q focuses on **real-world RL**, where the policy is updated using replay buffers populated with real-world data. In contrast, VLAC curriculum learning adopts a sim-to-real strategy, updating the policy **exclusively within the digital twin**.


## Task List

```
0. KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet.bddl
1. KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.bddl
2. KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl
3. KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.bddl
4. KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.bddl
5. KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.bddl
6. KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.bddl
7. KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.bddl
8. KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it.bddl
9. KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl
10. KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.bddl
11. KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet.bddl
12. KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate.bddl
13. KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate.bddl
14. KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate.bddl
15. KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet.bddl
16. KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.bddl
17. KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl.bddl
18. KITCHEN_SCENE3_put_the_frying_pan_on_the_stove.bddl
19. KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl
20. KITCHEN_SCENE3_turn_on_the_stove.bddl
21. KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl
22. KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet.bddl
23. KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.bddl
24. KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.bddl
25. KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet.bddl
26. KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.bddl
27. KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack.bddl
28. KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet.bddl
29. KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.bddl
30. KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.bddl
31. KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet.bddl
32. KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.bddl
33. KITCHEN_SCENE6_close_the_microwave.bddl
34. KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.bddl
35. KITCHEN_SCENE7_open_the_microwave.bddl
36. KITCHEN_SCENE7_put_the_white_bowl_on_the_plate.bddl
37. KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.bddl
38. KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove.bddl
39. KITCHEN_SCENE8_turn_off_the_stove.bddl
40. KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf.bddl
41. KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet.bddl
42. KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf.bddl
43. KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet.bddl
44. KITCHEN_SCENE9_turn_on_the_stove.bddl
45. KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it.bddl
46. LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl
47. LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.bddl
48. LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket.bddl
49. LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl
50. LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.bddl
51. LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl
52. LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket.bddl
53. LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket.bddl
54. LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.bddl
55. LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.bddl
56. LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray.bddl
57. LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray.bddl
58. LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray.bddl
59. LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.bddl
60. LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.bddl
61. LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.bddl
62. LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray.bddl
63. LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.bddl
64. LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.bddl
65. LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate.bddl
66. LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate.bddl
67. LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate.bddl
68. LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate.bddl
69. LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate.bddl
70. LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl
71. LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate.bddl
72. LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate.bddl
73. STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl
74. STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.bddl
75. STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.bddl
76. STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy.bddl
77. STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl
78. STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl
79. STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.bddl
80. STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.bddl
81. STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl
82. STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy.bddl
83. STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy.bddl
84. STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy.bddl
85. STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy.bddl
86. STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf.bddl
87. STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf.bddl
88. STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf.bddl
89. STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf.bddl
```