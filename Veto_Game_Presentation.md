# Veto Game: Human-AI Collaboration via Veto Mechanisms

## Slide 1: Title Slide

**Title:** Enhancing Human-AI Collaboration in Tactical Environments: A Veto Mechanism Approach

**Presenter:** [Your Name/Affiliation]

**Date:** [Date]

**(Presenter Script):**
"(Lynn): Good morning/afternoon, everyone. Thank you for coming. Today, I'll be discussing our research on improving how humans and artificial intelligence can work together effectively, particularly in complex and dynamic tactical settings. We'll explore a specific approach using what we call the 'Veto Game,' a platform designed to investigate how mechanisms allowing human intervention – specifically, the ability to veto an AI's decision – can enhance collaboration."

---

## Slide 2: Abstract & Introduction (1/2)

**Title:** Introduction: The Challenge of Human-AI Teaming

**Content:**
*   Increasing use of AI agents in complex, high-stakes domains (e.g., defense, autonomous driving, disaster response) [2].
*   Challenge: How to ensure effective collaboration and appropriate human oversight, especially under time pressure? [8, 12].
*   Need for interfaces that allow humans to leverage AI strengths while mitigating potential AI failures or unexpected behaviors.
*   Focus: Human intervention mechanisms – specifically, the ability to *veto* an AI's proposed action.

**(Presenter Script):**
"(Lynn): Artificial intelligence is increasingly being deployed in critical, high-stakes domains like defense, autonomous systems, and emergency response. While powerful, integrating AI into these areas presents a significant challenge: how do we ensure humans and AI can collaborate effectively? Especially when decisions need to be made quickly under pressure, how can we maintain appropriate human oversight? Often, AI operates as a bit of a black box, making trust and management difficult. Simple automation isn't always the best approach. This highlights the need for carefully designed interfaces that allow humans to harness the AI's capabilities while safeguarding against potential errors or unexpected actions. Our research focuses on one such human intervention mechanism: the ability for a human operator to *veto* an AI's proposed course of action."

---

## Slide 3: Abstract & Introduction (2/2)

**Title:** The "Veto Game" Research Platform

**Content:**
*   **Goal:** Develop and evaluate methods for effective human-AI collaboration using veto mechanisms in a simulated tactical environment.
*   **Platform:** A grid-based tactical exploration game where an AI agent navigates, gathers resources, and engages targets.
*   **Intervention:** A human participant can observe the AI and choose to veto its intended actions based on presented information (e.g., AI uncertainty, risk assessment).
*   **Key Research Questions:**
    1.  How do different veto mechanism designs (e.g., threshold-based vs. uncertainty-aware) affect task performance and human interaction patterns?
    2.  How can AI uncertainty be represented to best support human veto decisions?
    3.  Under what conditions does human veto intervention improve overall team performance compared to the AI acting alone?

**(Presenter Script):**
"(Eric): To investigate this, we developed the 'Veto Game' research platform. Our primary goal is to develop and evaluate methods for effective human-AI collaboration, specifically focusing on veto interactions within a simulated tactical environment. The platform itself is a grid-based game where an AI agent autonomously explores, manages resources, and deals with enemy targets. The key element is the human intervention: a participant observes the AI's actions and intentions, and based on information provided by the system – such as the AI's estimated uncertainty or a risk assessment – they can choose to veto the AI's planned action. This platform allows us to address several key research questions: Firstly, how do different ways of designing the veto trigger, like simple thresholds versus using the AI's uncertainty, impact performance and interaction? Secondly, what's the best way to show AI uncertainty to a human to help them make good veto decisions? And finally, when does allowing a human to veto actually help the team perform better compared to just letting the AI run on its own?"

---

## Slide 4: Problem Formulation (1/2)

**Title:** Problem: Optimizing AI Actions with Human Oversight

**Content:**
*   **Environment:** Modeled as a Markov Decision Process (MDP).
    *   States \( s \in \mathcal{S} \): Grid configuration, agent status (health, ammo, shields), enemy locations, explored areas.
    *   Actions \( a \in \mathcal{A} \): Movement (4), Shooting (4), Place Trap, Use Cover, Call Support.
    *   Transitions \( P(s' | s, a) \): Probability of reaching state \( s' \) from state \( s \) after action \( a \) (Includes stochastic elements like enemy movement, combat outcomes).
    *   Rewards \( R(s, a, s') \): Feedback signal (e.g., + for eliminating enemy, - for taking damage, small negative cost for time/movement).
    *   Goal: Maximize expected discounted cumulative reward: \( \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R_{t+1}] \).
*   **Challenge:** AI agent learns a policy \( \pi(a|s) \) to maximize reward, but this policy might be suboptimal, brittle, or take unacceptable risks in certain situations.

**(Presenter Script):**
"(Eric): From a technical standpoint, we model the game environment as a Markov Decision Process, or MDP. The state space, denoted by script S, includes all relevant information like the map layout, the agent's health and resources, enemy positions, and which parts of the map have been explored. The action space, script A, consists of the possible moves the agent can make – moving, shooting, placing traps, and so on. The transitions define the probability of moving to a new state given the current state and action, accounting for uncertainties like enemy behavior. The reward function provides feedback – positive for good outcomes like neutralizing a target, negative for bad ones like taking damage. The AI agent's objective, typical in reinforcement learning, is to learn a policy, pi, that maximizes the expected long-term discounted reward. However, the core challenge we address is that even a well-trained AI policy might not always be optimal or might take actions deemed too risky from a human perspective, especially when facing novel situations."

---

## Slide 5: Problem Formulation (2/2)

**Title:** The Role of Veto Intervention

**Content:**
*   **Hypothesis:** Allowing a human to veto specific AI actions can improve safety and overall performance by combining AI's computational speed with human judgment and context awareness [8, 12].
*   **Mechanism:**
    1.  AI proposes action \( a_t = \pi(s_t) \).
    2.  System assesses action \( a_t \) based on current state \( s_t \), potentially using AI's internal metrics (Q-values, uncertainty).
    3.  If assessment flags potential issue (high risk/uncertainty), prompt human: "AI proposes \( a_t \). Veto?" (Information provided: reason, uncertainty level, alternatives) [13].
    4.  Human decides: Veto / Accept.
    5.  If Vetoed, AI selects alternative action \( a'_t \).
    6.  Execute final action.
*   **Need:** Evaluate different assessment/prompting strategies (Conditions: Baseline, Threshold Veto, Uncertainty Veto).

**(Presenter Script):**
"(Lynn): Our central hypothesis is that enabling a human to veto specific AI actions can lead to better and safer outcomes by effectively combining the AI's speed and analytical power with human intuition and broader contextual understanding. The proposed veto mechanism works as follows: First, the AI determines its intended action based on its learned policy. Second, the system analyzes this proposed action in the current context, potentially using the AI's own internal calculations like its confidence or uncertainty. Third, if this assessment indicates a potential problem – perhaps the action is deemed too risky or the AI is highly uncertain – the system prompts the human operator, presenting the proposed action and relevant information like the reason for the flag and potential alternatives. Fourth, the human makes a decision: accept the action or veto it. Fifth, if vetoed, the AI needs to select an alternative action. Finally, the resulting action is executed in the environment. A key part of our research is evaluating different strategies for that second step – how the system assesses the action – which leads to our experimental conditions: a baseline with no veto, a veto triggered by risk thresholds, and a veto triggered by AI uncertainty."

---

## Slide 6: Approach (1/2)

**Title:** Methodology: AI Agent & Veto Mechanisms

**Content:**
*   **AI Agent:**
    *   Reinforcement Learning: Dueling Deep Q-Network (Dueling DQN) [5].
    *   Learns action-value function \( Q(s, a) \).
    *   Architecture separates state value \( V(s) \) and action advantages \( A(s, a) \):
        \[ Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a' \in \mathcal{A}} A(s,a') \right) \]
    *   Includes standard DQN features: Experience Replay, Target Network. (Note: Could also incorporate attention mechanisms similar to [6])
*   **Uncertainty Estimation (for Uncertainty Veto):**
    *   Method: Monte Carlo (MC) Dropout.
    *   Network includes Dropout layers, kept active during inference.
    *   Multiple stochastic forward passes (\(N\)) are run for a given state \( s \).
    *   Uncertainty \( U(s,a) \) is estimated as the standard deviation of the \( Q(s,a) \) predictions across the \( N \) passes:
        \[ U(s,a) \approx \text{Std}(\hat{Q}_1(s,a), ..., \hat{Q}_N(s,a)) \]

**(Presenter Script):**
"(Lynn): Now let's look at the specific methods used. The AI agent is trained using reinforcement learning, specifically employing a Dueling Deep Q-Network, or Dueling DQN. This type of network learns the expected value, Q(s,a), of taking each action 'a' in a given state 's'. The dueling architecture, shown in the formula here, is known to be effective as it separates the estimation of the overall value of a state, V(s), from the advantage, A(s,a), of each specific action in that state. It also uses standard techniques like experience replay and a target network for stable learning. For the uncertainty-based veto condition, we need a way to estimate the AI's uncertainty. We use Monte Carlo Dropout. This involves adding dropout layers to the network, which are usually turned off during testing. However, for MC Dropout, we keep them active during inference and run the input state through the network multiple times – say, N times. Because dropout randomly deactivates neurons, we get slightly different Q-value predictions each time. The uncertainty for an action is then estimated as the standard deviation of these Q-value predictions across the N forward passes, giving us a measure of the model's confidence, specifically its epistemic uncertainty."

---

## Slide 7: Approach (2/2)

**Title:** Methodology: Experimental Conditions

**Content:**
*   **Objective:** Compare performance and interaction patterns across different intervention strategies.
*   **Conditions:**
    1.  **Baseline:** RL Agent acts autonomously. No veto mechanism involved. Represents standard AI performance.
        \[ a_t = \arg\max_a Q(s_t, a) \]
    2.  **Threshold Veto:** Veto prompt triggered if a rule-based `RiskAssessor` flags the AI's chosen action \( a_t \) as high-risk (based on heuristics like low health, resource constraints, etc.). \( \text{VetoTrigger} = \text{RiskAssessor}(s_t, a_t) \) (Similar concept to risk-aware systems [13])
    3.  **Uncertainty Veto:** Veto prompt triggered primarily if the AI's uncertainty \( U(s_t, a_t) \) for the chosen action exceeds a dynamic threshold \( \theta_U \). Also considers risk assessment as secondary trigger. \( \text{VetoTrigger} = (U(s_t, a_t) > \theta_U) \lor \text{RiskAssessor}(s_t, a_t) \)
*   **Simulation:** Headless runs using predefined scenarios ('Tutorial' through 'Extreme') with increasing complexity. Veto decisions in headless mode are simulated randomly (30% rejection rate if triggered) for 'Threshold' and 'Uncertainty' conditions.

**(Presenter Script):**
"(Eric): Our experimental objective is to compare these different intervention strategies. We defined three main conditions. First, the Baseline, where the RL agent simply acts according to its learned policy by choosing the action with the highest Q-value, with no human intervention or veto possible. This serves as our control. Second, the Threshold Veto condition, where a potential veto is triggered if a separate module, the RiskAssessor, identifies the AI's chosen action as high-risk based on predefined rules or heuristics. Third, the Uncertainty Veto condition, where a veto is triggered mainly if the AI's estimated uncertainty for its chosen action, calculated using MC Dropout, goes above a certain threshold, theta-U. This condition also incorporates the RiskAssessor as a secondary check. For the results presented today, we used headless simulations, running the agent through a set of predefined scenarios of increasing difficulty. In these simulations, for the Threshold and Uncertainty conditions, the human veto decision itself was simulated – if the mechanism triggered a prompt, there was a fixed 30% chance the action would be rejected."

---

## Slide 8: Key Results (1/4)

**Title:** Performance Comparison: Average Reward

**Content:**
*   Metric: Average reward obtained per action across all scenarios (higher is better).
*   Comparison of Conditions (from simulation run `experiment_20250424_135256`):

| Condition    | Mean Reward | Std Dev |
|--------------|-------------|---------|
| Baseline     |   +0.0404   |  ~0.37  |
| Threshold    |   +0.0438   |  ~0.50  |
| Uncertainty  |   -0.0167   |  ~0.48  |

*   *(Insert `reward_by_condition.png` plot here)*

**(Presenter Script):**
"(Eric): Moving on to the key results from our simulation run, specifically experiment `20250424_135256`. Our primary performance metric is the average reward the agent received per action, aggregated across all the scenarios it played. Higher average reward indicates better overall task performance. This table summarizes the findings. We can see that the Baseline condition, with no veto, achieved a mean reward of about +0.04. The Threshold veto condition performed very similarly, slightly higher at +0.044. Interestingly, the Uncertainty veto condition resulted in a slightly negative average reward of -0.017. Here you can see the plot visualizing these means [Point to plot]. Note the error bars, which represent the standard deviation. They are quite large in all conditions, especially Threshold and Uncertainty, indicating considerable variability in the rewards received from step to step during this run."

---

## Slide 9: Key Results (2/4)

**Title:** Veto Behavior

**Content:**
*   Analysis of veto interventions (Threshold & Uncertainty conditions combined, run `experiment_20250424_135256`):
    *   Total Veto Assessments: 122
    *   Actions Accepted (Not Vetoed): 80 (~65.6%)
    *   Actions Vetoed (Rejected): 42 (~34.4%)
*   *(Insert `veto_distribution.png` plot here)*

**(Presenter Script):**
"(Lynn): We also analyzed how often the veto mechanisms actually intervened. This pie chart [Point to plot] shows the combined results for the Threshold and Uncertainty conditions from that same experiment run. The system assessed potential vetoes 122 times in total across these two conditions. Of those assessments, about 66% resulted in the action being accepted, while the remaining 34% resulted in the action being vetoed or rejected by the simulated human response. It's important to remember that this rejection rate reflects both how often the mechanism's internal logic triggered an assessment *and* the 30% simulated random rejection chance applied during the headless run when a trigger occurred."

---

## Slide 10: Key Results (3/4)

**Title:** Interpretation of Performance Results

**Content:**
*   **Baseline vs. Threshold:** Threshold veto performed slightly *better* than baseline (+0.0438 vs +0.0404). Difference is small relative to variance. Suggests rule-based veto might offer marginal benefit or be neutral in this setup.
*   **Baseline vs. Uncertainty:** Uncertainty veto performed slightly *worse* than baseline (-0.0167 vs +0.0404). Suggests that intervening based on MC Dropout uncertainty (with the current thresholding and simulated responses) might have slightly hindered performance in this run.
*   **Threshold vs. Uncertainty:** Threshold outperformed Uncertainty in this specific simulation run.
*   **High Variance:** Large standard deviations across all conditions highlight significant variability in outcomes depending on the specific situations encountered in the scenarios.

**(Presenter Script):**
"(Lynn): So, what can we interpret from these performance numbers, keeping in mind this is a single simulation? Comparing the Threshold veto to the Baseline, the performance was very similar, with Threshold being marginally better. Given the high variance, this suggests that this simple rule-based veto might be neutral or perhaps offer a very small benefit in this setup. However, comparing the Uncertainty veto to the Baseline, we saw slightly worse performance in the Uncertainty condition. This tentatively suggests that using MC Dropout uncertainty with the current configuration and simulated vetoes might have actually hindered performance slightly in this run. Consequently, the Threshold condition outperformed the Uncertainty condition. Again, the high variance across all conditions is critical – it tells us that performance was inconsistent and these small average differences could easily be due to the specific situations encountered in this particular run. We need more data for firm conclusions."

---

## Slide 11: Key Results (4/4)

**Title:** Limitations & Observations

**Content:**
*   **Single Simulation Run:** Results are based on one headless run with simulated veto responses. Conclusions are preliminary.
*   **State Size Mismatch:** Warnings ("Input size mismatch") indicate the agent's model (fixed size) processed padded/truncated state vectors from scenarios with varying grid sizes. This likely impacted agent performance and learning consistency across scenarios.
*   **Missing Data:** Current analysis lacks:
    *   Reward data linked to specific veto outcomes (reward difference between vetoed vs. accepted actions).
    *   Veto decision times (not applicable in headless simulation).
*   **Uncertainty Calibration:** MC Dropout uncertainty might require better calibration or different thresholding strategies to be effective for veto decisions. The observed negative impact could be due to suboptimal parameters or thresholds.

**(Presenter Script):**
"(Eric): It's crucial to acknowledge the limitations of these results. Most importantly, they stem from a single simulation run with simulated human responses, making any conclusions preliminary. We also observed numerous 'Input size mismatch' warnings during the run. This happened because our different scenarios used varying map sizes, but the AI agent's neural network expected a fixed input size. The temporary fix was to pad or truncate the input, but this inconsistency likely affected the agent's performance and ability to generalize across scenarios. Furthermore, our current data logging and analysis doesn't allow us to see if actions that *were* vetoed ultimately led to better or worse rewards than those that were accepted. We also lack decision time data, though that's expected in a headless simulation. Finally, the potential negative impact of the uncertainty veto highlights that simply using an uncertainty measure isn't enough; methods like MC Dropout might need careful calibration and tuning of the decision thresholds to be truly beneficial."

---

## Slide 12: Implication of Results (1/2)

**Title:** Implications for Human-AI Collaboration

**Content:**
*   **Veto is Nuanced:** Simply adding a veto mechanism doesn't guarantee improvement. The *design* of the trigger and the *information* presented are crucial.
*   **Rule-Based Veto:** Simple, interpretable rules (Threshold condition) showed potential neutral-to-positive impact in this run, but may lack flexibility.
*   **Uncertainty-Based Veto:**
    *   Potentially powerful for identifying situations where the AI is likely to err.
    *   Requires careful implementation:
        *   Reliable uncertainty estimation (MC Dropout is one method, others exist).
        *   Effective thresholding/calibration (adaptive thresholds might be key).
        *   Clear presentation to the human user.
    *   Poorly calibrated uncertainty might lead to unnecessary or harmful interventions (as potentially seen in this run).

**(Presenter Script):**
"(Eric): Stepping back from the specific numbers, what does this tell us about designing human-AI teams? Firstly, it reinforces that adding a veto capability is a nuanced process. It's not a magic bullet; the effectiveness heavily depends on *how* the veto is triggered and *what* information supports the human's decision. Our preliminary results suggest a simple, rule-based veto might perform reasonably well and has the benefit of being interpretable, though it might not adapt well to diverse situations. Uncertainty-based vetoes hold significant promise, as they could theoretically flag exactly when the AI is unsure or likely to make a mistake. However, our results also hint at the implementation challenges. You need reliable uncertainty estimates, effective ways to set and adapt the veto thresholds, and clear ways to present this complex information to the user. If the uncertainty isn't well-calibrated or the thresholds are wrong, you risk interrupting the AI too often or failing to intervene when necessary, potentially harming performance, as we might have seen here."

---

## Slide 13: Implication of Results (2/2)

**Title:** Design Considerations for Veto Systems

**Content:**
*   **Information Presentation:** How should risk/uncertainty be communicated to the human to enable informed, timely decisions? (e.g., numeric values, visualizations, natural language explanations).
*   **Threshold Tuning:** How to set appropriate thresholds for triggering vetos (fixed, adaptive, context-dependent)? Balancing missed interventions vs. excessive interruptions.
*   **Alternative Actions:** Should the system suggest alternatives if an action is vetoed? How are these generated?
*   **Cognitive Load:** How to minimize the burden on the human operator, especially in time-critical scenarios?

**(Presenter Script):**
"(Lynn): This leads to several key design considerations for anyone building these kinds of veto systems. First, information presentation: what is the best way to show risk or uncertainty to the human operator so they can make a quick and informed decision? Should it be raw numbers, charts, or plain language? Second, threshold tuning: finding the right level to trigger a veto prompt is critical. Should it be fixed, should it adapt based on context or history? There's a fundamental trade-off between intervening too much and interrupting workflow, versus intervening too little and missing critical errors. Third, alternative actions: if the AI's first choice is vetoed, should the system proactively suggest alternatives? And how should those alternatives be chosen? Finally, cognitive load: especially in high-pressure situations, the interface must be designed to minimize the mental effort required from the human operator. These are all critical questions our research platform aims to help address."

---

## Slide 14: Conclusion and Future Work (1/2)

**Title:** Conclusion

**Content:**
*   Developed "Veto Game," a research platform for studying human-AI collaboration via veto mechanisms in a tactical grid-world environment.
*   Implemented and compared an RL agent (Dueling DQN) operating under three conditions: Baseline (no veto), Threshold-based Veto, and Uncertainty-based Veto (MC Dropout).
*   Preliminary simulation results suggest:
    *   Threshold-based veto showed performance comparable to the baseline.
    *   Uncertainty-based veto (in its current configuration) slightly underperformed the baseline.
    *   High performance variability observed across all conditions.
*   Highlights the importance of careful veto mechanism design and calibration.

**(Presenter Script):**
"(Lynn): In conclusion, we've developed the 'Veto Game,' a simulation platform designed specifically for exploring human-AI collaboration through veto interactions in a tactical setting. We implemented a Dueling DQN agent and evaluated its performance autonomously (our Baseline) and when augmented with two different veto mechanisms: one based on risk thresholds and another based on the AI's uncertainty estimated via MC Dropout. Our initial simulation run, while limited, suggested that the simple threshold-based veto performed comparably to the baseline, while the uncertainty-based veto, as currently configured, slightly underperformed. We observed high performance variability across all conditions. The main takeaway is that the design and calibration of these veto mechanisms are critical and non-trivial factors in determining their effectiveness."

---

## Slide 15: Conclusion and Future Work (2/2)

**Title:** Future Work

**Content:**
*   **More Extensive Experiments:**
    *   Run with actual human participants to evaluate usability, trust, and real decision-making.
    *   Increase number of simulation runs for statistical validity.
    *   Conduct parameter sweeps for veto thresholds.
*   **Technical Improvements:**
    *   Address state-size mismatch issue for consistent agent evaluation across scenarios.
    *   Improve data logging to capture reward outcomes linked to veto decisions and response times (in human studies).
    *   Explore alternative/improved uncertainty estimation methods (e.g., ensembles, evidential deep learning).
    *   Investigate methods for automatically learning optimal veto policies or thresholds.
*   **Advanced Veto Mechanisms:** Explore richer forms of interaction beyond simple veto (e.g., suggestions, explanations, dialogue).

**(Presenter Script):**
"(Eric): Based on this work and its limitations, we've identified several key directions for future work. Experimentally, the most important next step is to conduct studies with actual human participants to understand usability, trust, and how people truly interact with these systems. We also need more simulation runs to ensure our results are statistically robust, and we should perform parameter sweeps, particularly for the veto thresholds. Technically, we need to fix the state-size mismatch problem to allow for more consistent agent evaluation. Our data logging needs improvement to capture finer-grained details like rewards following specific veto decisions, and response times during human studies. We should also explore alternative ways to estimate AI uncertainty beyond MC Dropout, such as using model ensembles. Furthermore, investigating methods to automatically learn the best veto thresholds or policies would be valuable. Finally, looking beyond simple veto, we plan to explore more advanced interaction mechanisms, perhaps involving AI explanations or dialogue."

---

## Slide 16: Publishable (Worthy of Publication: 1 slide)

**Title:** Novelty and Potential Contributions

**Content:**
*   **Current Status:** While the research direction is promising, the current results are preliminary (based on a single simulation with limitations) and **not yet sufficient for publication** in a peer-reviewed venue.
*   **Path to Publication - Novelty:** The work's potential for publication lies in its novel aspects, contingent on further validation:
    *   Application of Dueling DQN with specific veto mechanisms (threshold vs. uncertainty-based) in a procedurally generated tactical grid-world designed for HRI research.
    *   Direct comparison of baseline AI vs. AI augmented with different human intervention paradigms.
    *   Framework enabling study of uncertainty communication and its impact on human oversight.
*   **Path to Publication - Potential Contributions:** Upon successful completion of future work (addressing limitations, human studies), the research could offer significant contributions:
    *   Insights into the effectiveness (and potential pitfalls) of different veto strategies in human-AI teams.
    *   Methodological contribution via the Veto Game platform for future HRI studies.
    *   Data-driven guidance for designing interfaces that balance AI autonomy with human control based on risk and uncertainty.
    *   Highlights challenges in calibrating and utilizing AI uncertainty for collaborative tasks.

**(Presenter Script):**
"(Eric): So, the crucial question: is this work publishable right now? Honestly, given that our results are from a few simulation runs and have limitations like the state-size mismatch and simulated vetoes, the work in its current form isn't ready for a peer-reviewed publication. However, we believe the *direction* and the framework itself hold significant promise. The potential for publication hinges on addressing the limitations we've discussed and conducting more robust experiments, especially with human participants. 

Why do we think it *can* become publishable? The novelty lies in applying these specific RL and veto techniques in our custom HRI environment, directly comparing intervention methods, and providing a platform to study uncertainty communication. 

If we successfully validate these preliminary findings through future work, the potential contributions are substantial: providing real insights into effective veto strategies, offering the Veto Game as a tool for the research community, guiding interface design, and clarifying the challenges of using AI uncertainty in practice. So, while not publishable today, we have a clear path towards generating publishable findings by building on this foundation. Thank you." 