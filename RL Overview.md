# RL Overview
Definition : A **framework for learning** how to interact with the environment **from experience.**

![Figure 1 : Scheme of reinforcement learning model](figs/Figure%201.png)

Figure 1 : Scheme of reinforcement learning model

# Introduction

Reinforcement learning is a **semi-supervised** learning due to its **timed-delayed** labels.

Labels in RL are modeled as **rewards**, distributed **sporadically**.

The **Agent** is the entity who is learning, through the iterations of **states** $s_t$, **actions** $a_t$ and **rewards** $r_t$

Then, how is the **knowledge** modeled? 

**Policy:**  $\pi(s,a)=P(a=a |s=s)$; given a state $s$, what is the best action $a$ to do?

Learning a **robust** and **complex** policy means: given any state $s$ the agent is capable to take one of  or the **best** action to do.

How can we evaluate our policy?

Evaluating policy is at the core of learning.

**Value:**  $V_\pi(s)= \Bbb E(\sum_t\gamma^tr_t|s_0=s)$

Given that policy $\pi$ at the state $s$, what is the expected rewards I will get in the future if I start at that state and I enact that policy.

$\gamma$: discount rate, constant between $0$ and $1$.  it’s related to economic theory and psychology that explain why people are more eager to have a reward instantly than much later. 

**Goal:** Optimize policy to **maximize** future Rewards. At the end it is an optimization problem to solve for $\pi$

## Model environment as a Markov Decision Process (MDP):

> A **sequential** decision problem for a fully/partially observable, **stochastic environment** with a **Markovian** transition model and **additive** rewards is called a Markov decision process, or MDP, and consists of a set of **States** (with an initial state $s₀$); a set **Actions**(s) of actions in each state; a transition model P(s′| s, a); and a **Reward** function R(s).
> 

### Keywords:

***Stochastic*** means our world is not ***deterministic***. From a certain state, if we choose the same action, we are not **guaranteed** to move into the **same** next state.

***Additive rewards*** mean the rewards from different moves are cumulative.

***Sequential*** means our current situation is affected by previous decisions. By the same token, all future situations are affected by our current decision.

***Markovian*** means our world is ***memoryless***. This may seem in contrary to our definition of *sequential*, but they actually have completely different meanings.

***Memoryless*** means no matter how we reached the state $s$, once we are in that state, we always face the same situation with the **same set** of possible actions. If we make a decision, what will happen is **not dependent** on what happened in the **past**.

## Credit Assignment Problem:

- Dense Vs. Sparse rewards
- Sample efficiency

The CAP focuses on figuring out which past actions from the agent actually contributed to the achieved reward, especially when there's a delay between actions and rewards.

### Dense & Sparse rewards

Rewards are awarded sporadically, meaning that there is a certain **frequency**, ***Dense*** refers to high frequency and ***Sparse*** to low frequency.  

A model being **guided** by a human after each $k$  states will be given **extra** dense reward which can be helpful for training. It is called reward shaping to help the model knowing which actions contributed to getting a reward or not.

### Sample efficiency

A reinforcement model can be **effective** while having a **small** dataset to train on or after **not much iterations**, this can be called a **sample efficient** model. Other models will need **more samples** and potentially **more time** to train to reach the results we are aiming for thus the model is labeled **sample inefficient**.

## Optimize the policy:

### Algorithms

- Differential programming
- Monte Carlo
- Temporal difference (Model free)
- Bellman 1957

### **Concepts**

- Exploration Vs. Exploitation
- Policy iteration : GD, SGD, Evolutionary optimization, Simulated annealing

## Q-Learning:

$Q(s,a)=$  Quality of **State/Action** Pair

Instead of learning the policy and value function **separately**, we can learn them both **simultaneously**. 

So, this value express the **quality** of an **action** $a$ in the context of the **state** $s$ assuming that we take **best** actions in the **future**.

**Update function:**

$Q^{update}(s_t,a_t)=Q^{old}(s_t,a_t)+\alpha(r_t+\gamma  \max_a Q(s_{t+1},a)-Q^{old}(s_{t+1},a))$

$\alpha$  is the learning rate

$\gamma$ is the discount rate

$max$ means here that we assume best action will be taken in the future

This quality function can now be learned with **neural networks.**

## Hindsight replay

In CAP, Instead of **throwing out** all the data that doesn’t get you a **reward**, this method explain that maybe you will need this data in the **future** so you got to store it. In other words, If some actions didn’t get you a reward doesn’t mean it won’t be relevant to use it to make the model **more accurate**.

This helps knowing how to get to **different states** (not only the targeted one) and use the **sequence of actions** that led to them to refine the model.

The method used is “**artificial rewards**”.

Hindsight replay was a big **improvement** for making more **sample efficient** models.

# Overview of Methods

![Figure 2: Summary of methods in Reinforcement Learning](figs/Figure%202.png)

Figure 2: Summary of methods in Reinforcement Learning

# Model-based RL

In model based reinforcement learning, we assume that we have a model for how the system work, the values & rewards for each state

## Markov Decision Process (Dynamic programming & Bellman optimality):

In MDP, we assume that the environment evolve according to a Markov decision process, it is modeled with this Probability:

$P(s',s,a)=Pr(s_{k+1}=s'|s_k=s,a_k=a)$:

what is the **probability** of going to state $s_{k+1}$ according to state $s$ and action $a$

$R(s',s,a)=Pr(r_{k+1}|s_{k+1}'=s',s_k=s,a_k=a)$:

what is the **probability** of getting the reward $r_{k+1}$ according to state $s_{k}$ and $s_{k+1}$ and action $a$

**VALUE FUNCTION:**

the value of state $s$ **given** a **policy** $\pi$:

$V_\pi(s)= \Bbb E(\sum_t\gamma^tr_t|s_0=s)$

**THE** value function consider that we are using the **absolute best** policy $\pi$ :

$V(s)= \max_\pi \Bbb E(\sum_k^\infin\gamma^k r_k|s_0=s)$

So **bellman** figured this **property** out:

$V(s)= \max_\pi \Bbb E(r_0+\sum_1^\infin\gamma^k r_k|s_1=s)$

so if the value function is a recursive function of itself at next time step, you can break this problem into sub-problems.

$V(s)= \max_\pi \Bbb E(r_0+\gamma V(s'))$

Given a value function, we can extract the policy function.

$\pi(s)= \argmax_\pi \Bbb E(r_0+\gamma V(s'))$

This is the basis of dynamic programming.

> Richard Bellman: How to solve multi-step optimization problems by breaking into smaller recursive sub-problems
> 

Two of the main algorithms that rely on dynamic programming for model-based RL are **Value iteration & Policy iteration** 

### Value iteration

This algorithm allows to use bellman **optimality condition** to **iteratively** build a refined **estimated** of the **Value** function.

$V(s)= \max_a \sum_{s'} P(s'|s,a)(R(s',s,a)+\gamma V(s'))$:

What is the best action $a$ that **maximize** the value at the next state $V(s')$(with a discount) **added** to the current reward, and **multiplied** by the probability of going to the state $s'$ by taking the action $a$  at the state $s$ because we are in a Markov decision process and **not a deterministic** system.

So we need a **table** of every state **rewards** and **probability** of moving from every state to other states, and a **table** of what we **think** is the **value** of each state; this can be **initialized randomly**.

We start at a **random** state and pick the best action according to our three tables, then change the $V(s)$, after many **iterations** the value functions will get more **accurate**.

At the end extract the best policy

$\pi(s,a)= argmax_a \sum_{s'} P(s'|s,a)(R(s',s,a)+\gamma V(s'))$:

### Policy iteration

It is the same recursive idea in value iteration, but it has two step iteration here:

1.**Lock** in a policy $\pi$ and iterate through the $V_\pi(s)$’s and update these values enacting the policy $\pi(s)$ to maximize the expected future rewards.

$V\pi(s)=\Bbb E(R(s',s,\pi(s))+\gamma V_\pi(s'))$

$=\sum_{s'} P(s'|s,\pi(s))(R(s',s,\pi(s))+\gamma V_\pi(s'))$

1. **Lock** in the value function and **sweep** through the actions to update the policy to take the best action, of course here the values need to be estimated correctly so it is related to the first step. Here we are trying to **maximize** over actions the expected future rewards.

$\pi(s)=argmax_a\Bbb E(R(s',s,a)+\gamma V_\pi(s'))$

Typically **converges** in **fewer iterations** than value iteration.

Here both steps are **depending** on the **expectation** of the future rewards, so there is **redundancy**, This one of the reasons why the quality function has been **introduced**. 

$Q(s,a)=\Bbb E(R(s',s,a)+\gamma V_\pi(s'))$

$=\sum_{s'} P(s'|s,a)(R(s',s,a)+\gamma V(s'))$

Then, we can extract the value and policy functions :

$V(s)=max_aQ(s,a)$

$\pi(s,a)=argmax_aQ(s,a)$

The advantage of it is that we don’t need a model of our future $s’$ state, here this information is implicitly stored in the $Q(s,a)$ value.

This can be seen as the same value function redefined, this is the differences between them:

 

**Value function:**

- Represents the expected long-term reward an agent can get starting from a specific state.
- It tells the agent how "good" a particular state is based on future rewards.
- **Input:** Takes only the current state $s$ as input.
- **Notation:** $V(s)$

**Q-value function (Q-function):**

- Represents the expected long-term reward an agent can get by taking a specific action $a$ in a specific state $s$.
- It provides a more granular evaluation, considering both the current state and the potential action.
- **Input:** Takes both the current state  $s$ and the action $a$ as input.
- **Notation:** $Q(s, a)$

## Non-linear dynamics:

This method combines optimal nonlinear control and reinforcement learning techniques, even though there is some overlap between them. 

There is a more general mathematical treatment of Bellman’s equation in this section.

### Hamilton-Jacobi-Bellman equation

In optimal control, the goal is often to find a control input $u(t)$ to drive a dynamical system

$\frac{d}{dt}x=f(x(t),u(t),t)dt$

**to follow a trajectory x(t) that minimizes a cost function**

$J(x(t),u(t),t_0,t_f)=Q(x(t_f),t_f)+\int_{t_0}^{t_f} \mathcal L(x(\tau),u(\tau))d_{\tau}$

**Breakdown:**

$x$:  is the **state** of the system, a vector of **information** that characterize the state

$f(x(t),u(t),t)$: This is a function that defines the **rate of change** of the state variable. It **depends** on the current **state**  $x(t)$, the control **input** $u(t)$ applied to the system at time $t$, and possibly **time** itself $t$.

**Cost function:**

Is function of the trajectory $x(t)$ which varies in time, the control $u(t)$ which varies in time. It has a start time $t_0$ and an end time $t_f$.

$Q(x(t_f),t_f)$: Is called a t**erminal cost** . It captures the **penalty** or **reward** associated with the system **ending** up in a particular state $x(t_f)$ and maybe we can penalize also **how long** it took me to get to there so having a cost proportional to the final time $t_f$.

$\int_{t_0}^{t_f} \cal L(x(\tau),u(\tau))d_{\tau}$: This integral represents the **running cost** at each state**.** It sums up the cost incurred by the system over the time interval from $t_0$ to $t_f$ and also a cost associated with using that control input $u(t)$, maybe you are burning energy and it costs money.  $\tau$  is a dummy variable of integration that helps to sum across all small time steps within the interval because we are operating in a continuous time .

We introduce the value function just like in classic RL that is the cost $J$ when it is minimized over the control input $u$. Given an initial state $x(o)=x(t_0)$, an optimal control $u(t_0)$ will result an optimal cost function $J$

$\partial V(x(t_0),t_0,t_f)=\min_{u(t)}J(x(t),u(t),t_0,t_f)$

It is **not a function** of $u$ because it has been already optimized on it and **neither a function** of $x$ because that is also **specified** by $u$ and the **dynamics** that is guiding the trajectory of $x$, It is **only** a function of the initial condition(state) $x(t_0)$ and the start/end time $t_0,t_f$ respectively. It is often called the ***cost-to-go*** in control theory.

As the value function evaluated at any point $x(t)$ on an optimal trajectory will represent the **remaining** cost associated with continuing to enact this optimal policy until the final time $t_f$. In fact, this is a **statement** of **Bellman’s** optimality principle, that the value function $V$ **remains** optimal starting with any point on an optimal trajectory.

$V(x(t_0),t_0,t_f)=V(x(t_0),t_0,t)+V(x(t),t,t_f)$ : **Bellman Optimality**

**Hamilton-Jacobi-Bellman equation:**

Bellman generalized the Hamilton-Jacobi equation so he wrote this partial differential equation that has to be true for the optimal solutions given by the optimal $u(t)$:

$\large -\frac{\partial V}{\partial t}=\min_{u_t} \left[ (\frac{\partial V}{\partial x})^T f(x(t),u(t))+\mathcal L(x(t) ,u(t))\right]$

To derive the *HJB* equation, we may compute the total time derivative of the value function $V(x(t),t,t_f)$ at some intermediate time $t$:

$\large \frac{d}{dt} V(x(t),t,t_f)= \frac{\partial V}{\partial t} \frac{\partial V}{\partial x}^T \frac{dx}{dt}$

$\large =\min_{u(t)} \frac{d}{dt}\left( \int_0^{t_f} \mathcal L (x(\tau),u(\tau))d\tau+   Q(x(t_f),t_f) \right)$

$\large =\min_{u(t)} \left(\frac{d}{dt}\int_0^{t_f}\mathcal L(x(\tau),u(\tau))d\tau  \right)$

$\implies \large -\frac{\partial V}{\partial t}=\min_{u(t)} \left((\frac{\partial V}{\partial x})^T f(x,u)+\mathcal L(x,u)  \right)$

### Discrete-time HJB

$x_{k+1}=F(x_k,u_k)$ : Discrete-time dynamics

We can write similarly the cost function as the sum of intermediate costs plus the terminal cost:

$J(x_0,\{u_k\}^n_{k=0},n) = \sum_{k=0}^n \mathcal L(x_k,u_k)+Q(x_n,t_n).$

Value function still the same; It is the best possible cost $J$ that I can possibly have starting at a point $x_0$ if I optimize over that control sequence $u(t)$: 

$V(x_0,n)=\min_{\{u_k\}^n_{k=0}} J(x_0,\{ u_k \}^n_{k=0},n)$

Again, Bellman’s principle of optimality states that an optimal control policy has the property that at any point along the optimal trajectory $x(t)$, the remaining control policy is optimal with respect to this new initial state:

![Figure 3: Bellman’s principle of optimality](figs/Figure%203.png)

Figure 3: Bellman’s principle of optimality

$V(x_0,n)=V(x_0,k)+V(x_k,n-k) \forall k \in (0,n)$ :  **Statement of optimality condition**

Thus, the value at an intermediate time step k may be written as:

$V(x_k,n)=(\min_{u_k} \mathcal L(x_k,u_k))+V(x_{k+1},n)$

$=\min_{u_k}(\mathcal L(x_k,u_k)+V(F(x_k,u_k),n)).$ 

This stipulate that taking the best $u_k$ time step so that the current cost plus my future value function at the next step is minimized. So the value function can be broke down into recursion.

Now given that value function $V(x_k,n)$, we can determine the next optimal control action $u_k$ by returning the $u_k$ that minimize it. This defines a policy $u=\pi(x)$

$V(x)=\min_{u}(\mathcal L(x,u)+V(F(x,u))).$ 

$\pi(x)=\argmin_{u}(\mathcal L(x,u)+V(F(x,u))).$ 

# Model-free RL

The motivation behind model-free algorithms is that often we don’t have a model including the probabilities of navigating through states and the rewards at each state. 

## Gradient free

### Off Policy:

### →Monte-Carlo Learning

Is an **episodic** algorithm, it means that it need to run an **entire episode** before it can learn from it. It works in theory for **games** or whatever has a **definitive end.**

We first define this **cumulative reward** function:

$R_{\sum}=\sum_{k=1}^n \gamma^k r_k$

Pick a **policy** and run through an iteration, compute the cumulative function.

Take that reward and **divvy** it **equally** among **every** state (its value) that the system went through.

**Update function:**

$V^{new}(s_k)=V^{old}(s_k)+\frac{1}{n}(R_{\sum}-V^{old}(s_k)) \forall{k}\in [1,..,n]$

And this can be applied to Q-learning:

$Q^{new}(s_k,a_k)=Q^{old}(s_k,a_k)+\frac{1}{n}(R_{\sum}-Q^{old}(s_k,a_k)) \forall{k}\in [1,..,n]$

We **subtract** the old value from the cumulated reward, because if we had the perfect model then this **difference** will be **null**, otherwise the value need to be **modified**.

Still, this isn’t an **optimal** algorithm because it takes a lot of iterations to **converge** (Sample inefficient), One of its advantages is that it has no **bias** i.e. running this algorithm enough time it should converge.

### →Temporal difference Learning : TD(0) 1-step look ahead

The issue with Monte-Carlo algorithm was that the cumulative rewards was distributed **equally** and this isn’t relevant for example if you **play** a game of chess and be good at the **beginning** but blunder at the **end** it will **penalize** every state value even good ones.

What TD(0) does is **relating** the state value to the next state value, and that preserve the **chronology** and **logic** of the experience.

This is a property of **biological** learning like demonstrated in **Pavlov Dog** experiment for example.

So the number of states that we assume can be related to our state $s_k$ is a **tunable** parameter that we can fix.

**Value function:**

$V(s_k)=  \Bbb E(r_k+\gamma V(s_{k+1}))$

**Update function:**

$V^{new}(s_k)=V^{old}(s_k)+\alpha(r_k+\gamma V^{old}(s_{k+1})-V^{old}(s_k))$ 

$r_k+\gamma V^{old}(s_{k+1})$ : **TD Target estimate** $R_{\sum}$       

$(r_k+\gamma V^{old}(s_{k+1})-V^{old}(s_k))$: **TD Error**

It might seem that we are **subtracting** a value from **itself** if we watch the value function, but here $V^{old}(s_{k+1})$ represent the **real** next state $s_{k+1}$ value not the **estimated** one in $V^{old}(s_k)$.

So we have an estimated value of our state $s_k$, and after taking an action an being in the real state $s_{k+1}$ we **measure** how much we were right about it.                                                                                                            

---

There are **analogue** of this in **neuroscience** that when **dopamine** is released, it **strengthen** connections between **cells** that has been activated just **before** the dopamine **release**. This is what’s being modeled by **TD Error**; when there is a reward there is a large **TD Error** so things with $1\Delta t$ will get strengthen in the framework.

### →TD(n): n-step look ahead

This $\Delta t$ can be modified to become :

$V(s_k)=  \Bbb E(r_k+\gamma r_{k+1}+\gamma^2 V(s_{k+2}))$

so the update function become:

$V^{new}(s_k)=V^{old}(s_k)+\alpha(r_k+\gamma r_{k+1}+\gamma^2 V^{old}(s_{k+2})-V^{old}(s_k))$ 

This can be **expanded** to $N$ so it will **converge** to Monte-Carlo learning at $\infin$ as showed in Figure 2.

$R_{\sum}^{(n)}=r_k+\gamma r_{k+1}+\gamma^2 r_{k+2}+...+\gamma^n r_{k+n}+\gamma^{n+1} V(s_{k+n+1})$

$=\sum_{j=0}^n \gamma^jr_{k+j}+\gamma^{n+1}V(s_{k+n+1})$

### →Temporal difference Learning : TD($\lambda$) Weighted look ahead

There is also TD($\lambda$) that instead of taking $\Delta 1$ or $\Delta n$, compute the target estimate $R_{\sum}$ for all states and weight them:

$R_{\sum}^{\lambda}=(1-\lambda)\sum_{n=1}^{\infin} \lambda^{n-1}R_{\sum}^n$

with a factor $1 \gt \lambda \geq0$

**Value update:**

$V^{new}(s_k)=V^{old}(s_k)+\alpha(R_{\sum}^{\lambda}-V^{old}(s_k))$ 

> Monte Carlo learning and TD learning exemplify the bias-variance tradeoff in machine learning. Monte Carlo learning typically has high variance but no bias, while TD learning has lower variance but introduces a bias because of the bootstrapping.
> 

### →Q-Learning

It is temporal difference learning on $Q$-function

$Q^{new}(s_k,a_k)=Q^{old}(s_k,a_k)+\alpha(r_k+\gamma \max_a Q(s_{k+1},a)-Q^{old}(s_k,a_k))$ 

Off policy TD(0) learning of the quality function Q

What we mean by **Off policy** is that we can take **sub-optimal** $a_k$ actions to get the reward but still **maximize** the next action in $s_{k+1}$ though, this helps to learn even when **not taking best** $a_k$ actions.

The Off policy can be **confusing** since we are saying that we can take **sub-optimal** actions but there is that **term** in the update function: $max_a Q(s_{k+1},a)$

**Many** policies are used in **experiments** and at the **experience replay** step we iterate through actions even if they are sub-optimal but we **assume** that the **best** actions will be taken in next steps. This is done by replaying experiments done **by us** or **importing** others and learn from them; this ensure treating **different** policies.

**Exploration vs. exploitation: $\epsilon$-greedy actions**

**Random** exploration element is introduced to $Q$-learning, the popular technique is the  **$\epsilon$-greedy.** Taking the action $a_k$ will be taken based on the current $Q$ function, with a probability $1-\epsilon$, where $\epsilon \in[0,1]$. for example $\epsilon=0.05$ there will be a 95% **probability** of taking best action and 5% **chance** of exploring a sub-optimal one. 

This epsilon value can be decayed as we iterate to go more **On-Policy** once we learned a good $Q$-function.

$Q$ -learning applies to **discrete** action spaces $A$ and state spaces $S$ governed by a **finite** MDP. A table of $Q$ values is used to represent the $Q$ function, and thus it doesn’t **scale** well to **large** state spaces. Typically function **approximation** is used to represent the $Q$ function, such as a **neural network** in deep $Q$-learning.

> Because $Q$-learning is off-policy, it is possible to learn from action-state sequences that do not use the current optimal policy. For example, it is possible to store past experiences, such as previously played games, and replay these experiences to further improve the Q function.
> 

**COMPARISON: $Q$-LEARNING - SARSA** 

$Q$**-learning** learn faster but has higher variance,

can learn from imitation and experience replay.

**SARSA** is often safer because it doesn’t explore what is considered sub-optimal actions,

It has better total reward while learning.

### On Policy:

### →SARSA (State-Action-Reward-State-Action)

 Is the **On-Policy** learning of the $Q$-function       

$Q^{new}(s_k,a_k)=Q^{old}(s_k,a_k)+\alpha(r_k+\gamma Q(s_{k+1},a_{k+1})-Q^{old}(s_k,a_k))$        

Here we need to take the best possible $a_k,a_{k+1}$ actions, otherwise we will learn sub-optimal estimate of the quality function and will start to degrade as we iterate through.

The good thing about **SARSA** is that it can be generalized to any **TD** variance (i.e. **TD(n)** for $n\in[0,N]$), we just need to replace the **TD Target estimate** term by the adequate one.

$R_{\sum}^{(n)}=r_k+\gamma r_{k+1}+\gamma^2 r_{k+2}+...+\gamma^n r_{k+n}+\gamma^{n+1} Q(s_{k+n+1},a_{k+n+1})$

$=\sum_{j=0}^n \gamma^jr_{k+j}+\gamma^{n+1}Q(s_{k+n+1},a_{k+n+1})$  

So you still need that trial and error experience and look back to see what was my rewards at each step and correct the quality function.

We can replace the **TD Target estimate** of course:

$Q^{new}(s_k,a_k)=Q^{old}(s_k,a_k)+\alpha(R_{\sum}^{(n)}-Q^{old}(s_k,a_k))$ 

                                                                                                     

> In an on-policy strategy, such as SARSA, using actions that are sub-optimal, based on the current optimal policy, will degrade the Q function, since the TD target will be a flawed estimate of future rewards based on a sub-optimal action. So experience replay isn’t advised or at least it requires adaptations to account for the on-policy nature of the algorithm.
> 

## Gradient based

# Deep RL

![Figure 4: Deep RL](figs/Figure%204.png)

Q-Learning is mostly used in Deep Reinforcement learning and is called Deep Q-network that learn the $Q(s,a)$ function, once you learn that function if you are in state $s$ it will predict the best action $a$.

This $Q(s,a)$ function can be really complex and that’s one of the strengths of DNN.

## Policy Gradient Optimization

### Deep Policy Network

A policy can be complex specially when number of possible states and actions grow, Creating a Neural network that get in input the state of our agent and parameters through hidden layers that represent the policy to get an output of probabilities of what action to take.

### Policy gradient

Policy gradient is a powerful technique to optimize a policy that is parameterized, so it is possible to use gradient optimization on the parameters  $\theta$ to improve the policy.

The parameterization may be a multi-layer neural network (*deep policy network)*, although other representations and functions approximations may be useful.

Instead of extracting the policy as the argument maximizing the value or quality functions, it is possible to directly optimize the parameters $\theta$ for example through gradient descent or stochastic gradient descent.

The value function $V_\pi(s)$, depending on a policy then becomes $V (s,\theta)$ and a similar modification is possible for the quality function $Q$.

**The total estimated reward is given by** 

 $R_{\sum,\theta}={\sum_{s \in S}\mu_\theta(s)} \sum_{a \in A}{\pi_\theta(s,a)Q(s,a)}$

where $\mu_\theta$ is the asymptotic steady state of the MDP given a policy $\pi_\theta$ parameterized by $\theta$. It it then possible to compute the gradient of the total estimated reward with respect to $\theta$.

$\pi_\theta(s,a)$ represents the probability of taking the action a given that policy $\theta$

$Q(s,a)$  represents the quality of taking that action $a$ at the state $s$ given the policy $\theta$

- **Inner Summation$\sum_{a \in A}$:**
    - For each state $s$ (iterating over all possible states with the $(\sum_{s \in S}$ symbol), we consider all possible actions $a$ the agent can take in that state.
    - For each state-action pair (s, a), we calculate the product of:
        - The probability of taking action $a$ in state $s$ according to the current policy $(\pi(a | s, \theta)$).
        - The $Q$-value for taking action $a$ in state $s$ $(Q(s, a))$.
- **Outer Summation $(\sum_{s \in S}$):**
    - After calculating these products for all actions within a specific state $s$, we take the sum of those products. This essentially sums up the expected future rewards (from $Q$-values) weighted by the action probabilities $(\pi(a | s, \theta))$ for all possible actions in that state $s$.
- **Final Step:**
    - Finally, we perform another summation (represented by the outer $(\sum_{s \in S}$ symbol) over all the possible states $s$ the agent might encounter. This adds up the expected discounted rewards from each state, considering the policy's influence on both reaching those states $(\pi(s | \theta))$ and the action choices within those states $(\pi(a | s, \theta) * Q(s, a))$.

$\nabla_\theta R_{\sum,\theta}={\sum_{s \in S}\mu_\theta(s)} \sum_{a \in A}{Q(s,a)\nabla_\theta\pi_\theta(s,a)}$

$={\sum_{s \in S}\mu_\theta(s)} \sum_{a \in A}{\pi_\theta(s,a)Q(s,a)\frac{\nabla_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)}}$

$={\sum_{s \in S}\mu_\theta(s)} \sum_{a \in A}{\pi_\theta(s,a)Q(s,a)\nabla_\theta log(\pi_\theta(s,a))}$

$=E( Q(s,a)\nabla_\theta log(\pi_\theta(s,a)))$

Then the policy parameters may be updated as

$\theta^{new}=\theta^{old}+ \alpha\nabla_\theta R_{\sum,\theta}$

## Deep Q-Learning

This must be the most used method in recent reinforcement projects.

The basic formula for $Q$-learning still:

$Q^{new}(s_k,a_k)=Q^{old}(s_k,a_k)+\alpha(r_k+\gamma \max_a Q(s_{k+1},a)-Q^{old}(s_k,a_k))$ 

**It is an off policy temporal difference learning of the quality function Q.**

Since we operate in a large state space $S$, addressing these problems with a neural network can solve that dimensionality problem.

We can parametrize the Q-function with a Neural Network:

$Q(s,a) \approx Q(s,a,\theta)$ 

  $\mathcal {L} = \Bbb E[(r_k+\gamma \max_a  Q(s_{k+1},a_{k+1},\theta)-Q(s_k,a_k,\theta))^2]$ 

So the loss function here is the expectation of the difference between the quality function at the state $s_k$ taking action $a_k$ and the quality function of the next state $s_{k+1}$ taking the action at that next state $a_{k+1}$. The NN will use its methods GD,SGD… to optimize the $Q$-function that minimize that temporal difference error. It looks somehow similar to the basic formula of $Q$-learning.

**Why  minimize the expected TD error and not just the raw TD error ?** 

1. **Noise and Variability:** The real world and most simulated environment are inherently noisy. Rewards at next state might not be exactly what the agent expect, minimizing the raw TD error might lead to overfit specific noise patterns, by taking the expectation we average over  multiple experiences and get a more accurate representation of our system trends.
2. **Mitigating Bootstrapping Errors:** Deep $Q$-learning leverages a technique called bootstrapping, Instead of waiting for the actual future reward, it uses its current estimates of future rewards (Q-values) to update its current Q-value. This is efficient, but introduces potential errors if the estimates are inaccurate. 
The expectation helps account for these bootstrapping errors. By averaging the TD error across multiple experiences (which might involve different bootstrapped estimates), the impact of any single inaccurate estimate gets mitigated.

### Advantage Network

This a variation of deep $Q$-learning, called **Deep Dueling Q Network (DDQN)** which split the quality function into two distinct networks : **Value Network $V$, Advantage Network $A$.**

$Q(s,a,\theta)=V(s,\theta_1)+A(s,a,\theta_2)$

**Value Network $V$:** A function of the **value** of current state $s$.

**Advantage Network $A$:** Calculate the **advantage** over just the value of being in that state for taking an action $a$.

This is a good architecture when the difference in quality between states are very **subtle**.

It still just another way of writing the $Q$-function.

**Also**, the advantage function can be defined as so:

$A^{\pi_{\theta}}(s,a)=Q^{\pi_{\theta}}(s,a)-V^{\pi_{\theta}}(s)$

whose value can be interpreted an advantage gained by taking a certain action over the action from the current policy $π_θ(s)$. 

where :

$Q^{\pi_{\theta}}(s_t,a_t)=\Bbb E_{s_{t+1}} \left[ r(s_t,a_t)+ \gamma V^{\pi_{\theta}}(s_{t+1})  \right]$

## Actor-Critic Network

This method came out way before Deep RL but it has been used lately in Neural Networks.

Actor-Critic takes the best out of Policy-based and Value-based learnings.

we are going to have two learners : An actor trying to learn a good policy, and the critic is criticizing  that policy based on its estimate of the value.

The actor is learning a policy and the critic is learning a value function.

$\pi(s,a) \approx \pi(s,a,\theta)$                       **Actor: Policy based**

$V(s_k)=\Bbb E(r_k+\gamma V(s_{k+1}))$        **Critic: Value based**

So how can we do that ?

$\theta_{k+1}=\theta_k+$ $\alpha(r_k+\gamma V(s_{k+1})-V(s_k))$

One simple way to do it is updating the policy parameters $\theta$ using the temporal difference signal from the value learner. 

## Advantage Actor-Critic Network (A2C)

> It is rather straightforward to incorporate deep learning into an actor-critic framework. For example, in the advantage actor critic (A2C) network, the actor is a deep policy network, and the critic is a DDQNs.
> 

$\pi(s,a) \approx \pi(s,a,\theta)$                       **Actor: Deep policy network**

$Q(s_k,a_k,\theta_2)$                                    **Critic: Deep dueling Q network**

$\theta_{k+1}=\theta_k+\alpha \nabla_{\theta}((log\pi(s_k,a_k,\theta))$ $Q(s_k,a_k,\theta_2))$          **Update**
