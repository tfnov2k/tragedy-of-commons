import numpy as np
import matplotlib.pyplot as plt

N = 4  # Number of farmers
K = 40  # Carrying capacity of the pasture
H = 100  # Initial health of the pasture
D = 0.06  # Degradation factor
R = 0.1  # Recovery factor
P = 5  # Profit per cow
λ = 500  # Weight for health in the objective function
learning_rate = 0.002  # Learning rate for gradient descent
T = 1  # Time step
rounds = 800  # Number of simulation rounds
cooperation_limit = K / N  # Number of cows allowed for cooperating farmers
defection_factor = 1.5  # Defecting farmers can have 1.5x more cows
growth_rate = 0.002  # Rate at which defecting farmers increase their cows each round
discount_factor = 0.9  # Discount factor for long-term profit calculation

np.random.seed(2)

cows = np.random.uniform(5, 10, N)  # Initial random cow count for each farmer
strategies = [True, True, False, False]  # Two farmers cooperate, two defect

health = H  # Initial health of the pasture

def total_cows(cows):
    return np.sum(cows)

def pasture_health(health, Ctotal, K, D, R):
    if Ctotal > K:
        overgrazing_factor = (Ctotal - K) ** 1.2 
        health -= D * overgrazing_factor
    else:
        health += R * (K - Ctotal)
    health = min(max(health, 50), 100)
    return health

def loss(cows, P, λ, health, defecting=False):
    sustainability_reward = 0.1 * (health - 50)
    penalty = 0
    if defecting and cows > cooperation_limit:
        penalty = (cows - cooperation_limit) ** 1.5  # Penalty for defectors overgrazing
    return P * cows - λ * health + sustainability_reward - penalty

def update_cows(cows, strategies, Ctotal, K, health):
    new_cows = np.copy(cows)
    for i in range(N):
        if strategies[i]:  # Cooperating farmers
            grad = P - λ * (-D if Ctotal > K else R)
            new_cows[i] -= learning_rate * grad
            new_cows[i] = max(0, new_cows[i]) 
        else:  # Defecting farmers
            defect_limit = defection_factor * cooperation_limit
            if new_cows[i] < defect_limit:
                new_cows[i] += growth_rate * (defect_limit - new_cows[i])
            new_cows[i] = min(new_cows[i], defect_limit)  # Limit defection
    return new_cows

# Simulation
health_history = []
cow_history = []
loss_history = []

for round in range(rounds):
    Ctotal = total_cows(cows)
    health = pasture_health(health, Ctotal, K, D, R)
    new_cows = update_cows(cows, strategies, Ctotal, K, health)

    print(f"--- Round {round + 1} ---")
    print(f"Total cows before update: {Ctotal:.2f}")
    print(f"Pasture Health before update: {health:.2f}")

    total_loss = 0
    long_term_loss = 0

    for i in range(N):
        # Calculate individual loss for farmer
        farmer_loss = loss(new_cows[i], P, λ, health)
        total_loss += farmer_loss

        # Calculate long-term loss for cooperating farmers
        if strategies[i]:
            long_term_loss += (P * new_cows[i] - λ * health) * discount_factor ** round

        action = "Cooperated" if strategies[i] else "Defected"
        print(f"Farmer {i+1}: {action}, Cows = {cows[i]:.2f} -> {new_cows[i]:.2f}, Loss = {farmer_loss:.2f}")

    cows = new_cows
    health_history.append(health)
    cow_history.append(np.sum(cows))
    loss_history.append(total_loss)

    # Output current state
    print(f"Total cows after update: {np.sum(cows):.2f}")
    print(f"Pasture Health after update: {health:.2f}")
    print(f"Total loss: {total_loss:.2f}\n")

# Plotting the health, total cows, and loss over time
plt.figure(figsize=(16, 5))

# Plot pasture health over time
plt.subplot(1, 2, 1)
plt.plot(health_history)
plt.title('Pasture Health Over Time')
plt.xlabel('Round')
plt.ylabel('Health')

# Plot total loss over time
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title('Total Loss Over Time')
plt.xlabel('Round')
plt.ylabel('Total Loss')

plt.tight_layout()
plt.show()

