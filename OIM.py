import numpy as np
from oracles import MonteCarloOracle, TIM_Oracle, l_parameter
from load_data import load_graph


class Graph:
    
    def __init__(self, E, W, n):
        self.nb_nodes = n
        self.nb_edges = len(E)
        self.weight_matrix = np.zeros((n, n), dtype=np.float)
        self.E = E
        self.in_neighb = [[] for _ in range(n)]
        self.in_weights = [[] for _ in range(n)]
        self.out_neighb = [[] for _ in range(n)]
        self.out_weights = [[] for _ in range(n)]

        for k, (i, j) in enumerate(E):
            self.weight_matrix[i, j] = W[k]  # has to be removed for sparse graphs
            self.in_neighb[j].append(i)
            self.in_weights[j].append(W[k])
            self.out_neighb[i].append(j)
            self.out_weights[i].append(W[k])

    def in_neighbours(self, i):
        return self.in_neighb[i], self.in_weights[i]
        
    def out_neighbours(self, i):
        return self.out_neighb[i], self.out_weights[i]

    def p(self, i, j):
        return self.weight_matrix[i, j]

    def dfs(self, S, F):
        """ Determines all nodes accessible from the set of nodes S through the
        subset of edges F, using the Depth-First Search algorithm """
        assert (all(x in self.E for x in F)), "Please provide a subset of edges of the graph"
        visited, stack = set(), S.copy()
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                neighbors_v = self.out_neighbours(v)
                to_visit = [w for w in neighbors_v if (v, w) in F]
                stack.extend(list(set(to_visit) - visited))
        return visited
    
    
class IM(Graph):
    
    def __init__(self, W):
        super(IM, self).__init__(W)

    def spread(self, S):
        assert (all(x in self.V for x in S)), "Seed set should be a subset of the graph nodes set"
        if len(S) != len(set(S)):
            print("Removing duplicates from the seed set")
            S = list(set(S))
        
        # Initialize activated edges and triggered arms data structures
        activated_edges = []
        triggered_arms = []
        
        # Sample the arms (i.e. the edges) from their distribution
        # (independent Bernoulli variables)
        for (i, j) in self.E:
            u = np.random.uniform()
            if u < self.p(i, j):
                activated_edges.append((i, j))
                
        # Determine the triggered arms and the influenced nodes
        influenced_nodes = self.dfs(S, activated_edges)
        triggered_arms = [e for e in self.E if e[0] in influenced_nodes]
        
        # Return the activated edges, the triggered arms and the reward
        return activated_edges, triggered_arms, len(influenced_nodes)


class CUCB(IM):
    
    def __init__(self, W):
        super(CUCB, self).__init__(W)
        
        
    def oracle(self, o, mu, k):
        """ Returns an action S using oracle o and estimated probabilities 
        mu for the edges """
        
        W_hat = np.zeros([self.nb_nodes, self.nb_nodes])
        for (i, j) in mu.keys():
            W_hat[i, j] = mu[(i, j)]
            
        print("Created the graph. Beginning approximation")
            
        return o.approx(IM(W), k)
        
        
    def bandit(self, T, k, o):
        """ Online Influence Maximization Bandit
        T: number of times steps
        k: maximum size of the seed sets used
        o: oracle used """
        
        counts = {e : 0 for e in self.E}
        mu_hat = {e : 1 for e in self.E}
        cumulated_reward = 0
        
        for t in range(1, T + 1):
            print("Time step : ", t)
            
            # Define confidence radiuses and upper confidence bounds
            rho = {}
            for e in self.E:
                c = counts[e]
                rho[e] = np.inf if c == 0 else np.sqrt((3 * np.log(t)) / (2 * c))
            mu_bar = {e : min(1, mu_hat[e] + rho[e]) for e in self.E}
                
            # Draw action to play from the oracle
            print("Drawing action from the oracle")
            S = self.oracle(o, mu_bar, k)
            activated_edges, triggered_arms, reward = self.run(S)
            
            # Update counts, empirical means, and cumulated reward
            for e in triggered_arms:
                counts[e] += 1
                X = 1 if e in activated_edges else 0
                mu_hat[e] += (X - mu_hat[e]) / counts[e]
            cumulated_reward += reward
        
        return mu_hat, cumulated_reward   
    

if __name__ == '__main__':
    # Directed graph with 6 nodes
    # W = np.array([
    #        [0, 0.5, 0, 0, 0, 0],
    #        [0, 0, 0.2, 0, 0, 0],
    #        [0.2, 0, 0, 0, 0, 0],
    #        [0, 0, 0.5, 0, 0.5, 0],
    #        [0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0.5, 0.3, 0]
    #    ])

    # Facebook subgraph
    E, W, n = load_graph('twitter', 12831)
    g = Graph(E, W, n)

    # im = IM(W)
    # print("V : ", im.V)
    # print("E : ", im.E)
    # activated_edges, triggered_arms, reward = im.run([3])
    # print("Activated edges : ", activated_edges)
    # print("Triggered arms : ", triggered_arms)
    # print("Reward : ", reward)

    alg = CUCB(W)
    T = 10
    k = 2

    l_MC = 200  # number of simulations used for the Monte-Carlo averages
    MC = MonteCarloOracle(g, l_MC)

    epsilon = 0.2  # performance criterion
    p = 0.95  # performance criterion
    l_TIM = l_parameter(alg.nb_nodes, p)
    print("l_TIM : ", l_TIM)
    TIM = TIM_Oracle(g, epsilon, l_TIM)

    # print("\nWith Monte Carlo oracle : ")
    # mu_hat, cumulated_reward = alg.bandit(T, k, MC)
    # print("mu_hat : ", mu_hat)
    # print("cumulated_reward : ", cumulated_reward)

    print("\nWith Two Phase Influence Maximization (TIM) oracle : ")
    mu_hat, cumulated_reward = alg.bandit(T, k, TIM)
    print("mu_hat : ", mu_hat)
    print("cumulated_reward : ", cumulated_reward)
