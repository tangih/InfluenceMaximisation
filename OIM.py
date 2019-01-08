import numpy as np
from oracles import MonteCarloOracle, TIM_Oracle, l_parameter
from load_data import load_graph
import matplotlib.pyplot as plt


class Graph:
    """
    Implementation of a graph structure
    For more efficiency, we use a sparse graph structure.
    """
    def __init__(self, E, W, n):
        """
        creates the sparse graph structure
        :param E: the list of edges
        :param W: the associated weights to the edges contained in E
        :param n: we ensure that V = \{1, ..., n\}
        """
        self.nb_nodes = n
        self.nb_edges = len(E)
        self.weight_matrix = np.zeros((n, n), dtype=np.float)  # TODO: remove this
        self.E = E
        self.in_neighb = [[] for _ in range(n)]  # self.in_neighb[i] is the list of j s.t. (j, i) \in E
        self.in_weights = [[] for _ in range(n)]  # the associated weigths to the edges contained in self.in_neighb
        self.out_neighb = [[] for _ in range(n)]  # self.out_neighb[i] is the list of j s.t. (i, j) \in E
        self.out_weights = [[] for _ in range(n)]  # the associated weigths to the edges contained in self.out_neighb

        for k, (i, j) in enumerate(E):
            self.weight_matrix[i, j] = W[k]  # TODO: remove this
            self.in_neighb[j].append(i)
            self.in_weights[j].append(W[k])
            self.out_neighb[i].append(j)
            self.out_weights[i].append(W[k])

    def in_neighbours(self, i):
        return self.in_neighb[i], self.in_weights[i]
        
    def out_neighbours(self, i):
        return self.out_neighb[i], self.out_weights[i]

    def p(self, i, j):
        # TODO: remove this
        return self.weight_matrix[i, j]

    def dfs(self, S, F):
        # TODO: improve this
        """ Determines all nodes accessible from the set of nodes S through the
        subset of edges F, using the Depth-First Search algorithm """
        assert (all(x in self.E for x in F)), "Please provide a subset of edges of the graph"
        visited, stack = set(), S.copy()
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                out_neighbors_v, _ = self.out_neighbours(v)
                to_visit = [w for w in out_neighbors_v if (v, w) in F]
                stack.extend(list(set(to_visit) - visited))
        return visited
    
    
class IM(Graph):
    def __init__(self, E, W, n):
        super(IM, self).__init__(E, W, n)

    def spread(self, S):
        # assert (all(x in self.V for x in S)), "Seed set should be a subset of the graph nodes set"
        # if len(S) != len(set(S)):
        #     print("Removing duplicates from the seed set")
        #     S = list(set(S))
        
        activated_edges = []

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


class Bandit(IM):
    def __init__(self, E, W, n):
        super(Bandit, self).__init__(E, W, n)
        
        
    def picked_action(self, mu, k, o):
        """ Returns an action S using oracle o and estimated probabilities 
        mu for the edges """
        weights = []
        edges = []
        for e in mu.keys():
            weights.append(mu[e])
            edges.append(e)
        graph = IM(edges, weights, self.nb_nodes)
        # W_hat = np.zeros([self.nb_nodes, self.nb_nodes])
        # for (i, j) in mu.keys():
        #     W_hat[i, j] = mu[(i, j)]
            
#        print("Created the graph. Beginning approximation")
        return o.action(graph, k)
        
    def cucb(self, T, k, o):
        """
        Online Influence Maximization Bandit, using the CUCB algorithm
        T: number of times steps
        k: maximum size of the seed sets used
        o: oracle used """
        
        counts = {e: 0 for e in self.E}
        mu_hat = {e: 1 for e in self.E}
        rewards = []
        
        for t in range(1, T + 1):
            print("Time step : ", t)
            
            # Define confidence radiuses and upper confidence bounds
            rho = {}
            for e in self.E:
                c = counts[e]
                rho[e] = np.inf if c == 0 else np.sqrt((3 * np.log(t)) / (2 * c))
            mu_bar = {e: min(1, mu_hat[e] + rho[e]) for e in self.E}

            # Draw action to play from the oracle
#            print("Drawing action from the oracle")
            S = self.picked_action(mu_bar, k, o)
            activated_edges, triggered_arms, reward = self.spread(S)
            
            # Update counts, empirical means, and cumulated reward
            for e in triggered_arms:
                counts[e] += 1
                X = 1 if e in activated_edges else 0
                mu_hat[e] += (X - mu_hat[e]) / counts[e]
            rewards.append(reward)
        
        return mu_hat, rewards

    def thompson(self, T, k, o):
        """
        Thompson sampling algorihm
        Similar usage to CUCB
        """
        cum_rew = {e: 0 for e in self.E}
        count = {e: 0 for e in self.E}
        rewards = []

        for t in range(1, T + 1):
            print("Time step : ", t)

            mu_tilde = {}
            for e in self.E:
                mu_tilde[e] = np.random.beta(cum_rew[e] + 1, count[e] - cum_rew[e] + 1)

            # Draw action to play from the oracle
            #            print("Drawing action from the oracle")
            S = self.picked_action(mu_tilde, k, o)
            activated_edges, triggered_arms, reward = self.spread(S)

            # Update counts, empirical means, and cumulated reward
            for e in triggered_arms:
                count[e] += 1
                X = 1 if e in activated_edges else 0
                cum_rew[e] += X
            rewards.append(reward)

        return mu_hat, rewards


if __name__ == '__main__':

    np.random.seed(0)
    graph_name = 'twitter'
    graph_node = 12831
    E, W, n = load_graph(graph_name, graph_node)
    print('Loaded graph from {} dataset, node {}'.format(graph_name, graph_node))
    print('Loaded {} vertices and {} edges'.format(n, len(E)))
    g = Graph(E, W, n)

    T = 20
    k = 10

    l_mc = 3  # number of simulations used for the Monte-Carlo averages
    mc_oracle = MonteCarloOracle(l_mc)

    epsilon = 0.1  # performance criterion
    p = 0.99  # performance criterion
    l_TIM = l_parameter(n, p)
    tim_oracle = TIM_Oracle(epsilon, l_TIM)
    
#    S = tim_oracle.action(g, 5)
#    print("S : ", S)
    
    alg = Bandit(E, W, n)
    
#    print("\nWith Monte Carlo oracle : ")
#    mu_hat, rewards = alg.cucb(T, k, mc_oracle)
##    print("mu_hat : ", mu_hat)
#    print("cumulated_reward : ", np.cumsum(rewards)[-1])
    
    print("\nWith Two Phase Influence Maximization (TIM) oracle and CUCB : ")
    mu_hat, rewards = alg.cucb(T, k, tim_oracle)
#    print("mu_hat : ", mu_hat)
#    print("cumulated_reward : ", np.cumsum(rewards)[-1])
    plt.plot(np.arange(T), rewards)
    plt.show()
    
#    print("\nWith Two Phase Influence Maximization (TIM) oracle and Thompson sampling : ")
#    mu_hat, rewards = alg.thompson(T, k, tim_oracle)
##    print("mu_hat : ", mu_hat)
##    print("cumulated_reward : ", np.cumsum(rewards)[-1])
#    plt.plot(np.arange(T), rewards)
#    plt.show()
