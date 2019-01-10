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
        self.weight_matrix = np.zeros((n, n), dtype=np.float)
        self.E = E
        self.in_neighb = [[] for _ in range(n)]  # self.in_neighb[i] is the list of j s.t. (j, i) \in E
        self.in_weights = [[] for _ in range(n)]  # the associated weigths to the edges contained in self.in_neighb
        self.out_neighb = [[] for _ in range(n)]  # self.out_neighb[i] is the list of j s.t. (i, j) \in E
        self.out_weights = [[] for _ in range(n)]  # the associated weigths to the edges contained in self.out_neighb

        for k, (i, j) in enumerate(E):
            self.weight_matrix[i, j] = W[k]
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
                out_neighbors_v, _ = self.out_neighbours(v)
                to_visit = [w for w in out_neighbors_v if (v, w) in F]
                stack.extend(list(set(to_visit) - visited))
        return visited
    
    
class IM(Graph):
    def __init__(self, E, W, n):
        super(IM, self).__init__(E, W, n)

    def spread(self, S):
        """
        perform spread forward pass
        :param S: seed set
        :return: the list of the edges activated during the spread,
                 the list of the arms triggered during the spread (ie. the edges connected to an influence node)
                 the number of influenced nodes (ie. the reward)
        """
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
        mean = []
        n_trials = 100  # number of trials to evaluate the average spread
#        timesteps_eval = [1] + [int((T / 100) * i) for i in range(1, 100)]
#        timesteps_eval = list(range(1, T + 1))
        
        for t in range(1, T + 1):
            print("Time step : ", t)
            
            # Define confidence radiuses and upper confidence bounds
            rho = {}
            for e in self.E:
                c = counts[e]
                rho[e] = np.inf if c == 0 else np.sqrt((3 * np.log(t)) / (2 * c))
            mu_bar = {e: min(1, mu_hat[e] + rho[e]) for e in self.E}

            # Draw action to play from the oracle
            # print("Drawing action from the oracle")
            S = self.picked_action(mu_bar, k, o)
            activated_edges, triggered_arms, reward = self.spread(S)

            # Update counts, empirical means, and cumulated reward
            for e in triggered_arms:
                counts[e] += 1
                sum = (counts[e] - 1) * mu_hat[e]
                X = 1 if e in activated_edges else 0
                mu_hat[e] = (X + sum) / counts[e]
            
#            if t in timesteps_eval:
            
            # compute average reward associated to S
            sum = 0
            for trial in range(n_trials):
                _, _, reward = self.spread(S)
                sum += reward
            rewards.append(sum/n_trials)  

            # compute mean distance to GT parameters
            dist = []
            for e in self.E:
                dist.append(abs(mu_bar[e] - self.weight_matrix[e]))
            mean.append(np.mean(dist))

        plt.plot(np.arange(len(mean)), mean)
        plt.xlabel('iteration')
        plt.ylim((0, 1))
        plt.ylabel('mean distance to edge probabilities')
        plt.show()
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
    # graph_name = 'test'
    graph_node = 12831
    # graph_node = 0
    E, W, n = load_graph(graph_name, graph_node)
    # W = [.1, .1, .1, .1, .1, .1, .5, .5, .3, .3, .3, .3,
    #      .3, .3, .3, .3, .5, .5, .7, .7, .5, .5, .7, .7,
    #      .3, .3, .5, .5, .3, .3, .3, .3, .3, .3, .1, .1,
    #      .1, .1, .1, .1, .1, .1, .1, .1]  # only for 'test' graph

    print('Loaded graph from {} dataset, node {}'.format(graph_name, graph_node))
    print('Loaded {} vertices and {} edges'.format(n, len(E)))
    g = Graph(E, W, n)

    T = 10000
    k = 5

    l_mc = 3  # number of simulations used for the Monte-Carlo averages
    mc_oracle = MonteCarloOracle(l_mc)

    epsilon = 0.2  # performance criterion
    p = 0.95  # performance criterion
    l_TIM = l_parameter(n, p)
    tim_oracle = TIM_Oracle(epsilon, l_TIM)

    alg = Bandit(E, W, n)

    print("\nWith Two Phase Influence Maximization (TIM) oracle and CUCB : ")
    mu_hat, rewards = alg.cucb(T, k, tim_oracle)

    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel('iteration')
    plt.ylabel('mean reward')
    plt.show()

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    # plt.xlabel('iteration')
    # plt.ylabel(r'$\text{Reg}(T) / \sqrt{t\log t}$')
    # plt.plot(np.arange(len(cum_regret)),
    #          [cum_regret[t - 1] / np.sqrt(t*np.log(t))
    #           for t in range(1, len(cum_regret) + 1)])
    # plt.show()
