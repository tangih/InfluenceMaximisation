import numpy as np
from scipy.special import binom as binom
import progressbar


class Oracle:
    def approx(self, k):
        pass


class MonteCarloOracle(Oracle):
    """
    Implementation of Kempe et al.'s greedy oracle
    """

    def __init__(self, graph, l):
        self.g = graph
        self.l = l

    def expected_spread(self, S):
        """ Approximates the expected number of influenced nodes f(S),
        starting with seed set S, by averaging l simulations """
        r_hat = 0
        for i in range(self.l):
            _, _, reward = self.g.spread(S)
            r_hat += reward
        return int(r_hat / self.l)

    def approx(self, k):
        S = []
        non_chosen = self.g.V.copy()

        for t in range(k):
            values = []
            for v in self.g.V:
                if v in non_chosen:
                    values.append(self.expected_spread(S + [v]))
                else:
                    values.append(- np.inf)
            v_t = np.argmax(values)
            S.append(v_t)
            non_chosen.remove(v_t)

        return S
    
    
class TIM_Oracle(Oracle):
    """
    Implementation of Tang et al.'s Two-phase Influence Maximisation (TIM) oracle
    """
    def __init__(self, graph, epsilon, l):
        self.g = graph
        self.n = graph.nb_nodes
        self.m = graph.nb_edges
        self.epsilon = epsilon
        self.l = l

    def random_rr_set(self):
        v_0 = np.random.randint(self.n)
        queue = [v_0]
        R = []
        visited = [False for _ in range(self.n)]
        while len(queue) > 0:
            v = queue.pop()
            R.append(v)
            visited[v] = True
            neighbours, weights = self.g.in_neighbours(v)
            
            for w, u in list(zip(weights, neighbours)):
                if u not in R:
                    x = np.random.uniform()
                    if x < w:
                        queue.append(u)
        return R
    
    def width(self, R):
        return sum([len(self.g.in_neighbors(v)) for v in R])

    def kpt_estimation(self, k):
        c = 6 * self.l * np.log(self.n) + 6 * np.log(np.log2(self.n))

        print('KPT estimation:')
        for i in progressbar.ProgressBar(range(1, int(np.log2(self.n)))):
            s = 0
            c *= 2
            for j in range(1, int(c)):
                R = self.random_rr_set()
                s += 1 - pow(1 - self.width(R) / self.m, k)
            if s / c > pow(2, -i):
                return (self.n * s) / (2 * c)
        return 1
    
    def node_selection(self, k, theta):
        R_list = []
        S = []
        print('Node selection:')
        for _ in progressbar.ProgressBar(range(int(theta))):
            R_list.append(self.random_rr_set())
        
        values = [sum([v in R for R in R_list]) for v in range(self.n)]
        for t in range(k):
            v_t = np.argmax(values)
            S.append(v_t)
            values[v_t] = -np.inf
        return S
        
    def approx(self, k):
        kpt = self.kpt_estimation(k)
        print("Finished KPT estimation")
        print("KPT : ", kpt)
        lbda = (8 + 2 * self.epsilon) * self.n
        lbda *= (self.l * np.log(self.n) + np.log(binom(self.n, k)) + np.log(2))
        lbda *= pow(self.epsilon, -2)
        theta = lbda / kpt
        print("Theta : ", theta)
    
        print("Beginning node selection")
        return self.node_selection(k, theta)
    
    
def l_parameter(n, p):
    """ Returns the l parameter such that the TIM oracle guarantees its
    performance with probability p when running on a graph with n nodes """
    return (- np.log(1 - p)) / np.log(n)
