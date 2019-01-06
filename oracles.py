import numpy as np
from scipy.special import binom as binom
import progressbar


class Oracle:
    """
    Abstract class for oracles
    An oracle has to have a method action(g, k) returning the predicted k-set chosen from graph g
    """
    def action(self, g, k):
        pass


class MonteCarloOracle(Oracle):
    """
    Implementation of Kempe et al.'s greedy oracle
    """

    def __init__(self, l):
        self.l = l

    def expected_spread(self, g, S):
        """ Approximates the expected number of influenced nodes f(S),
        starting with seed set S, by averaging l simulations """
        r_hat = 0
        for i in range(self.l):
            _, _, reward = g.spread(S)
            r_hat += reward
        return int(r_hat / self.l)

    def action(self, g, k):
        
        S = []
        
        for t in range(k):
            values = [self.expected_spread(g, S + [v]) for v in range(g.nb_nodes)]
            for v in S:
                values[v] = - np.inf
            v_t = np.argmax(values)
            S.append(v_t)

        return S
    
    
class TIM_Oracle(Oracle):
    """
    Implementation of Tang et al.'s Two-phase Influence Maximisation (TIM) oracle
    """
    def __init__(self, epsilon, l):
        self.epsilon = epsilon
        self.l = l

    def random_rr_set(self, g):
        v_0 = np.random.randint(g.nb_nodes)
        queue = [v_0]
        R = []
        visited = [False for _ in range(g.nb_nodes)]
        while len(queue) > 0:
            v = queue.pop()
            R.append(v)
            visited[v] = True
            neighbours, weights = g.in_neighbours(v)
            
            for w, u in list(zip(weights, neighbours)):
                if u not in R:
                    x = np.random.uniform()
                    if x < w:
                        queue.append(u)
        return R
    
    def width(self, g, R):
        return sum([len(g.in_neighb[v]) for v in R])

    def kpt_estimation(self, g, k):
        c = 6 * self.l * np.log(g.nb_nodes) + 6 * np.log(np.log2(g.nb_nodes))
        eps_= 5 * np.power((self.l * pow(self.epsilon, 2)) / (k + self.epsilon), 1 / 3)

#        print('KPT estimation')
        over = False
        kpt = 0
        R_list = []
        for i in range(1, int(np.log2(g.nb_nodes))):
            s = 0
            c *= 2
            R_list = []
            for j in range(1, int(c)):
                R = self.random_rr_set(g)
                R_list.append(R)
                s += 1 - pow(1 - self.width(g, R) / g.nb_edges, k)
            if s / c > pow(2, -i):
                kpt = (g.nb_nodes * s) / (2 * c)
                over = True
                break
            
        if not over:
            kpt = 1
            return kpt
            
#        print('KPT refinement')
        S = []

        for t in range(k):
            values = [sum([v in R for R in R_list]) for v in range(g.nb_nodes)]
            for v in S:
                values[v] = - np.inf
            v_t = np.argmax(values)
            S.append(v_t)
            R_list = [R for R in R_list if v_t not in R]

        lbda_ = (2 + eps_) * self.l * g.nb_nodes * np.log(g.nb_nodes) * np.power(eps_, -2)
        theta_ = int(lbda_ / kpt)
#        print("lbda_ : ", lbda_)
#        print("kpt : ", kpt)
#        print("theta_ : ", theta_)
        R_list = []
        for _ in range(theta_):
            R_list.append(self.random_rr_set(g))
            
        f = 1
        if theta_ != 0:
            f = sum([any(v in R for v in S) for R in R_list]) / len(R_list)
            
        kpt_ = f * g.nb_nodes / (1 + eps_)
        
        return max(kpt, kpt_)
    

    def node_selection(self, g, k, theta):
        
        R_list = []
        S = []
        
        for _ in range(int(theta)):
            R_list.append(self.random_rr_set(g))
        
        
#        print('Node selection:')
#        for _ in progressbar.ProgressBar(range(int(theta))):
#            R_list.append(self.random_rr_set(g))

        for t in range(k):
            values = [sum([v in R for R in R_list]) for v in range(g.nb_nodes)]
            for v in S:
                values[v] = - np.inf
            v_t = np.argmax(values)
            S.append(v_t)
            R_list = [R for R in R_list if v_t not in R]
            
        return S
        
    def action(self, g, k):
        kpt = self.kpt_estimation(g, k)
#        print("Finished KPT estimation")
#        print("KPT : ", kpt)
        lbda = (8 + 2 * self.epsilon) * g.nb_nodes
        lbda *= (self.l * np.log(g.nb_nodes) + np.log(binom(g.nb_nodes, k)) + np.log(2))
        lbda *= pow(self.epsilon, -2)
        theta = lbda / kpt
#        print("Theta : ", theta)
    
#        print("Beginning node selection")
        return self.node_selection(g, k, theta)
    
    
def l_parameter(n, p):
    """ Returns the l parameter such that the TIM oracle guarantees its
    performance with probability p when running on a graph with n nodes """
    return (- np.log(1 - p)) / np.log(n)
