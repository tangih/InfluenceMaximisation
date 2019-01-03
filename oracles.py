import numpy as np
from scipy.special import binom as binom


class MonteCarloOracle:
    
    def __init__(self, l):
        self.l = l
    
    
    def expected_spread(self, g, S):
        """ Approximates the expected number of influenced nodes f(S),
        starting with seed set S, by averaging l simulations """
        r_hat = 0
        for i in range(self.l):
            _, _, reward = g.run(S)
            r_hat += reward
        return int(r_hat / self.l)
    
    
    def approx(self, g, k):
        
        S = []
        non_chosen = g.V.copy()
            
        for t in range(k):
            values = []
            for v in g.V:
                if v in non_chosen:
                    values.append(self.expected_spread(g, S + [v]))
                else:
                    values.append(- np.inf)
            v_t = np.argmax(values)
            S.append(v_t)
            non_chosen.remove(v_t)
        
        return S
    
    
class TIMOracle:
    
    def __init__(self, epsilon, l):
        self.epsilon = epsilon
        self.l = l
    
    
    def random_RR_set(g):
        v_0 = np.random.choice(g.V)
        queue = [v_0]
        R = []
        
        while queue != []:
            v = queue.pop()
            R.append(v)
            
            for u in g.in_neighbors(v):
                if u not in R:
                    x = np.random.uniform()
                    if x < g.p(u, v):
                        queue.append(u)
        
        return R
    
    
    def width(g, R):
        return sum([len(g.in_neighbors(v)) for v in R])
    
    
    def KPT_estimation(self, g, k):
        n = g.nb_nodes
        m = len(g.E)
        c = 6 * self.l * np.log(n) + 6 * np.log(np.log2(n))
        
        for i in range(1, int(np.log2(n))):
            s = 0
            print("Step {} of KPT estimation".format(i))
            c *= 2
            for j in range(1, int(c)):
                R = TIMOracle.random_RR_set(g)
                s+= 1 - pow(1 - TIMOracle.width(g, R) / m, k)
            if ((s / c) > (pow(2, -i))):
                return (n * s) / (2 * c)
        
        return 1
    
    
    def node_selection(g, k, theta):
        
        R_list = []
        S = []
        non_chosen = g.V.copy()
        
        for i in range(int(theta)):
            if (i % 10000) == 0:
                print("Step {} of node selection RR sets generation".format(i))
            R_list.append(TIMOracle.random_RR_set(g))
        
        values = [sum([v in R for R in R_list]) for v in g.V]
        for t in range(k):
            v_t = np.argmax(values)
            S.append(v_t)
            non_chosen.remove(v_t)
            values[v_t] = - np.inf
        
        return S
    
        
    def approx(self, g, k):
        
        n = g.nb_nodes
        KPT = self.KPT_estimation(g, k)
        print("Finished KPT estimation")
        print("KPT : ", KPT)
        lbda = (8 + 2 * self.epsilon) * n
        lbda *= (self.l * np.log(n) + np.log(binom(n, k)) + np.log(2))
        lbda *= pow(self.epsilon, -2)
        theta = lbda / KPT
        print("Theta : ", theta)
    
        print("Beginning node selection")
        return TIMOracle.node_selection(g, k, theta)
    
    
def l_parameter(n, p):
    """ Returns the l parameter such that the TIM oracle guarantees its
    performance with probability p when running on a graph with n nodes """
    return (- np.log(1 - p)) / np.log(n)
    