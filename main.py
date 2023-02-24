import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def main():
    n = int(input())
    m = int(input())
    K = int(input())

    G = [[] for _ in range(n)]
    P = [[0 for _ in range(n)] for _ in range(n)]

    for _ in range(m):
        
        u, v, p = input().split()

        u = int(u)
        v = int(v)
        p = float(p)
        
        G[u].append(v)
        P[u][v] = p

    def consensus(states_in : list):
        # print(f"Initial states: {states}")
        states = states_in.copy()
        for k in range(K):
            state = [0 for _ in range(n)]
            for i in range(n):
                sum = 0
                for j in G[i]:
                    sum += P[i][j] * states[k][j]
                state[i] = sum
            states.append(state)
        return states

    def gossip(states_in : list):
        states = states_in.copy()
        print(f"Initial states: {states}")

        for k in range(K):
            i = np.random.randint(0, n) # Select random i
            p_ij = [P[i][j] for j in G[i]] # We get the p_ij from the P matrix using i neighs
            j = np.random.choice(G[i], 1, p = p_ij)[0] # select random with probability
            
            states.append(states[k].copy()) # It only changes two values i & j, the others remains the same
            states[k + 1][i] = (states[k][i] + states[k][j])/2
            states[k + 1][j] = states[k + 1][i]
        return states

    states = [[np.random.uniform(0, 1) for _ in range(n)]] # We set the initial state as random float [0..1]
    states = [[0.83, 0.46, 0.71, 0.19, 0.04]] # Example shown in the article

    res1 = gossip(states)
    res2 = consensus(states)
    # print(f"Final states: {res}")

    plotGraph(G, P)
    plotResults(("gossip",res1), ("consensus", res2))
    

def plotGraph(g, P):
    G = nx.MultiDiGraph()
    
    for i, N_i in enumerate(g):
        for j in N_i:
            G.add_edge(i, j, weight = round(P[i - 1][j], 2))
    from networkx.drawing.nx_agraph import graphviz_layout

    nx.draw(G, with_labels=True, arrows = True, connectionstyle='arc3, rad = 0.1')

def plotResults(*args):

    num_plots = len(args)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 4*num_plots))


    for i, ax in enumerate(axs):
        # Transpose the matrix to make the columns into rows
        transposed_matrix = list(map(list, zip(*args[i - 1][1])))
        # Plot each row as a separate line
        for row in transposed_matrix:
            ax.plot(row)
        ax.set_title(args[i - 1][0])
        ax.set_xlabel('k')
        ax.set_ylabel('x')

    plt.show()




main()