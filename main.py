import networkx as nx
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.sparse import csc_matrix

# The function to calulate the PageRank

def pageRank(G, s = .85, maxerr = .0001):
    
    # Computes the pagerank for each of the n states

    # G: matrix representing state transitions
    #    Gij is a binary value representing a transition from state i to j.

    # s: probability of following a transition. 1-s probability of teleporting to another state.

    # maxerr: if the sum of pageranks between iterations is bellow this we will have converged.

    n = G.shape[0]
    A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]

    sink = rsums==0

    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > maxerr:
        ro = r.copy()
        for i in range(0,n):
            Ai = np.array(A[:,i].todense())[:,0]
            Di = sink / float(n)
            Ei = np.ones(n) / float(n)

            r[i] = ro.dot( Ai*s + Di*s + Ei*(1-s) )
    return r/float(sum(r))

#Function to calculate the Rank Prestige

def rankpres(G):
    n = G.shape[0]
    A = csc_matrix(G,dtype=np.float)
    rsums = np.array(A.sum(1))[:,0]
    ri, ci = A.nonzero()
    A.data /= rsums[ri]
    sink = rsums==0
    ro, r = np.zeros(n), np.ones(n)
    while np.sum(np.abs(r-ro)) > .0001:
        ro = r.copy()
        for i in range(0,n):
            Ai = np.array(A[:,i].todense())[:,0]
            Di = sink / float(n)
            Ei = np.ones(n) / float(n)
            r[i] = ro.dot( Ai*1 + Di*1)
    return r/float(sum(r))

G=nx.DiGraph()

#Open and Read the CSV File. Get the names of the Nodes.

with open('node.csv', 'r') as nodecsv:
    nodereader = csv.reader(nodecsv)
    nodes = [n for n in nodereader][1:]
node_names = [n[0] for n in nodes] 
with open('edges.csv', 'r') as edgecsv:
    edgereader = csv.reader(edgecsv) 
    edges = [tuple(e) for e in edgereader][1:]

G.add_nodes_from(node_names)
G.add_edges_from(edges)

#Printing the information of the graph.

print(nx.info(G))
v=nx.spring_layout(G)
plt.axis('off')
nx.draw(G,pos=v,with_labels=False,node_size = 20)
plt.show()

# Network Density Of the Graph

density = nx.density(G)
print("Network density:", density)

#Indegree and OutDegree Centrality of the nodes

indeg=nx.in_degree_centrality(G)
outdeg=nx.out_degree_centrality(G)
print("\n\n In-Degree centrality: \n",indeg)
print("\n\n Out-Degree centrality: \n",outdeg)

#Betweenness Centraity of the nodes.

bet=nx.betweenness_centrality(G)
print("\n\nNetwork betweenness:", bet)

#Closness Centrality of the nodes.

close=nx.closeness_centrality(G)
print("\n\nNetwork Closeness:\n",close)

#Total Number of nodes.

length = len(G)

#Indegree and Outdegree of the nodes

print("\n\n Total number of nodes : ", length)
print('\n\n Indegree of all the nides are : ')
l = G.in_degree(node_names)
print(l)
print('\n\n Outdegree of all the nodes are : ')
m = G.out_degree(node_names)
print(m)

#Degree Prestige of nodes.

x = [(name, v/(length - 1)) for (name, v) in l]
print("\n\nDegree prestige of the nodes are :\n",x)

#Proximity Prestige of nodes.

pin=[]
for y in node_names:
    b=0
    for z in node_names:
        b=b+nx.shortest_path_length(G, source=y, target=z, weight=None)
    pin.insert(len(pin),b)
pp=[]
for q in pin:
    pp.append(q/length-1)

print("\n\nProximity prestige of the nodes are : \n",pp)

#Adjacency Matrix

A = nx.adjacency_matrix(G)
#print(A.toarray())
E = np.array(A.toarray())
C = np.array(A.toarray()).tolist()
#print(C)

#Rank Prestige of all nodes.

print("\n\nRank Prestige of the nodes are :\n",rankpres(E))

#printing of Adjacency Matrix

print("\n\nAdjacency Matrix :",A.toarray())

#Input i and j are taken from the user.

i = int(input("Enter node one(i): "))
j = int(input("Enter node two(j): "))


#Co-citation calculation.

result=0
k=0
for k in range(length):
    result=result+(C[k][i]*C[k][j])
    k=k+1

#Printing of Co-citation

print("\n\n Co-citation of the nodes :\n",result)

#Biblographic Coupling Calculation

result1=0
k=0
for k in range(length):
    result1=result1+(C[i][k]*C[j][k])
    k=k+1

#Printing of Biblographic Coupling

print("\n\n Biblographic Coupling of the nodes :\n",result1)

# Printing of pagerank of each node.

print("\n\nPageRank of the nodes are :\n",pageRank(E))
