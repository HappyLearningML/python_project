#-*- coding:utf-8-*-
'''
topological sort:拓扑排序，对一个有向无环图(Directed Acyclic Graph简称DAG)G进行排序，
是将G中所有顶点排成一个线性序列，使得图中任意一对顶点u和v，若边(u,v)∈E(G)，则u在线性序列中出现在v之前
'''
#     a
#    / \
#   b  c
#  / \
# d  e
edges = {'a': ['c', 'b'], 'b': ['d', 'e'], 'c': [], 'd': [], 'e': []}
vertices = ['a', 'b', 'c', 'd', 'e']


def topological_sort(start, visited, sort):
    """Perform topolical sort on a directed acyclic graph."""
    current = start
    # add current to visited
    visited.append(current)
    neighbors = edges[current]
    for neighbor in neighbors:
        # if neighbor not in visited, visit
        if neighbor not in visited:
            sort = topological_sort(neighbor, visited, sort)
    # if all neighbors visited add current to sort
    sort.append(current)
    # if all vertices haven't been visited select a new one to visit
    if len(visited) != len(vertices):
        for vertice in vertices:
            if vertice not in visited:
                sort = topological_sort(vertice, visited, sort)
    # return sort
    return sort
