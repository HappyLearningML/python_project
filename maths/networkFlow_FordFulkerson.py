#-*- coding:utf-8-*-
'''
计算最大流量的FordFulkerson算法
FordFulkerson算法如下：
Description:
    1从初始流量开始，记为0
    2选择从信号源到接收器的增强路径，并将路径添加到流中
'''
def BFS(graph, s, t, parent):
    # Return True if there is node that has not iterated.
    visited = [False]*len(graph)
    queue=[]
    queue.append(s)
    visited[s] = True
    
    while queue:
        u = queue.pop(0)
        for ind in range(len(graph[u])):
            if visited[ind] == False and graph[u][ind] > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u

    return True if visited[t] else False
     
def FordFulkerson(graph, source, sink):
    # This array is filled by BFS and to store path
    parent = [-1]*(len(graph))
    max_flow = 0 
    while BFS(graph, source, sink, parent) :
        path_flow = float("Inf")
        s = sink

        while(s !=  source):
            # Find the minimum value in select path
            path_flow = min (path_flow, graph[parent[s]][s])
            s = parent[s]

        max_flow +=  path_flow
        v = sink

        while(v !=  source):
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]
    return max_flow


def mincut(graph, source, sink):
    # This array is filled by BFS and to store path
    parent = [-1]*(len(graph))
    max_flow = 0 
    res = []
    temp = [i[:] for i in graph]   # Record orignial cut, copy.
    while BFS(graph, source, sink, parent) :
        path_flow = float("Inf")
        s = sink

        while(s !=  source):
            # Find the minimum value in select path
            path_flow = min (path_flow, graph[parent[s]][s])
            s = parent[s]

        max_flow +=  path_flow
        v = sink
        
        while(v !=  source):
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]

    for i in range(len(graph)):
        for j in range(len(graph[0])):
            if graph[i][j] == 0 and temp[i][j] > 0:
                res.append((i,j))

    return res

if __name__ == "__main__":
    graph = [[0, 16, 13, 0, 0, 0],
         [0, 0, 10 ,12, 0, 0],
         [0, 4, 0, 0, 14, 0],
         [0, 0, 9, 0, 0, 20],
         [0, 0, 0, 7, 0, 4],
         [0, 0, 0, 0, 0, 0]]

    source, sink = 0, 5
    print(FordFulkerson(graph, source, sink))
    print(mincut(graph, source, sink))