from collections import deque

def func(a):
    n = len(a)

    adj = [set() for _ in range(n)]

    for i in range(n):
        sx, sy, ex, ey = a[i]
        for j in range(n):
            if j == i:
                continue
            sjx, sjy, ejx, ejy = a[j]
            if sjy > ey:
                adj[i].add(j)
            if sx > ejx and sy < ejy:
                adj[i].add(j)

    indegree = [0] * n
    for i in range(n):
        for j in adj[i]:
            indegree[j] += 1

    q = deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)

    result = []
    while q:
        node = q.popleft()
        result.append(node)

        for it in adj[node]:
            indegree[it] -= 1
            if indegree[it] == 0:
                q.append(it)
    return result