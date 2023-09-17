from typing import List

def numIslands(M: int, N: int, grid: List[List[int]]) -> int:
    def dfs(i: int, j: int) -> None:
        if i < 0 or i >= M or j < 0 or j >= N or grid[i][j] == 0:
            return
        grid[i][j] = 0
        dfs(i - 1, j)
        dfs(i + 1, j)
        dfs(i, j - 1)
        dfs(i, j + 1)

    count = 0
    for i in range(M):
        for j in range(N):
            if grid[i][j] == 1:
                count += 1
                dfs(i, j)
    return count

def run_test_cases():
    M, N = 3, 3
    matrix = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 1]
    ]
    matrix = [[int(x) for x in row] for row in matrix]
    result = numIslands(M, N, matrix)
    print(f"Test Case 1: {result} (Expected Output: 2)")

    M1, N1 = 3, 4
    matrix1 = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ]
    matrix1 = [[int(x) for x in row] for row in matrix1]
    result1 = numIslands(M1, N1, matrix1)
    print(f"Test Case 2: {result1} (Expected Output: 3)")

    M2, N2 = 3, 4
    matrix2 = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 0]
    ]
    matrix2 = [[int(x) for x in row] for row in matrix2]
    result2 = numIslands(M2, N2, matrix2)
    print(f"Test Case 3: {result2} (Expected Output: 2)")

if __name__ == "__main__":
    run_test_cases()
