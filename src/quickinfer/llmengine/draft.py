from dataclasses import dataclass
@dataclass
class Draft:
    method:str 
    def gen():
        pass
@dataclass
class N_gram(Draft):
    method: str = "ngram"
    n_min:int=2
    n_max:int=4
    k:int=4
    
    def gen(self, token_ids: list[int]) -> list[int]:
        N = len(token_ids)

        if not (1 <= self.n_min <= self.n_max) or self.k <= 0:
            raise ValueError("Invalid n-gram parameters")

        max_n = min(self.n_max, N - 1)
        if max_n < self.n_min:
            return []

        # 1. 反转 token 序列
        rev = token_ids[::-1]
        pattern = rev[:max_n]

        # 2. 构造 KMP 的 LPS 数组
        lps = [0] * max_n
        j = 0

        for i in range(1, max_n):
            while j > 0 and pattern[i] != pattern[j]:
                j = lps[j - 1]

            if pattern[i] == pattern[j]:
                j += 1

            lps[i] = j

        # 3. 使用 KMP 搜索
        matched = 0
        best_len = 0
        best_end = -1

        for i in range(1, N):
            while matched > 0 and rev[i] != pattern[matched]:
                matched = lps[matched - 1]

            if rev[i] == pattern[matched]:
                matched += 1

            # 记录最长匹配
            if matched >= self.n_min and matched >= best_len:
                best_len = matched
                best_end = i

            # 达到最大匹配长度，利用 LPS 回退
            if matched == max_n:
                matched = lps[max_n - 1]

        if best_len < self.n_min:
            return []

        # 4. 将反转序列的匹配位置映射回原序列
        start = N - 1 - best_end + best_len

        # 5. 复制匹配片段后面的 k 个 token
        return token_ids[start:start + self.k]
    
class Model_Draft(Draft):
    pass
