import numpy as np
import random
from collections import defaultdict, Counter
import time
import sys

# 焼きなまし法のパラメータ
INITIAL_TEMP = 2000.0
FINAL_TEMP = 0.1
ITERATIONS = 200  # 実行時間を考慮して短縮100000
COOLING_RATE = (FINAL_TEMP / INITIAL_TEMP) ** (1.0 / ITERATIONS)

# 文字列の最大長とシミュレーションパラメータ
MAX_STRING_LEN = 12
SIMULATION_LENGTH = 500  # 実行時間を考慮して短縮10000
SIMULATIONS_PER_SCORE = 1  # 実行時間を考慮して短縮

class LovelyLanguageModel:
    def __init__(self, N, M, L, strings, preferences):
        self.N = N  # 文字列の数
        self.M = M  # 状態数
        self.L = L  # 生成文字列の長さ
        self.strings = strings  # 好きな文字列のリスト
        self.preferences = preferences  # 好ましさのリスト
        self.alphabet = 'abcdef'  # 使用する文字セット
        
        # シミュレーション結果のキャッシュ
        self.cache_key = None
        self.cached_score = None
        
        # 文字列の特性を分析
        self.analyze_strings()
        
        # 初期モデルの構築
        self.initialize_model()
    
    def analyze_strings(self):
        """文字列の分析と優先順位付け - より詳細な分析"""
        # 文字列の重要度スコアを計算
        self.string_importance = []
        for i in range(self.N):
            # 確率の粗い推定と好ましさの積を基にしたスコア
            # 短い文字列は出現確率が高い
            length_factor = 1.0 / (len(self.strings[i]) ** 0.5)
            importance = self.preferences[i] * length_factor
            self.string_importance.append((i, importance, len(self.strings[i])))
        
        # 重要度の降順でソート
        self.string_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 上位の重要な文字列
        self.top_strings_idx = [idx for idx, _, _ in self.string_importance[:min(10, self.N)]]
        
        # 文字の出現頻度を計算（好ましさで重み付け）
        self.char_freq = defaultdict(float)
        for i, s in enumerate(self.strings):
            weight = self.preferences[i] / (len(s) ** 0.5)
            for c in s:
                self.char_freq[c] += weight
        
        # 2-gram, 3-gram (n文字の並び) の頻度を計算（好ましさで重み付け）
        self.bigram_freq = defaultdict(float)
        self.trigram_freq = defaultdict(float)
        
        for i, s in enumerate(self.strings):
            weight = self.preferences[i] / (len(s) ** 0.5)
            # 2-gram
            for j in range(len(s) - 1):
                self.bigram_freq[s[j:j+2]] += weight
            # 3-gram
            for j in range(len(s) - 2):
                self.trigram_freq[s[j:j+3]] += weight
        
        # 共通の部分文字列を特定
        self.common_substrings = self.find_common_substrings()
        
        # 文字列間の関連性を分析
        self.string_relation = self.analyze_string_relations()
    
    def find_common_substrings(self):
        """文字列間で共通する部分文字列を特定"""
        common_subs = []
        # 長さ2以上の共通部分文字列を探す
        for i, s1 in enumerate(self.strings):
            for j in range(i+1, self.N):
                s2 = self.strings[j]
                for length in range(2, min(len(s1), len(s2)) + 1):
                    for pos1 in range(len(s1) - length + 1):
                        substring = s1[pos1:pos1+length]
                        if substring in s2:
                            weight = (self.preferences[i] + self.preferences[j]) / (length ** 0.5)
                            common_subs.append((substring, weight, length))
        
        # 重みでソート
        common_subs.sort(key=lambda x: x[1] * x[2], reverse=True)
        return common_subs[:min(20, len(common_subs))]
    
    def analyze_string_relations(self):
        """文字列間の関連性（オーバーラップなど）を分析"""
        relations = []
        for i in range(self.N):
            s1 = self.strings[i]
            for j in range(i+1, self.N):
                s2 = self.strings[j]
                
                # 末尾と先頭のオーバーラップを見つける
                max_overlap = 0
                for overlap_len in range(1, min(len(s1), len(s2)) + 1):
                    if s1[-overlap_len:] == s2[:overlap_len]:
                        max_overlap = overlap_len
                
                if max_overlap >= 2:  # 意味のあるオーバーラップのみ
                    weight = (self.preferences[i] + self.preferences[j]) * max_overlap
                    relations.append((i, j, max_overlap, weight))
        
        # 重みでソート
        relations.sort(key=lambda x: x[3], reverse=True)
        return relations[:min(30, len(relations))]
    
    def initialize_model(self):
        """モデルの初期化: 状態への文字割り当てと遷移確率の設定"""
        # 状態への文字割り当て - より効果的なアプローチ
        self.state_chars = [''] * self.M
        
        # 1. 最も重要な文字列のパターンに基づいて初期状態を設定
        top_string_idx = self.string_importance[0][0]
        top_string = self.strings[top_string_idx]
        
        # 最初の状態には最重要文字列の最初の文字を割り当て
        self.state_chars[0] = top_string[0]
        
        # 2. 共通部分文字列と高頻度n-gramを考慮
        assigned_states = 1  # 状態0は既に割り当て済み
        
        # 共通部分文字列から重要なものを選んで状態に割り当て
        for substring, _, _ in self.common_substrings[:min(3, len(self.common_substrings))]:
            if assigned_states < self.M:
                self.state_chars[assigned_states] = substring[0]
                assigned_states += 1
        
        # トップの2-gram, 3-gramから割り当て
        top_bigrams = sorted(self.bigram_freq.items(), key=lambda x: x[1], reverse=True)
        for bg, _ in top_bigrams[:min(4, len(top_bigrams))]:
            if assigned_states < self.M:
                self.state_chars[assigned_states] = bg[0]
                assigned_states += 1
        
        # 3. 残りの状態には、最重要文字列のパターンや高頻度文字を割り当て
        while assigned_states < self.M:
            # 重み付けされた文字頻度に基づいて割り当て
            char_weights = {c: freq for c, freq in self.char_freq.items()}
            chars = list(char_weights.keys())
            weights = list(char_weights.values())
            
            if not weights:
                # 万が一重みがなければランダムに割り当て
                self.state_chars[assigned_states] = random.choice(self.alphabet)
            else:
                self.state_chars[assigned_states] = random.choices(chars, weights=weights, k=1)[0]
            
            assigned_states += 1
        
        # 4. トップ文字列がよく生成されるように調整
        if len(top_string) <= self.M:
            # トップ文字列のパターンを直接状態に反映
            for i in range(min(len(top_string), self.M)):
                self.state_chars[i] = top_string[i]
        
        # 遷移確率行列の初期化 - スマートな初期化
        self.initialize_transition_matrix()
    ################################################################################3
    #################################################################################
    def initialize_transition_matrix(self):
        """遷移確率行列をより効果的に初期化"""
        self.transition_probs = np.zeros((self.M, self.M), dtype=int)
        
        # 最も重要な文字列に基づいて遷移確率を設定
        top_string_idx = self.string_importance[0][0]
        top_string = self.strings[top_string_idx]
        
        # 基本的な遷移確率設定
        for i in range(self.M):
            # まず均等に配分
            base_prob = 100 // self.M
            remaining = 100
            
            for j in range(self.M):
                if i == j:  # 自己遷移に少し高い確率
                    self.transition_probs[i][j] = min(remaining, base_prob + 5)
                else:
                    self.transition_probs[i][j] = min(remaining, base_prob)
                remaining -= self.transition_probs[i][j]
            
            # 残りを最後の状態に割り当て
            if remaining > 0:
                self.transition_probs[i][self.M - 1] += remaining
        
        # 2-gramパターンに基づいて遷移確率を調整
        self.adjust_transitions_for_patterns()
    
    def adjust_transitions_for_patterns(self):
        """n-gramパターンに基づいて遷移確率を調整"""
        # 上位10個の重要な文字列からパターンを抽出
        for idx in self.top_strings_idx:
            s = self.strings[idx]
            
            for i in range(len(s) - 1):
                char1, char2 = s[i], s[i+1]
                
                # char1を含む状態から、char2を含む状態への遷移確率を高める
                for state1 in range(self.M):
                    if self.state_chars[state1] == char1:
                        for state2 in range(self.M):
                            if self.state_chars[state2] == char2:
                                # この遷移の確率を増加（しすぎないように注意）
                                boost = min(20, 100 - sum(self.transition_probs[state1]))
                                if boost > 0:
                                    # 他の遷移から確率を取り除く
                                    for j in range(self.M):
                                        if j != state2 and self.transition_probs[state1][j] > 0:
                                            reduce = min(self.transition_probs[state1][j], boost)
                                            self.transition_probs[state1][j] -= reduce
                                            boost -= reduce
                                            if boost <= 0:
                                                break
                                    
                                    # 取り除いた確率を目的の遷移に加える
                                    self.transition_probs[state1][state2] += (20 - boost)
        
        # 全ての行の合計が100になるよう最終調整
        for i in range(self.M):
            total = sum(self.transition_probs[i])
            if total != 100:
                # 合計が100でない場合は調整
                diff = 100 - total
                
                if diff > 0:
                    # 不足している場合：一番大きい遷移に追加
                    max_idx = np.argmax(self.transition_probs[i])
                    self.transition_probs[i][max_idx] += diff
                else:
                    # 多すぎる場合：一番大きい遷移から減らす
                    max_idx = np.argmax(self.transition_probs[i])
                    self.transition_probs[i][max_idx] += diff  # diffは負なので減算される
    
    def simulate(self, num_steps=SIMULATION_LENGTH):
        """モデルをシミュレーションして文字列を生成し、各文字列の出現確率を推定"""
        # 初期状態は0
        current_state = 0
        # 最初に割り当てられた文字を出力
        generated = [self.state_chars[current_state]]
        
        # 文字列生成
        for _ in range(num_steps - 1):
            # 次の状態を確率に基づいて選択
            next_probs = self.transition_probs[current_state]
            next_state = random.choices(range(self.M), weights=next_probs, k=1)[0]
            generated.append(self.state_chars[next_state])
            current_state = next_state
        
        # 生成された文字列
        generated_str = ''.join(generated)
        
        # 各文字列の出現をカウント - Knuth-Morris-Prattアルゴリズムで効率的に
        appearances = {}
        for i, s in enumerate(self.strings):
            # 部分文字列検索を効率的に行う
            appears = self.kmp_search(generated_str, s)
            appearances[i] = 1 if appears else 0
        
        return appearances
    
    def kmp_search(self, text, pattern):
        """Knuth-Morris-Prattアルゴリズムによる効率的な文字列検索"""
        if not pattern:
            return True
            
        # 失敗関数の計算
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                else:
                    lps[i] = 0
                    i += 1
        
        # テキスト内でのパターン検索
        i = 0  # テキストのインデックス
        j = 0  # パターンのインデックス
        
        while i < len(text):
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == len(pattern):
                return True  # パターンが見つかった
            elif i < len(text) and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
        
        return False  # パターンが見つからなかった

    def calculate_score(self, simulations=SIMULATIONS_PER_SCORE):
        """スコアを計算: Σ(P_i * Q_i)"""
        # 現在の設定のハッシュをキーとしてキャッシュを使用
        current_key = (tuple(self.state_chars), tuple(map(tuple, self.transition_probs)))
        if self.cache_key == current_key:
            return self.cached_score
        
        total_appearances = defaultdict(int)
        
        # 複数回シミュレーションして平均を取る
        for _ in range(simulations):
            appearances = self.simulate()
            for i, appears in appearances.items():
                total_appearances[i] += appears
        
        # 各文字列の出現確率を推定
        estimated_probs = {i: total_appearances[i] / simulations for i in range(self.N)}
        
        # スコア計算
        score = sum(self.preferences[i] * estimated_probs[i] for i in range(self.N))
        rounded_score = round(score)
        
        # キャッシュに保存
        self.cache_key = current_key
        self.cached_score = rounded_score
        
        return rounded_score
    
    def modify_transitions(self, temp):
        """遷移確率を微調整 (焼きなまし法用) - より効果的な修正戦略"""
        # ランダムに状態を選択
        state_i = random.randint(0, self.M - 1)
        
        # 変更前の確率を保存
        old_probs = self.transition_probs[state_i].copy()
        
        # 変更方法の種類をランダムに選択
        method = random.randint(0, 3)
        
        if method == 0:
            # 方法1: ランダムに2つの遷移先を選び、確率を調整
            indices = random.sample(range(self.M), 2)
            j1, j2 = indices
            
            # 調整量 (温度に比例、最小1)
            amount = max(1, int(temp * 0.05))
            
            # j1からj2へ確率を移す (j1の確率が十分あれば)
            if self.transition_probs[state_i][j1] >= amount:
                self.transition_probs[state_i][j1] -= amount
                self.transition_probs[state_i][j2] += amount
                return state_i, old_probs
        
        elif method == 1:
            # 方法2: 最も確率の高い遷移から、弱い遷移へ少し確率を移す
            max_idx = np.argmax(self.transition_probs[state_i])
            
            # 確率の低い遷移をランダムに選択
            candidates = [j for j in range(self.M) if self.transition_probs[state_i][j] < 10]
            if candidates and self.transition_probs[state_i][max_idx] >= 5:
                j = random.choice(candidates)
                
                # 調整量 (小さく)
                amount = random.randint(1, 5)
                
                self.transition_probs[state_i][max_idx] -= amount
                self.transition_probs[state_i][j] += amount
                return state_i, old_probs
        
        elif method == 2:
            # 方法3: 複数の確率を少しずつ調整
            # 確率を減らす候補（10%以上の確率がある遷移）
            decrease_candidates = [j for j in range(self.M) if self.transition_probs[state_i][j] >= 10]
            # 確率を増やす候補
            increase_candidates = [j for j in range(self.M) if j not in decrease_candidates]
            
            if decrease_candidates and increase_candidates:
                # 減らす側と増やす側をランダムに複数選ぶ
                to_decrease = random.sample(decrease_candidates, min(2, len(decrease_candidates)))
                to_increase = random.sample(increase_candidates, min(2, len(increase_candidates)))
                
                amount = random.randint(2, 5)  # 移行する確率の総量
                
                # 確率の移行
                for j in to_decrease:
                    # 各減少候補から均等に減らす
                    dec_amount = min(amount // len(to_decrease), self.transition_probs[state_i][j] - 1)
                    self.transition_probs[state_i][j] -= dec_amount
                    amount -= dec_amount
                
                # 増加側に配分
                for j in to_increase:
                    inc_amount = amount // len(to_increase)
                    self.transition_probs[state_i][j] += inc_amount
                    amount -= inc_amount
                
                # 残りがあれば最初の増加候補に加算
                if amount > 0 and to_increase:
                    self.transition_probs[state_i][to_increase[0]] += amount
                
                return state_i, old_probs
        
        else:
            # 方法4: 特定のパターンを強化
            # 上位の重要な文字列からランダムに1つ選び、そのパターンを強化
            if self.top_strings_idx:
                idx = random.choice(self.top_strings_idx)
                s = self.strings[idx]
                
                if len(s) >= 2:
                    pos = random.randint(0, len(s) - 2)
                    char1, char2 = s[pos], s[pos+1]
                    
                    # char1を含む状態とchar2を含む状態を探す
                    states1 = [st for st in range(self.M) if self.state_chars[st] == char1]
                    states2 = [st for st in range(self.M) if self.state_chars[st] == char2]
                    
                    if states1 and states2:
                        state1 = random.choice(states1)
                        state2 = random.choice(states2)
                        
                        if state1 != state_i:  # 選んだ状態と一致しない場合はスキップ
                            state_i = state1
                            old_probs = self.transition_probs[state_i].copy()
                        
                        # すでに高確率（50%以上）の場合は強化しない
                        if self.transition_probs[state_i][state2] < 50:
                            # 調整量（温度に応じて変化）
                            amount = max(1, min(10, int(temp * 0.03)))
                            
                            # 他の遷移から確率を取り除く
                            removed = 0
                            for j in range(self.M):
                                if j != state2 and self.transition_probs[state_i][j] > 0:
                                    reduce = min(amount - removed, self.transition_probs[state_i][j] - 1)
                                    if reduce > 0:
                                        self.transition_probs[state_i][j] -= reduce
                                        removed += reduce
                                    
                                    if removed >= amount:
                                        break
                            
                            # 取り除いた確率を目的の遷移に加える
                            self.transition_probs[state_i][state2] += removed
                            return state_i, old_probs
        
        # 変更できなかった場合
        return None, None

    def modify_state_chars(self):
        """状態への文字割り当てを微調整 - 戦略的アプローチ"""
        # 方法の種類をランダムに選択
        method = random.randint(0, 2)
        
        if method == 0:
            # 方法1: ランダムに状態を選択して文字を変更
            state_i = random.randint(0, self.M - 1)
            old_char = self.state_chars[state_i]
            
            # 新しい文字を文字頻度に基づいて選択
            char_items = list(self.char_freq.items())
            if char_items:
                chars, freqs = zip(*char_items)
                new_char = random.choices(chars, weights=freqs, k=1)[0]
            else:
                new_char = random.choice(self.alphabet)
            
            # 文字を変更
            self.state_chars[state_i] = new_char
            return state_i, old_char
        
        elif method == 1:
            # 方法2: 重要な文字列のパターンを反映
            if self.top_strings_idx:
                idx = random.choice(self.top_strings_idx)
                s = self.strings[idx]
                
                if len(s) > 0:
                    pos = random.randint(0, len(s) - 1)
                    state_i = random.randint(0, self.M - 1)
                    old_char = self.state_chars[state_i]
                    
                    # 文字列の特定位置の文字を割り当て
                    self.state_chars[state_i] = s[pos]
                    return state_i, old_char
        
        else:
            # 方法3: 共通部分文字列のパターンを強化
            if self.common_substrings:
                substring, _, _ = random.choice(self.common_substrings)
                
                if len(substring) > 0:
                    pos = random.randint(0, len(substring) - 1)
                    state_i = random.randint(0, self.M - 1)
                    old_char = self.state_chars[state_i]
                    
                    # 共通部分文字列の文字を割り当て
                    self.state_chars[state_i] = substring[pos]
                    return state_i, old_char
        
        # 変更できなかった場合、ランダムに状態を選んで文字を変更
        state_i = random.randint(0, self.M - 1)
        old_char = self.state_chars[state_i]
        new_char = random.choice(self.alphabet)
        self.state_chars[state_i] = new_char
        return state_i, old_char
    
    def optimize_simulated_annealing(self):
        """焼きなまし法による最適化 - より効果的な実装"""
        temp = INITIAL_TEMP
        current_score = self.calculate_score()
        best_score = current_score
        best_solution = (self.state_chars.copy(), self.transition_probs.copy())
        
        print(f"Initial score: {current_score}", file=sys.stderr)
        
        # 一定期間スコアが改善しなかった回数
        stagnation_count = 0
        # 改善がなくても続けるしきい値
        max_stagnation = ITERATIONS // 10
        
        last_improvement = 0  # 最後に改善があった反復回数
        
        for iteration in range(ITERATIONS):
            # 温度の更新
            temp *= COOLING_RATE
            
            # 95%の確率で遷移確率を変更、5%の確率で文字割り当てを変更
            if random.random() < 0.95:
                # 遷移確率の微調整
                state_i, old_probs = self.modify_transitions(temp)
                if old_probs is None:
                    continue  # 変更できなかった場合はスキップ
                
                # スコア計算
                new_score = self.calculate_score()
                
                # 変更の採用判定
                if new_score > current_score or random.random() < np.exp((new_score - current_score) / temp):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_solution = (self.state_chars.copy(), self.transition_probs.copy())
                        print(f"Iteration {iteration}, New best score: {best_score}", file=sys.stderr)
                        last_improvement = iteration
                        stagnation_count = 0
                    elif iteration - last_improvement > max_stagnation:
                        stagnation_count += 1
                else:
                    # 変更を元に戻す
                    self.transition_probs[state_i] = old_probs
            else:
                # 文字割り当ての微調整
                state_i, old_char = self.modify_state_chars()
                
                # スコア計算
                new_score = self.calculate_score()
                
                # 変更の採用判定
                if new_score > current_score or random.random() < np.exp((new_score - current_score) / temp):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_solution = (self.state_chars.copy(), self.transition_probs.copy())
                        print(f"Iteration {iteration}, New best score: {best_score} (char change)", file=sys.stderr)
                        last_improvement = iteration
                        stagnation_count = 0
                    elif iteration - last_improvement > max_stagnation:
                        stagnation_count += 1
                else:
                    # 変更を元に戻す
                    self.state_chars[state_i] = old_char
            
            # 停滞が長すぎる場合にリスタート
            if stagnation_count >= 5:
                print(f"Restarting at iteration {iteration} due to stagnation", file=sys.stderr)
                # 現在の最良解を維持しつつ、一部をランダム化
                self.state_chars = best_solution[0].copy()
                self.transition_probs = best_solution[1].copy()
                
                # 30%の状態をランダムに文字再割り当て
                states_to_change = random.sample(range(self.M), max(1, self.M // 3))
                for state_i in states_to_change:
                    self.state_chars[state_i] = random.choice(self.alphabet)
                
                # 30%の遷移確率行をランダム化
                rows_to_change = random.sample(range(self.M), max(1, self.M // 3))
                for row in rows_to_change:
                    # 均等分布からスタート
                    self.transition_probs[row] = np.zeros(self.M, dtype=int)
                    #####################################################################3
                    ########################################################################3
                    # 均等分布からスタート
                    self.transition_probs[row] = np.zeros(self.M, dtype=int)
                    remaining = 100
                    for j in range(self.M - 1):
                        prob = random.randint(0, min(20, remaining))
                        self.transition_probs[row][j] = prob
                        remaining -= prob
                    self.transition_probs[row][self.M - 1] = remaining
                
                # リスタート後は温度を少し上げる
                temp = min(INITIAL_TEMP, temp * 10)
                stagnation_count = 0
                
                # 新しいスコアを計算
                current_score = self.calculate_score()
            
            # 進捗表示（5%ごと）
            if iteration % (ITERATIONS // 20) == 0:
                print(f"Iteration {iteration}, Temp: {temp:.2f}, Current score: {current_score}, Best: {best_score}", file=sys.stderr)
        
        # 最良解を復元
        self.state_chars, self.transition_probs = best_solution
        print(f"Final best score: {best_score}", file=sys.stderr)
    
    # def get_solution(self):
    #     """解をフォーマットして返す"""
    #     result = []
    #     for i in range(self.M):
    #         row = [self.state_chars[i]] + list(map(str, self.transition_probs[i]))
    #         result.append(" ".join(row))
    #     return result
    def get_solution(self):
        """解をフォーマットして返す（遷移確率の合計を確認）"""
        # 遷移確率の合計を確認して修正
        for i in range(self.M):
            total = sum(self.transition_probs[i])
            if total != 100:
                # 合計が100でない場合は調整
                diff = 100 - total
                if diff > 0:
                    # 不足している場合：一番大きい遷移に追加
                    max_idx = np.argmax(self.transition_probs[i])
                    self.transition_probs[i][max_idx] += diff
                else:
                    # 多すぎる場合：一番大きい遷移から減らす
                    max_idx = np.argmax(self.transition_probs[i])
                    self.transition_probs[i][max_idx] += diff  # diffは負なので減算される
        
        # 解をフォーマット
        result = []
        for i in range(self.M):
            row = [self.state_chars[i]] + list(map(str, self.transition_probs[i]))
            result.append(" ".join(row))
        return result


def main():
    # 入力読み込み
    N, M, L = map(int, input().strip().split())
    strings = []
    preferences = []
    
    for _ in range(N):
        s, p = input().strip().split()
        strings.append(s)
        preferences.append(int(p))
    
    # LLMモデルの初期化と最適化
    start_time = time.time()
    
    # 並列実行のための設定 - 複数の初期解を試す（オプション）
    num_trials = 1  # 並列実行するモデル数（時間制約に応じて調整）
    best_score = 0
    best_solution = None
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}", file=sys.stderr)
        
        # シード値を変えて異なる初期解から開始
        random.seed(int(time.time()) + trial)
        
        # モデルの初期化
        llm = LovelyLanguageModel(N, M, L, strings, preferences)
        
        # 最適化実行
        llm.optimize_simulated_annealing()
        
        # スコア計算
        score = llm.calculate_score(simulations=10)  # より正確なスコア計算
        
        print(f"Trial {trial + 1} score: {score}", file=sys.stderr)
        
        # より良いスコアが得られたら更新
        if score > best_score:
            best_score = score
            best_solution = llm.get_solution()
    
    # 最良解を出力
    if best_solution:
        for line in best_solution:
            print(line)
    else:
        # 念のため、何らかの理由で解が得られなかった場合のフォールバック
        llm = LovelyLanguageModel(N, M, L, strings, preferences)
        for line in llm.get_solution():
            print(line)
    
    #print(f"Total time: {time.time() - start_time:.2f} seconds", file=sys.stderr)


if __name__ == "__main__":
    main()
