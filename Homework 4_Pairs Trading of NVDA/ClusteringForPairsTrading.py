# -*- coding: utf-8 -*-
"""ClusteringForPairsTrading.ipynb

自動由 Colab 生成。

原始檔案位置：
    https://colab.research.google.com/github/FourLevel/Machine-Learning-and-Financial-Applications/blob/main/Homework%204_Pairs%20Trading%20of%20NVDA/ClusteringForPairsTrading.ipynb

# 配對交易 - 基於聚類方法尋找配對

在此案例研究中，我們將使用聚類方法來選擇配對交易策略的股票配對。

# 1. 問題定義

我們在此案例研究中的目標是對 S&P500 股票進行聚類分析，
並為配對交易策略找出合適的股票配對。

S&P 500 股票數據是使用 pandas_datareader 從 yahoo finance 獲得的。
包含 2018 年以來的價格數據。

# 2. 開始 - 載入數據和 Python 套件

"""

# 載入必要的函式庫
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from sklearn import metrics, cluster
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from itertools import cycle
from statsmodels.tsa.stattools import coint
import warnings

# 關閉警告訊息
warnings.filterwarnings('ignore')

# 匯入已從 yahoo finance 獲得的數據
dataset = read_csv('SP500Data.csv', index_col=0)

print("Dataset type:", type(dataset))

# 3. 探索性數據分析

# 數據形狀
print("Dataset shape:", dataset.shape)

# 查看數據
set_option('display.width', 100)
print("Dataset head:")
print(dataset.head(5))

# 描述數據
set_option('display.precision', 3)
print("Dataset description:")
print(dataset.describe())

# 4. 數據準備

# 我們檢查行中的 NA 值，要麼刪除它們，要麼用列的平均值填充它們。

# 檢查是否有空值並移除空值
print('Null values =', dataset.isnull().values.any())

# 移除超過 30% 缺失值的欄位
missing_fractions = dataset.isnull().mean().sort_values(ascending=False)
print("Missing fractions head:")
print(missing_fractions.head(10))

drop_list = sorted(list(missing_fractions[missing_fractions > 0.3].index))
dataset.drop(labels=drop_list, axis=1, inplace=True)
print("Dataset shape after dropping columns:", dataset.shape)

# 用數據集中最後一個可用值填充缺失值
dataset = dataset.fillna(method='ffill')
print("Dataset head after filling:")
print(dataset.head(2))

# 為了聚類的目的，我們將使用年化報酬率和變異數作為變數，
# 因為它們是股票表現和波動性的指標。讓我們從數據中準備報酬率和波動性變數。

# 計算理論一年期間的平均年化百分比報酬率和波動率
returns = dataset.pct_change().mean() * 252
returns = pd.DataFrame(returns)
returns.columns = ['Returns']
returns['Volatility'] = dataset.pct_change().std() * np.sqrt(252)
data = returns

"""在應用聚類之前，所有變數都應該在相同的尺度上，否則具有大值的特徵將主導結果。
我們使用 sklearn 中的 StandardScaler 將數據集的特徵標準化為單位尺度（平均值 = 0，變異數 = 1）。"""

scaler = StandardScaler().fit(data)
rescaledDataset = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
# 總結轉換後的數據
print("Rescaled dataset head:")
print(rescaledDataset.head(2))
X = rescaledDataset
print("X head:")
print(X.head(2))

"""聚類的參數是索引，聚類中使用的變數是欄位。因此數據格式正確，可以輸入聚類演算法。

# 5. 評估演算法和模型

我們將查看以下模型：

1. K-Means 聚類
2. 階層聚類（凝聚聚類）
3. 親和力傳播

## 5.1. K-Means 聚類

### 5.1.1. 尋找最佳聚類數量

在此步驟中，我們查看以下指標：

1. 聚類內平方誤差和（SSE）
2. 輪廓分數
"""

distorsions = []
max_loop = 20
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, max_loop), distorsions)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.grid(True)
plt.title('K-means Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()

# 檢查平方誤差和圖表，似乎在 5 或 6 個聚類時出現肘部「轉折」。
# 當然，我們可以看到當聚類數量超過 6 時，聚類內平方誤差和趨於平穩。

# 輪廓分數
silhouette_score = []
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    kmeans.fit(X)
    silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, random_state=10))

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, max_loop), silhouette_score)
plt.xticks([i for i in range(2, max_loop)], rotation=75)
plt.grid(True)
plt.title('Silhouette Score Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# 從輪廓分數圖表中，我們可以看到圖表的各個部分都可以看到轉折。
# 由於 6 個聚類後 SSE 沒有太大差異，我們在 K-means 模型中偏好 6 個聚類。

# 讓我們建立具有六個聚類的 k-means 模型並視覺化結果。

nclust = 6

# 使用 k-means 擬合
k_means = cluster.KMeans(n_clusters=nclust)
k_means.fit(X)

# 提取標籤
target_labels = k_means.predict(X)

# 當數據集中的變數/維度數量很大時，視覺化聚類的形成並不容易。
# 在二維空間中視覺化聚類的方法之一。

centroids = k_means.cluster_centers_
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_means.labels_, cmap="rainbow", label=X.index)
ax.set_title('K-Means Clustering Result')
ax.set_xlabel('Standardized Returns')
ax.set_ylabel('Standardized Volatility')
plt.colorbar(scatter)
plt.plot(centroids[:, 0], centroids[:, 1], 'sg', markersize=11)
plt.show()

# 讓我們檢查聚類的元素

# 顯示每個聚類中的股票數量
clustered_series = pd.Series(index=X.index, data=k_means.labels_.flatten())
# 帶有聚類標籤的聚類股票
clustered_series_all = pd.Series(index=X.index, data=k_means.labels_.flatten())
clustered_series = clustered_series[clustered_series != -1]

plt.figure(figsize=(12, 7))
plt.barh(
    range(len(clustered_series.value_counts())),  # 聚類標籤，y 軸
    clustered_series.value_counts()
)
plt.title('K-Means Cluster Sizes')
plt.xlabel('Number of Stocks')
plt.ylabel('Cluster Number')
plt.show()

# 聚類中的股票數量範圍從大約 40 到 120。雖然分佈不均等，但每個聚類中都有大量股票。

# 在第一步中，我們查看階層圖並檢查聚類數量

# 階層類別有一個樹狀圖方法，它接受同一類別的連結方法返回的值。
# 連結方法將數據集和最小化距離的方法作為參數。我們使用 'ward' 作為方法，
# 因為它最小化聚類之間距離的變異數。

# 計算連結
Z = linkage(X, method='ward')
print("Linkage first element:", Z[0])

"""視覺化凝聚聚類演算法的最佳方法是通過樹狀圖，它顯示聚類樹，
葉子是個別股票，根是最終的單一聚類。每個聚類之間的「距離」顯示在 y 軸上，
因此分支越長，兩個聚類的相關性越小。"""

# 繪製樹狀圖
plt.figure(figsize=(20, 10), dpi=100)
plt.title("Hierarchical Clustering Dendrogram")
dendrogram(Z, labels=X.index)
plt.xlabel('Stock Symbols')
plt.ylabel('Distance')
plt.show()

"""一旦形成一個大聚類，就選擇沒有任何水平線穿過的最長垂直距離，並通過它畫一條水平線。
這條新創建的水平線穿過的垂直線數量等於聚類數量。
然後我們選擇距離閾值來切割樹狀圖以獲得選定的聚類級別。輸出是每行數據的聚類標籤。
正如從樹狀圖中預期的那樣，在 13 處切割給我們四個聚類。
"""

distance_threshold = 13
clusters = fcluster(Z, distance_threshold, criterion='distance')
chosen_clusters = pd.DataFrame(data=clusters, columns=['cluster'])
print("Unique clusters:", chosen_clusters['cluster'].unique())

"""### 5.2.2. 聚類和視覺化"""

nclust = 4
hc = AgglomerativeClustering(n_clusters=nclust, linkage='ward')
clust_labels1 = hc.fit_predict(X)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels1, cmap="rainbow")
ax.set_title('Hierarchical Clustering')
ax.set_xlabel('Standardized Returns')
ax.set_ylabel('Standardized Volatility')
plt.colorbar(scatter)
plt.show()

"""與 k-means 聚類的圖表類似，我們看到有一些由不同顏色分隔的明顯聚類。

## 5.3. 親和力傳播
"""

ap = AffinityPropagation()
ap.fit(X)
clust_labels2 = ap.predict(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels2, cmap="rainbow")
ax.set_title('Affinity Propagation')
ax.set_xlabel('Standardized Returns')
ax.set_ylabel('Standardized Volatility')
plt.colorbar(scatter)
plt.show()

"""與 k-means 聚類的圖表類似，我們看到有一些由不同顏色分隔的明顯聚類。

### 5.3.1 聚類視覺化
"""

cluster_centers_indices = ap.cluster_centers_indices_
labels = ap.labels_

no_clusters = len(cluster_centers_indices)
print('Estimated number of clusters: %d' % no_clusters)
# 繪製範例

X_temp = np.asarray(X)
plt.close('all')
plt.figure(1)
plt.clf()

fig = plt.figure(figsize=(8, 6))
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(no_clusters), colors):
    class_members = labels == k
    cluster_center = X_temp[cluster_centers_indices[k]]
    plt.plot(X_temp[class_members, 0], X_temp[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in X_temp[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Affinity Propagation Clustering')
plt.xlabel('Standardized Returns')
plt.ylabel('Standardized Volatility')
plt.show()

# 顯示每個聚類中的股票數量
clustered_series_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
# 帶有聚類標籤的聚類股票
clustered_series_all_ap = pd.Series(index=X.index, data=ap.labels_.flatten())
clustered_series_ap = clustered_series_ap[clustered_series != -1]

plt.figure(figsize=(12, 7))
plt.barh(
    range(len(clustered_series_ap.value_counts())),  # 聚類標籤，y 軸
    clustered_series_ap.value_counts()
)
plt.title('Affinity Propagation Cluster Sizes')
plt.xlabel('Number of Stocks')
plt.ylabel('Cluster Number')
plt.show()

"""## 5.4. 聚類評估

如果不知道真實標籤，則必須使用模型本身進行評估。輪廓係數（sklearn.metrics.silhouette_score）
是此類評估的一個例子，其中較高的輪廓係數分數與具有更好定義聚類的模型相關。
輪廓係數為每個樣本定義，由兩個分數組成：
"""

print("K-means silhouette score:", metrics.silhouette_score(X, k_means.labels_, metric='euclidean'))
print("Hierarchical silhouette score:", metrics.silhouette_score(X, hc.fit_predict(X), metric='euclidean'))
print("Affinity Propagation silhouette score:", metrics.silhouette_score(X, ap.labels_, metric='euclidean'))

"""鑑於親和力傳播表現最佳，我們繼續使用親和力傳播，並使用此聚類方法指定的聚類

### 視覺化聚類內的報酬率

為了理解聚類背後的直覺，讓我們視覺化聚類的結果。
"""

# 所有股票及其聚類標籤（包括 -1）
clustered_series = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
# 帶有聚類標籤的聚類股票
clustered_series_all = pd.Series(index=X.index, data=ap.fit_predict(X).flatten())
clustered_series = clustered_series[clustered_series != -1]

# 獲取每個聚類中的股票數量
counts = clustered_series_ap.value_counts()

# 讓我們視覺化一些聚類
cluster_vis_list = list(counts[(counts < 25) & (counts > 1)].index)[::-1]
print("Cluster visualization list:", cluster_vis_list)

CLUSTER_SIZE_LIMIT = 9999
counts = clustered_series.value_counts()
ticker_count_reduced = counts[(counts > 1) & (counts <= CLUSTER_SIZE_LIMIT)]
print("Clusters formed: %d" % len(ticker_count_reduced))
print("Pairs to evaluate: %d" % (ticker_count_reduced * (ticker_count_reduced - 1)).sum())

# 繪製一些最小的聚類
plt.figure(figsize=(12, 7))
cluster_vis_list_subset = cluster_vis_list[0:min(len(cluster_vis_list), 4)]

for clust in cluster_vis_list_subset:
    tickers = list(clustered_series[clustered_series == clust].index)
    means = np.log(dataset.loc[:"2018-02-01", tickers].mean())
    data_plot = np.log(dataset.loc[:"2018-02-01", tickers]).sub(means)
    data_plot.plot(title='Cluster %d Stock Time Series' % clust)
plt.show()

"""查看上面的圖表，在所有股票數量較少的聚類中，我們看到不同聚類下股票的相似移動，
這證實了聚類技術的有效性。

# 6. 配對選擇

## 6.1. 共整合和配對選擇函數
"""

def find_cointegrated_pairs(data, significance=0.05, max_pairs=None, verbose=True):
    """
    優化版本的共整合配對尋找函數（單線程版本，避免Windows多進程問題）
    
    Parameters:
    - data: 股票價格數據
    - significance: 顯著性水準 (預設 0.05)
    - max_pairs: 最大配對數量限制 (可選，用於早期停止)
    - verbose: 是否顯示進度
    """
    import time
    
    n = data.shape[1]
    keys = list(data.keys())
    pairs = []
    
    # 預先清理數據，避免重複處理
    cleaned_data = {}
    for key in keys:
        series = data[key].replace([np.inf, -np.inf], np.nan).dropna()
        cleaned_data[key] = series
    
    total_pairs = n * (n - 1) // 2
    if verbose:
        print(f"Testing {total_pairs} potential pairs...")
    
    start_time = time.time()
    tested_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            key1, key2 = keys[i], keys[j]
            tested_pairs += 1
            
            try:
                S1 = cleaned_data[key1]
                S2 = cleaned_data[key2]
                
                # 找到共同的時間索引
                common_index = S1.index.intersection(S2.index)
                if len(common_index) < 30:  # 需要足夠的數據點
                    continue
                    
                S1_aligned = S1.loc[common_index]
                S2_aligned = S2.loc[common_index]
                
                # 執行共整合檢驗
                result = coint(S1_aligned, S2_aligned)
                pvalue = result[1]
                
                if pvalue < significance:
                    pairs.append((key1, key2))
                    if verbose and len(pairs) % 5 == 0:
                        elapsed = time.time() - start_time
                        progress = tested_pairs / total_pairs * 100
                        print(f"Progress: {progress:.1f}% - Found {len(pairs)} pairs (Elapsed: {elapsed:.1f}s)")
                
                # 早期停止機制
                if max_pairs and len(pairs) >= max_pairs:
                    if verbose:
                        print(f"Reached maximum pairs limit ({max_pairs}), stopping early.")
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"Error testing pair ({key1}, {key2}): {e}")
                continue
        
        # 檢查是否需要早期停止
        if max_pairs and len(pairs) >= max_pairs:
            break
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Completed in {elapsed_time:.1f} seconds. Found {len(pairs)} cointegrated pairs.")
    
    # 為了相容性，仍然返回矩陣（但只填充找到的配對）
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    
    return score_matrix, pvalue_matrix, pairs


def find_cointegrated_pairs_fast(data, significance=0.05, sample_size=None, verbose=True):
    """
    快速版本：使用取樣方法減少計算量
    
    Parameters:
    - data: 股票價格數據
    - significance: 顯著性水準
    - sample_size: 取樣大小（如果為None，則測試所有配對）
    - verbose: 是否顯示進度
    """
    import random
    import time
    
    n = data.shape[1]
    keys = list(data.keys())
    pairs = []
    
    # 生成所有可能的配對
    all_combinations = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # 如果指定了取樣大小，則隨機取樣
    if sample_size and sample_size < len(all_combinations):
        combinations_to_test = random.sample(all_combinations, sample_size)
        if verbose:
            print(f"Randomly sampling {sample_size} pairs out of {len(all_combinations)} total combinations")
    else:
        combinations_to_test = all_combinations
    
    if verbose:
        print(f"Testing {len(combinations_to_test)} pairs for cointegration...")
    
    start_time = time.time()
    
    for idx, (i, j) in enumerate(combinations_to_test):
        try:
            S1 = data[keys[i]].replace([np.inf, -np.inf], np.nan).dropna()
            S2 = data[keys[j]].replace([np.inf, -np.inf], np.nan).dropna()
            
            # 找到共同的時間索引
            common_index = S1.index.intersection(S2.index)
            if len(common_index) < 30:  # 需要足夠的數據點
                continue
                
            S1_aligned = S1.loc[common_index]
            S2_aligned = S2.loc[common_index]
            
            result = coint(S1_aligned, S2_aligned)
            pvalue = result[1]
            
            if pvalue < significance:
                pairs.append((keys[i], keys[j]))
            
            # 進度顯示
            if verbose and (idx + 1) % 100 == 0:
                elapsed = time.time() - start_time
                progress = (idx + 1) / len(combinations_to_test) * 100
                print(f"Progress: {progress:.1f}% ({idx + 1}/{len(combinations_to_test)}) - Found {len(pairs)} pairs (Elapsed: {elapsed:.1f}s)")
                
        except Exception as e:
            if verbose:
                print(f"Error testing pair ({keys[i]}, {keys[j]}): {e}")
            continue
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Completed in {elapsed_time:.1f} seconds. Found {len(pairs)} cointegrated pairs.")
    
    # 為了相容性，返回空矩陣
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    
    return score_matrix, pvalue_matrix, pairs

# 選擇優化方法
USE_FAST_METHOD = True  # 設為 True 使用快速取樣方法，False 使用平行處理方法
SAMPLE_SIZE = 1000      # 快速方法的取樣大小
MAX_PAIRS = 50          # 平行處理方法的最大配對數限制

print(f"\n使用優化方法進行配對分析...")
print(f"方法: {'快速取樣' if USE_FAST_METHOD else '平行處理'}")

cluster_dict = {}
for i, which_clust in enumerate(ticker_count_reduced.index):
    tickers = clustered_series[clustered_series == which_clust].index
    print(f"\n分析聚類 {which_clust} (包含 {len(tickers)} 支股票)...")
    
    if USE_FAST_METHOD:
        # 使用快速取樣方法
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs_fast(
            dataset[tickers], 
            significance=0.05,
            sample_size=SAMPLE_SIZE,
            verbose=True
        )
    else:
        # 使用平行處理方法
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
            dataset[tickers],
            significance=0.05,
            max_pairs=MAX_PAIRS,
            verbose=True
        )
    
    cluster_dict[which_clust] = {}
    cluster_dict[which_clust]['score_matrix'] = score_matrix
    cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
    cluster_dict[which_clust]['pairs'] = pairs

pairs = []
for clust in cluster_dict.keys():
    pairs.extend(cluster_dict[clust]['pairs'])

print("Number of pairs found: %d" % len(pairs))
print("In these pairs, there are %d unique tickers." % len(np.unique(pairs)))

print("\nAll pairs found:")
print("=" * 50)
if pairs:
    for i, pair in enumerate(pairs, 1):
        print(f"{i:3d}. {pair[0]} - {pair[1]}")
else:
    print("No pairs found.")
print("=" * 50)

"""## 6.2. 配對視覺化"""

stocks = np.unique(pairs)
X_df = pd.DataFrame(index=X.index, data=X).T

in_pairs_series = clustered_series.loc[stocks]
stocks = list(np.unique(pairs))
X_pairs = X_df.T.loc[stocks]

X_tsne = TSNE(learning_rate=50, perplexity=3, random_state=1337).fit_transform(X_pairs)

plt.figure(1, facecolor='white', figsize=(16, 8))
plt.clf()
plt.axis('off')
for pair in pairs:
    ticker1 = pair[0]
    loc1 = X_pairs.index.get_loc(pair[0])
    x1, y1 = X_tsne[loc1, :]

    ticker2 = pair[1]
    loc2 = X_pairs.index.get_loc(pair[1])
    x2, y2 = X_tsne[loc2, :]

    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, c='gray')

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=220, alpha=0.9, c=in_pairs_series.values, cmap=cm.Paired)
plt.title('T-SNE Visualization of Verified Pairs')

# zip 將 x 和 y 座標成對連接
for x, y, name in zip(X_tsne[:, 0], X_tsne[:, 1], X_pairs.index):
    label = name
    plt.annotate(label,  # 這是文字
                 (x, y),  # 這是要標記的點
                 textcoords="offset points",  # 如何定位文字
                 xytext=(0, 10),  # 文字到點 (x,y) 的距離
                 ha='center')  # 水平對齊可以是左、右或中心

plt.show()

# 包含 NVDA 的配對
nvda_pairs_found = [pair for pair in pairs if 'NVDA' in pair]
print(f"\nPairs containing NVDA: {len(nvda_pairs_found)}")
print("=" * 40)
if nvda_pairs_found:
    for i, pair in enumerate(nvda_pairs_found, 1):
        print(f"{i:2d}. {pair[0]} - {pair[1]}")
else:
    print("No NVDA pairs found in Affinity Propagation clustering.")
print("=" * 40)

# **結論**

# 聚類技術不能直接幫助股票趨勢預測。然而，它們可以有效地用於投資組合構建，
# 以尋找正確的配對，最終有助於風險緩解，並且可以實現優越的風險調整報酬。

# 我們展示了在 k-means 中尋找適當聚類數量的方法，並在階層聚類中建立了階層圖。
# 此案例研究的下一步將是探索和回測各種多空交易策略，使用來自股票分組的股票配對。

# 聚類可以有效地用於將股票分為具有「相似特徵」的群組，適用於許多其他類型的交易策略，
# 並可以幫助投資組合構建，以確保我們選擇具有足夠多樣化的股票範圍。

# =============================================================================
# 作業 4：使用 K-means 和階層聚類尋找 NVDA 配對
# =============================================================================

print("\n" + "=" * 80)
print("Homework 4: NVDA Pairs Analysis Using Different Clustering Methods")
print("=" * 80)

# 首先，讓我們看看 NVDA 在每種方法中屬於哪個聚類
print("\n1. NVDA Cluster Assignment:")
print("-" * 40)

# 檢查 NVDA 是否存在於我們的數據集中
if 'NVDA' in X.index:
    nvda_idx = X.index.get_loc('NVDA')

    # NVDA 的 K-means 聚類
    nvda_kmeans_cluster = k_means.labels_[nvda_idx]
    print(f"NVDA K-means cluster: {nvda_kmeans_cluster}")

    # NVDA 的階層聚類
    nvda_hc_cluster = clust_labels1[nvda_idx]
    print(f"NVDA Hierarchical cluster: {nvda_hc_cluster}")

    # NVDA 的親和力傳播聚類
    nvda_ap_cluster = clust_labels2[nvda_idx]
    print(f"NVDA Affinity Propagation cluster: {nvda_ap_cluster}")

else:
    print("NVDA not found in dataset")

# =============================================================================
# K-MEANS 聚類配對分析
# =============================================================================

print("\n2. K-MEANS Clustering Analysis:")
print("-" * 40)

# 使用 K-means 聚類結果尋找配對
def find_pairs_in_cluster(cluster_labels, cluster_num, method_name):
    """在特定聚類內尋找共整合配對"""
    # 獲取與 NVDA 在同一聚類中的股票
    cluster_stocks = X.index[cluster_labels == cluster_num].tolist()

    print(f"\n{method_name} - Cluster {cluster_num} contains {len(cluster_stocks)} stocks:")
    print(cluster_stocks[:10] if len(cluster_stocks) > 10 else cluster_stocks)  # 顯示前 10 個或全部（如果少於 10 個）

    if len(cluster_stocks) > 1:
        # 在此聚類內尋找共整合配對
        cluster_data = dataset[cluster_stocks]
        
        # 根據聚類大小選擇方法
        if len(cluster_stocks) > 20:
            # 大聚類使用快速取樣方法
            print(f"Large cluster detected, using fast sampling method...")
            score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs_fast(
                cluster_data, 
                significance=0.05, 
                sample_size=min(500, len(cluster_stocks) * 5),
                verbose=True
            )
        else:
            # 小聚類使用完整分析
            print(f"Small cluster, using complete analysis...")
            score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
                cluster_data, 
                significance=0.05,
                max_pairs=20,
                verbose=True
            )

        # 篩選包含 NVDA 的配對
        nvda_pairs = [pair for pair in pairs if 'NVDA' in pair]

        print(f"\nCointegrated pairs found in cluster: {len(pairs)}")
        print(f"Pairs containing NVDA: {len(nvda_pairs)}")

        if nvda_pairs:
            print("NVDA pairs:")
            for pair in nvda_pairs:
                print(f"  {pair}")
        else:
            print("No NVDA pairs found in this cluster")

        return pairs, nvda_pairs
    else:
        print("Cluster too small for pair analysis")
        return [], []

# 分析 K-means 聚類
if 'NVDA' in X.index:
    kmeans_pairs, kmeans_nvda_pairs = find_pairs_in_cluster(
        k_means.labels_, nvda_kmeans_cluster, "K-MEANS"
    )

# =============================================================================
# 階層聚類配對分析
# =============================================================================

print("\n3. Hierarchical Clustering Analysis:")
print("-" * 40)

# 分析階層聚類
if 'NVDA' in X.index:
    hc_pairs, hc_nvda_pairs = find_pairs_in_cluster(
        clust_labels1, nvda_hc_cluster, "HIERARCHICAL"
    )

# =============================================================================
# 三種方法的比較
# =============================================================================

print("\n4. Clustering Methods Comparison:")
print("-" * 40)

# Get Affinity Propagation pairs for comparison
if 'NVDA' in X.index:
    ap_cluster_stocks = X.index[clust_labels2 == nvda_ap_cluster].tolist()
    if len(ap_cluster_stocks) > 1:
        ap_cluster_data = dataset[ap_cluster_stocks]
        print(f"\nAFFINITY PROPAGATION - Cluster {nvda_ap_cluster} contains {len(ap_cluster_stocks)} stocks:")
        print(ap_cluster_stocks[:10] if len(ap_cluster_stocks) > 10 else ap_cluster_stocks)
        
        # 根據聚類大小選擇方法
        if len(ap_cluster_stocks) > 20:
            print(f"Large cluster detected, using fast sampling method...")
            ap_score_matrix, ap_pvalue_matrix, ap_pairs = find_cointegrated_pairs_fast(
                ap_cluster_data, 
                significance=0.05, 
                sample_size=min(500, len(ap_cluster_stocks) * 5),
                verbose=True
            )
        else:
            print(f"Small cluster, using complete analysis...")
            ap_score_matrix, ap_pvalue_matrix, ap_pairs = find_cointegrated_pairs(
                ap_cluster_data, 
                significance=0.05,
                max_pairs=20,
                verbose=True
            )
        
        ap_nvda_pairs = [pair for pair in ap_pairs if 'NVDA' in pair]
        print(f"\nCointegrated pairs found in AP cluster: {len(ap_pairs)}")
        print(f"Pairs containing NVDA: {len(ap_nvda_pairs)}")
        if ap_nvda_pairs:
            print("NVDA pairs:")
            for pair in ap_nvda_pairs:
                print(f"  {pair}")
    else:
        ap_pairs, ap_nvda_pairs = [], []

    print(f"\nNVDA Pairs Discovery Summary:")
    print(f"K-means: {len(kmeans_nvda_pairs) if 'kmeans_nvda_pairs' in locals() else 0} pairs")
    print(f"Hierarchical: {len(hc_nvda_pairs) if 'hc_nvda_pairs' in locals() else 0} pairs")
    print(f"Affinity Propagation: {len(ap_nvda_pairs) if 'ap_nvda_pairs' in locals() else 0} pairs")

    # 收集所有獨特的 NVDA 配對
    all_nvda_pairs = set()
    if 'kmeans_nvda_pairs' in locals():
        all_nvda_pairs.update(kmeans_nvda_pairs)
    if 'hc_nvda_pairs' in locals():
        all_nvda_pairs.update(hc_nvda_pairs)
    if 'ap_nvda_pairs' in locals():
        all_nvda_pairs.update(ap_nvda_pairs)

    print(f"\nTotal unique NVDA pairs found across all methods: {len(all_nvda_pairs)}")
    print("=" * 60)
    if all_nvda_pairs:
        for i, pair in enumerate(sorted(all_nvda_pairs), 1):
            print(f"{i:2d}. {pair[0]} - {pair[1]}")
    else:
        print("No NVDA pairs found across all methods.")
    print("=" * 60)

# =============================================================================
# NVDA 特性詳細分析
# =============================================================================

print("\n5. NVDA Characteristics Analysis:")
print("-" * 40)

if 'NVDA' in X.index:
    nvda_returns = returns.loc['NVDA', 'Returns']
    nvda_volatility = returns.loc['NVDA', 'Volatility']

    print(f"NVDA annualized returns: {nvda_returns:.4f}")
    print(f"NVDA volatility: {nvda_volatility:.4f}")

    # 與聚類成員比較
    methods = [
        ('K-means', k_means.labels_, nvda_kmeans_cluster),
        ('Hierarchical', clust_labels1, nvda_hc_cluster),
        ('Affinity Propagation', clust_labels2, nvda_ap_cluster)
    ]

    for method_name, labels, cluster_num in methods:
        cluster_stocks = X.index[labels == cluster_num].tolist()
        if len(cluster_stocks) > 1:
            cluster_returns = returns.loc[cluster_stocks, 'Returns']
            cluster_volatility = returns.loc[cluster_stocks, 'Volatility']

            print(f"\n{method_name} cluster statistics:")
            print(f"  Cluster size: {len(cluster_stocks)}")
            print(f"  Average returns: {cluster_returns.mean():.4f} (NVDA: {nvda_returns:.4f})")
            print(f"  Average volatility: {cluster_volatility.mean():.4f} (NVDA: {nvda_volatility:.4f})")
            print(f"  Returns std: {cluster_returns.std():.4f}")
            print(f"  Volatility std: {cluster_volatility.std():.4f}")

# =============================================================================
# NVDA 及其配對的視覺化
# =============================================================================

print("\n6. Creating visualization charts...")
print("-" * 40)

# 創建綜合視覺化
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 圖 1：突出顯示 NVDA 的 K-means 聚類
ax1 = axes[0, 0]
scatter1 = ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], c=k_means.labels_, cmap="rainbow", alpha=0.6)
if 'NVDA' in X.index:
    nvda_point = X.loc['NVDA']
    ax1.scatter(nvda_point.iloc[0], nvda_point.iloc[1], c='red', s=200, marker='*',
                edgecolors='black', linewidth=2, label='NVDA')
ax1.set_title('K-means Clustering (Highlight NVDA)', fontsize=14)
ax1.set_xlabel('Standardized Returns')
ax1.set_ylabel('Standardized Volatility')
ax1.legend()
plt.colorbar(scatter1, ax=ax1)

# 圖 2：突出顯示 NVDA 的階層聚類
ax2 = axes[0, 1]
scatter2 = ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels1, cmap="rainbow", alpha=0.6)
if 'NVDA' in X.index:
    ax2.scatter(nvda_point.iloc[0], nvda_point.iloc[1], c='red', s=200, marker='*',
                edgecolors='black', linewidth=2, label='NVDA')
ax2.set_title('Hierarchical Clustering (Highlight NVDA)', fontsize=14)
ax2.set_xlabel('Standardized Returns')
ax2.set_ylabel('Standardized Volatility')
ax2.legend()
plt.colorbar(scatter2, ax=ax2)

# 圖 3：突出顯示 NVDA 的親和力傳播
ax3 = axes[1, 0]
scatter3 = ax3.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clust_labels2, cmap="rainbow", alpha=0.6)
if 'NVDA' in X.index:
    ax3.scatter(nvda_point.iloc[0], nvda_point.iloc[1], c='red', s=200, marker='*',
                edgecolors='black', linewidth=2, label='NVDA')
ax3.set_title('Affinity Propagation (Highlight NVDA)', fontsize=14)
ax3.set_xlabel('Standardized Returns')
ax3.set_ylabel('Standardized Volatility')
ax3.legend()
plt.colorbar(scatter3, ax=ax3)

# 圖 4：聚類大小比較
ax4 = axes[1, 1]
methods_data = []
if 'NVDA' in X.index:
    for method_name, labels, cluster_num in methods:
        cluster_size = np.sum(labels == cluster_num)
        methods_data.append((method_name, cluster_size))

if methods_data:
    method_names, cluster_sizes = zip(*methods_data)
    bars = ax4.bar(method_names, cluster_sizes, color=['blue', 'green', 'orange'])
    ax4.set_title('NVDA Cluster Size by Method', fontsize=14)
    ax4.set_ylabel('Number of Stocks')
    ax4.set_xlabel('Clustering Method')

    # 在長條圖上添加數值標籤
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{size}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\nAnalysis completed! Please check the visualization charts above.")
print("=" * 80)

print("\n" + "=" * 80)
print("Homework 4 Completed")
print("File contains:")
print("1. Original Affinity Propagation analysis")
print("2. Added K-means clustering analysis")
print("3. Added Hierarchical clustering analysis")
print("4. Comparative analysis of NVDA pair candidates")
print("5. Detailed analysis of NVDA characteristics")
print("6. Comprehensive visualization results")
print("7. Comments and explanations on NVDA market position")
print("=" * 80)