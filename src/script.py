# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import math

from sklearn.cluster import KMeans
# %%
df = pd.read_csv('../BancoDeDados.csv')
# %%
df.info()
# %%
df.head()
# %%
def plot_perc(st,dados):
  plt.figure(figsize = (20,8))
  
  g = sns.countplot(x = st, data = dados, orient = 'h')
  g.set_ylabel('Contagem', fontsize = 17)
  
  sizes = []
  
  for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x() + p.get_width() / 1.6,
           height + 200,
           '{:1.2f}%'.format(height / 116581 * 100),
           ha = 'center',
           va = 'bottom',
           fontsize = 12)
    g.set_ylim(0, max(sizes) * 1.1)
    
# %%
plot_perc('estado_cliente', df)
plot_perc('estado_vendedor', df)
plot_perc('pagamento_tipo', df)
# %%
df_olist = df[['id_unico_cliente','id_cliente','horario_pedido','item_id','preco']]
df_compra = df.groupby('id_unico_cliente').horario_pedido.max().reset_index()
df_compra.columns = ['id_unico_cliente', 'DataMaxCompra']
df_compra['DataMaxCompra'] = pd.to_datetime(df_compra['DataMaxCompra'])
# %%
# Recencia
df_compra['Recencia'] = (df_compra['DataMaxCompra'].max() - df_compra['DataMaxCompra']).dt.days
df_usuario = pd.merge(df_olist,df_compra[['id_unico_cliente','Recencia']], on = 'id_unico_cliente')
# %%
def calculate_wcss(data):
  wcss = []
  for k in range(1,10):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X = data)
    data['Clusters']= kmeans.labels_
    wcss.append(kmeans.inertia_)
  return wcss
# %%
df_recencia = df_usuario[['Recencia']]
soma_quadrados = calculate_wcss(df_recencia)
# %%
plt.figure(figsize= (10,5))
plt.plot(soma_quadrados)
plt.xlabel('Número de clusters')
plt.show()
# %%
def numero_otimo_clusters(wcss):
  x1, y1 = 2, wcss[0]
  x2, y2 = 20, wcss[len(wcss) - 1]
  
  distances = []
  
  for i in range(len(wcss)):
    x0 = i + 2
    y0 = wcss[i]
    
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2*y1 - y2*x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distances.append(numerator / denominator)
    
  return distances.index(max(distances)) + 2 
# %%
#recuperando o melhor n para o k-means
n = numero_otimo_clusters(soma_quadrados)
# %%
kmeans = KMeans(n_clusters= n)
df_usuario['RecenciaCluster'] = kmeans.fit_predict(df_recencia)
# %%
def ordenador_cluster(cluster, target, df):
  agrupado_por_cluster = df.groupby(cluster)[target].mean().reset_index()
  agrupado_por_cluster_ordenado = agrupado_por_cluster.sort_values(by = target, ascending = False).reset_index(drop = True)
  agrupado_por_cluster_ordenado['index'] = agrupado_por_cluster_ordenado.index
  juntando_cluster = pd.merge(df, agrupado_por_cluster_ordenado[[cluster,'index']], on= cluster)
  removendo_dados = juntando_cluster.drop([cluster], axis = 1)
  df_final = removendo_dados.rename(columns = {'index' : cluster})
  return df_final
# %%

df_frequencia = df.groupby('id_unico_cliente').pedido_aprovado.count().reset_index()
df_frequencia.columns = ['id_unico_cliente','Frequencia']

df_usuario = ordenador_cluster('RecenciaCluster', 'Recencia', df_usuario)
df_usuario = pd.merge(df_usuario,df_frequencia, on = 'id_unico_cliente')
df_frequencia = df_usuario[['Frequencia']]
df_usuario['FrequenciaCluster'] = kmeans.fit_predict(df_frequencia)
df_usuario = ordenador_cluster('FrequenciaCluster', 'Frequencia', df_usuario)