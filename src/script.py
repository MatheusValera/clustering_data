# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
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
# %%
plot_perc('estado_vendedor', df)
# %%
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
df_usuario
# %%
