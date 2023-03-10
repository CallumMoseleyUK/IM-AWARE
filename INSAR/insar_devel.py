

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import TruncatedSVD

file_name = '100points.csv'

data = pd.read_csv(file_name)
data = data.drop("Unnamed: 0",axis=1)
dam_ids = data['Dam'].unique()

data['unique_loc'] = data.apply(lambda row: '{:.4f}-{:.4f}'.format(row['Lat'], row['Long']),axis=1)


dam_data = {}
for d in dam_ids:
    tmp = data.loc[data.loc[:,'Dam'] == d,:]
    loc = tmp['unique_loc'].unique()
    
    dam_data[d] = {}
    dam_data[d]['raw_data'] = tmp
    dam_data[d]['formatted_data'] = pd.DataFrame(index=tmp.loc[:, 'End_Time'].unique(), columns=loc)
    for l in loc:
        dam_data[d]['formatted_data'].loc[:,l] = tmp.loc[tmp.loc[:,'unique_loc']==l,'Disp_rate_mmDay'].values

    decomp_dat = dam_data[d]['formatted_data'].values
    U, s, V = np.linalg.svd(decomp_dat)

    dam_data[d]['SVD'] = {}
    dam_data[d]['SVD']['U'] = U
    dam_data[d]['SVD']['s'] = s
    dam_data[d]['SVD']['V'] = V

    svd = TruncatedSVD(5)
    X_trnc = svd.fit_transform(dam_data[d]['formatted_data'])

    dam_data[d]['SVD']['Trunc'] = X_trnc

    S = np.zeros((decomp_dat.shape[0], decomp_dat.shape[1]))
    S[:decomp_dat.shape[1], :decomp_dat.shape[1]] = np.diag(s)
    n_component = 10
    S = S[:, :n_component]
    VT = V.T
    VT = VT[:n_component, :]
    A = U.dot(S.dot(VT))
    # print(A)
    plt.figure(figsize = (10,15))
    plt.matshow(A, cmap='jet')
    plt.title(d)
    

#Plot sigma
plt.figure(figsize=(20,10))
for d in dam_ids:
    plt.plot(dam_data[d]['SVD']['s'], label=d)
plt.yscale('log')
plt.ylim(10E-2,10E2)
plt.legend(fontsize=16)
plt.xlabel('Component',fontsize=18)
plt.ylabel(r'$\Sigma$',fontsize=28)
plt.show()





