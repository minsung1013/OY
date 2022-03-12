import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading data
@st.cache
def load_data2():
    data_ = pd.read_pickle('data_oliveyoung.pkl')
    df_preservatives_ = pd.read_pickle('df_preservatives2.pkl')
    return data_, df_preservatives_


data_copy, df_preservatives = load_data2()

# title
st.subheader('Clustering model Olive Young skincare product')
st.write('total 1433 products were clustered')
st.write('written by Mason Choi Evonik Korea')

menu = ['label', 'manufacturer', 'nation', '1,2-Hexanediol', 'Benzoic Acid', 'Benzyl Alcohol', 'Butylparaben',
        'Capryloyl Glycine', 'Caprylyl Glycol', 'Cetrimonium Bromide', 'Chlorhexidine Digluconate',
        'Dehydroacetic Acid', 'Ethylhexylglycerin', 'Ethylparaben', 'Ferulic Acid', 'Formic Acid',
        'Glyceryl Caprylate', 'Levulinic Acid', 'Methylisothiazolinone', 'Methylparaben', 'Pentylene Glycol',
         'Phenethyl Alcohol', 'Phenoxyethanol', 'Phenylpropanol', 'Potassium Sorbate', 'Propylparaben',
         'Sodium Benzoate', 'Sodium Dehydroacetate', 'Sodium Hydroxymethylglycinate', 'Sodium Levulinate',
        'Sodium Salicylate', 'Sorbic Acid', 'Zinc Salicylate', 'p-Anisic Acid']

######################################################################
# Market Analysis
with st.form(key='my_form2'):
    st.subheader('Market analysis')
    hue_select = st.selectbox('select an option', menu)
    submit_button = st.form_submit_button(label='Submit')


@st.cache(allow_output_mutation=True)
def plot_cluster(df, key, size):
    eliment = df[key].unique()
    tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    basic = ['b', 'g', 'r', 'c', 'm', 'y']

    color_list = tab + basic

    key_option = ['label', 'manufacturer', 'nation', 'kmeans']
    if key in key_option:
        dic_color = {}
        for i, e in enumerate(eliment):
            dic_color[e] = color_list[i]
    else:
        dic_color = {0:'tab:blue', 1:'tab:orange'}

    fig, ax = plt.subplots(figsize=(size, size))

    grouped = df.groupby(key)
    for k, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=k, c=dic_color[k], s=20, alpha=0.7)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return fig

st.pyplot(plot_cluster(data_copy, hue_select, 5))

st.write('total products:', data_copy.shape[0])

st.subheader('number of products')
col2_1, col2_2 = st.columns(2)
with col2_1:
    st.write(data_copy[hue_select].value_counts())

with col2_2:
    st.bar_chart(data_copy[hue_select].value_counts())

st.subheader('most frequent preservatives')
col2_1a, col2_2b = st.columns(2)
with col2_1a:
    mfp = pd.DataFrame(df_preservatives.sum(axis=0).sort_values(ascending=False))
    mfp['%'] = mfp.iloc[:][0] / data_copy.shape[0] * 100
    st.write(mfp)

with col2_2b:
    mfp_top_10 = mfp.iloc[:10]
    st.bar_chart(mfp_top_10['%'])

#####################################################################################
# analysis
st.write('\n\n\n\n')
with st.form(key='my_form3'):
    st.subheader('manufacturer analysis')
    manufact = st.selectbox("manufacturer",
                            ("kolmar", "cosmax", "AP", "LG", 'cosvision', 'cosmeca', 'intercos', 'koreana', 'others'))
    hue_select2 = st.selectbox('class', menu)
    num_cluster = int(st.slider('number of cluster', 1, 15, 15))
    submit_button = st.form_submit_button(label='Submit')

data_subset = data_copy[data_copy['manufacturer'] == manufact]
df_preservatives_subset = df_preservatives[data_copy['manufacturer'] == manufact]

st.subheader(manufact)
st.pyplot(plot_cluster(data_subset, hue_select2, 5))

col3_1, col3_2 = st.columns(2)
with col3_1:
    st.subheader("kmeans clustering")
    kmeans = KMeans(n_clusters=int(num_cluster), random_state=42).fit(data_subset.loc[:][['x', 'y']])
    data_subset['kmeans'] = kmeans.labels_
    st.pyplot(plot_cluster(data_subset, 'kmeans', 4))

with col3_2:
    st.subheader('# of products')
    st.write(data_subset['kmeans'].value_counts().sort_index())

col4_1, col4_2 = st.columns(2)
with col4_1:
    st.subheader('preservatives')
    mfp_sub = pd.DataFrame(df_preservatives_subset.sum(axis=0).sort_values(ascending=False))
    mfp_sub['%'] = mfp_sub.iloc[:][0] / data_subset.shape[0] * 100
    st.write(mfp_sub)

with col4_2:
    st.subheader('top 10')
    mfp_sub_top_10 = mfp_sub.iloc[:10]
    st.bar_chart(mfp_sub_top_10['%'])

st.subheader('preservative system')

# st.table(data_subset['p_system'].value_counts())
p_system_table = data_subset['p_system'].value_counts().sort_values(ascending=False)
st.write(p_system_table)

p_system_menu = p_system_table.index

st.write('\n\n\n\n')
with st.form(key='my_form4'):
    st.subheader('product finder')
    mode = st.radio('show by', ('cluster', 'p_system'))
    cluster_selected = st.slider('cluster number', 0, int(num_cluster)-1, 0)
    p_system_selected = st.selectbox('p_system', p_system_menu)
    submit_button = st.form_submit_button(label='Submit')

def open_cluster(df, num):
    cluster = df[df['kmeans'] == int(num)]
    for i in range(len(cluster)):
        st.write('product: ', cluster.iloc[i]['product'], '/', cluster.iloc[i]['label'])
        st.write(', '.join(cluster.iloc[i]['inci']))
        st.write('p_system:', cluster.iloc[i]['p_system'])
        st.text("==============================================================================")

def open_p_system(df, p_system_search):
    cluster = df[df['p_system'] == p_system_search]
    for i in range(len(cluster)):
        st.write('product: ', cluster.iloc[i]['product'], '/', cluster.iloc[i]['label'])
        st.write(', '.join(cluster.iloc[i]['inci']))
        st.write('p_system:', cluster.iloc[i]['p_system'])
        st.text("==============================================================================")

if mode == 'cluster':
    df_cluster = data_subset[data_subset['kmeans'] == cluster_selected]
    indice = list(df_cluster.index)

    col5_1, col5_2 = st.columns(2)
    with col5_1:
        st.subheader('product types')
        st.table(df_cluster.groupby('label')['product'].count())

    with col5_2:
        st.subheader('clustering map')
        data_subset['mode'] = data_subset['kmeans'].apply(lambda x: 1 if x == cluster_selected else 0)
        st.pyplot(plot_cluster(data_subset, 'mode', 4))

    st.subheader('product details')
    st.write('total number of products in the cluster: ', len(indice))
    open_cluster(data_subset, cluster_selected)


elif mode == 'p_system':
    df_cluster = data_subset[data_subset['p_system'] == p_system_selected]
    indice = list(df_cluster.index)

    col5_1, col5_2 = st.columns(2)
    with col5_1:
        st.subheader('product types')
        st.table(df_cluster.groupby('label')['product'].count())

    with col5_2:
        st.subheader('clustering map')
        data_subset['mode'] = data_subset['p_system'].apply(lambda x: 1 if x == p_system_selected else 0)
        st.pyplot(plot_cluster(data_subset, 'mode', 4))

    st.subheader('product details')
    st.write('total number of products in the p_system: ', len(indice))
    open_p_system(data_subset, p_system_selected)