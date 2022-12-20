# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:13:41 2022

@author: 706028
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import matlib
import random
import openpyxl
import datetime

st.markdown("### 仮想候補の生成")

def generating_samples_2(setting_of_generation,number_of_generating_samples=10000, sum_of_components=1):

    #desired_sum_of_components = 1 # 合計を指定する特徴量がある場合の、合計の値。例えば、この値を 100 にすれば、合計を 100 にできます
    # 0 から 1 の間の一様乱数でサンプル生成
    np.random.seed(11) # 乱数を生成するためのシードを固定
    x_generated = np.random.rand(number_of_generating_samples, setting_of_generation.shape[1])

    # 上限・下限の設定
    x_upper = setting_of_generation.loc['max', :]  # 上限値
    x_lower = setting_of_generation.loc['min', :]  # 下限値
    x_generated = x_generated * (x_upper.values - x_lower.values) + x_lower.values  # 上限値から下限値までの間に変換
    x_generated = pd.DataFrame(x_generated, columns=setting_of_generation.columns)

    # 合計を desired_sum_of_components にする特徴量がある場合
    x_generated_groups = pd.DataFrame()
    if setting_of_generation.loc['group', :].sum() != 0:
        for group_number in range(1, int(setting_of_generation.iloc[2, :].max()) + 1):
            variable_group = setting_of_generation.loc[:,(setting_of_generation.iloc[2, :] == group_number)].columns.tolist()
            x_generated_group = x_generated.loc[:,variable_group]
            #丸め込み
            for column in variable_group:
                x_generated_group.loc[:, column] = float(setting_of_generation.loc['kizami', column]) * np.round(x_generated_group.loc[:, column] /  float(setting_of_generation.loc['kizami', column]))
            x_generated_group.iloc[:,-1] = sum_of_components - x_generated_group.iloc[:,:-1].sum(axis=1)

            x_generated_group =  x_generated_group[x_generated_group[variable_group[-1]] <=  x_upper.loc[variable_group[-1]]]
            x_generated_group =  x_generated_group[x_generated_group[variable_group[-1]] >=  x_lower.loc[variable_group[-1]]]

            x_generated_groups = pd.concat([x_generated_groups,x_generated_group],axis=1).dropna()

        x_generated_no_group = x_generated.loc[x_generated_groups.index, :].drop(x_generated_groups.columns.tolist(),axis=1)
    else:
        x_generated_no_group = x_generated.copy()
        
    #丸めこみ
    for j in x_generated_no_group.columns:
        #x_generated[:, variable_number] = np.round(x_generated[:, variable_number], int(setting_of_generation.iloc[3, variable_number]))
        x_generated_no_group.loc[:, j] = float(setting_of_generation.loc['kizami', j]) * np.round(x_generated_no_group.loc[:, j] /  float(setting_of_generation.loc['kizami', j]))

    x_generated_final = pd.concat([x_generated_groups, x_generated_no_group],axis=1).reindex(columns=x_generated.columns)

    return x_generated_final


def D_optimization(x_generated, x_obtained=None, number_of_samples=10,number_of_random_searches = 1000):
    #一旦リセット
    selected_sample_indexes = None

    #number_of_random_searches = 1000 # ランダムにサンプルを選択して D 最適基準を計算する繰り返し回数

    # 実験条件の候補のインデックスの作成
    all_indexes = list(x_generated.index)

    # D 最適基準に基づくサンプル選択
    np.random.seed(10) # 乱数を生成するためのシードを固定
    for random_search_number in range(number_of_random_searches):
        # 1. ランダムに候補を選択
        new_selected_indexes = np.random.choice(all_indexes, number_of_samples)
        new_selected_samples = x_generated.loc[new_selected_indexes, :]

        if x_obtained is not None:
            new_selected_samples = pd.concat([x_obtained.loc[:,new_selected_samples.columns.tolist()],new_selected_samples])


        # 2. オートスケーリングした後に D 最適基準を計算
        autoscaled_new_selected_samples = (new_selected_samples - new_selected_samples.mean()) / new_selected_samples.std()
        xt_x = np.dot(autoscaled_new_selected_samples.T, autoscaled_new_selected_samples)
        d_optimal_value = np.linalg.det(xt_x)
        # 3. D 最適基準が前回までの最大値を上回ったら、選択された候補を更新
        if random_search_number == 0:
            best_d_optimal_value = d_optimal_value.copy()
            selected_sample_indexes = new_selected_indexes.copy()
        else:
            if best_d_optimal_value < d_optimal_value:
                best_d_optimal_value = d_optimal_value.copy()
                selected_sample_indexes = new_selected_indexes.copy()
    selected_sample_indexes = list(selected_sample_indexes) # リスト型に変換

    # 選択されたサンプル、選択されなかったサンプル
    selected_samples = x_generated.loc[selected_sample_indexes, :]  # 選択されたサンプル
    remaining_indexes = list(set(all_indexes) - set(selected_sample_indexes))  # 選択されなかったサンプルのインデックス
    remaining_samples = x_generated.loc[remaining_indexes, :]  # 選択されなかったサンプル

    #print(selected_samples.corr()) # 相関行列の確認
    return selected_samples, best_d_optimal_value

def generate_all_combination(df_setting_quantitative):
    df0 = pd.DataFrame()
    for column in df_setting_quantitative.columns:
        df0[column] = np.arange(df_setting_quantitative.loc['最小値',column], df_setting_quantitative.loc['最大値',column],df_setting_quantitative.loc['間隔',column])

    df0.insert(0,'dummy',1)

    data = df0[['dummy',df_setting_quantitative.columns[0]]].dropna()
    for column in df_setting_quantitative.columns[1:]:
        data = data.merge(df0[['dummy',column]].dropna(), on='dummy', how='outer')
    group_1 = df_setting_quantitative.T.query('グループ == 1').index.tolist()

    df1 = data[data[group_1].sum(axis=1).round(5)==1]    #適当に丸めないと1にならない

    return df1


w_setting = st.sidebar.file_uploader("仮想候補設定Excelアップロード", type='xlsx')

if w_setting:  #dataをアップロードしたらスタート
    df0 = pd.ExcelFile(w_setting)
    df_qualitative = df0.parse('質的変数', header=0,encoding='shift-jis')
    df_quantitative = df0.parse('量的変数', header=0,index_col=0,encoding='shift-jis')
    df0.close()

    st.markdown('質的変数')
    st.dataframe(df_qualitative)
    st.markdown('量的変数')
    st.dataframe(df_quantitative)

    How_to_make_samples = st.radio('仮想候補の作り方', ('ランダムに生成','全組み合わせ'))
    
    if df_quantitative.loc['group', :].sum() != 0:
        sum_of_components = st.radio('比率などの合計量',(1,100))
    else:
        sum_of_components = 1
    

    if How_to_make_samples == '全組み合わせ':
        #質的変数の組み合わせ
        df_qualitative_0 = pd.DataFrame(df_qualitative.iloc[:,0].dropna(how='all'))
        df_qualitative_0.insert(0,'dummy',1,allow_duplicates=True)
        df_qualitative_comb = df_qualitative_0.copy()
        for i in range(1,df_qualitative_0.shape[1]+1):
            df_qualitative_i = pd.DataFrame(df_qualitative.iloc[:,i].dropna(how='all'))
            df_qualitative_i.insert(0,'dummy',1,allow_duplicates=True)
            df_qualitative_comb = df_qualitative_comb.merge(df_qualitative_i, on='dummy', how='outer')
        #st.dataframe(df_qualitative_comb)

        #量的変数の組み合わせ
        df_quantitative_comb = generate_all_combination(df_quantitative)
        #st.dataframe(df_quantitative_comb)

        #全組み合わせ
        data1 = df_qualitative_comb.merge(df_quantitative_comb, on='dummy', how='outer')
        data2 = data1.drop(['dummy'], axis=1)

        data2 = data2.rename(index=lambda s: 'c_' + str(s).zfill(len(str(data2.shape[0]))))  #indexの先頭にcandidateの番号をつける

    elif How_to_make_samples == 'ランダムに生成':
        number_of_samples = st.number_input('生成するおよその候補数',value=10000)
        #量的変数
        df_quantitative_comb = generating_samples_2(df_quantitative,number_of_generating_samples=number_of_samples,sum_of_components=sum_of_components)
        df_quantitative_comb = pd.DataFrame(df_quantitative_comb,columns=df_quantitative.columns)

        #質的変数はランダムchoice
        for column in df_qualitative.columns:
            for index in df_quantitative_comb.index:
                df_quantitative_comb.loc[index,column]=random.choice(df_qualitative[column].dropna())

        #並び替え
        data1 = df_quantitative_comb.reindex(columns=df_qualitative.columns.tolist()+df_quantitative.columns.tolist())
        data2 = data1.copy()

        data2 = data2.rename(index=lambda s: 'c_' + str(s).zfill(len(str(data2.shape[0]))))

    st.markdown('候補の数:' + str(data2.shape[0]))
    st.dataframe(data2)

    get_dummy = st.checkbox('ダミー変数にするか',value=False)
    if get_dummy:
        dummy_var = st.multiselect('ダミー化する変数', df_qualitative.columns.tolist(), df_qualitative.columns.tolist())
        data3 = pd.get_dummies(data2, columns=dummy_var)
        #元の列を追加
        data3 = pd.concat([data2.loc[:, df_qualitative.columns],data3],axis=1)
    else:
        data3 = data2.copy()

    st.dataframe(data3)

    data4 = data3.copy()
    
    comb_name = st.text_input('比率や%などの単位名','ratio')
    
    st.markdown(comb_name + 'が0の名前はnan、ダミーは0にし、重複を削除')
    

    #ratioが0のダミーは0、ratioが0の名前はnan
    for j in [s for s in df_quantitative.columns.tolist() if comb_name in s]:
        k = j.replace('_'+comb_name,'')
        #ratioが0のダミーは0
        data4.loc[data4[data4[j]==0].index, data4.filter(regex=k,axis=1).columns] = data4.loc[data4[data4[j]==0].index, data4.filter(regex=k,axis=1).columns].replace(1,0)
        #ratioが0の名前はnan
        data4.loc[data4[data4[j]==0].index, k] = np.nan

    #重複削除
    data4 = data4.drop_duplicates()

    
    st.markdown('候補の数 :' + str(data4.shape[0]))
    st.dataframe(data4)

    #候補の出力
    csv_output = st.checkbox('生成された候補の出力',value=True)
    if csv_output:
        csv_name = st.text_input('ファイル名','candidates_'+str(datetime.date.today())+'.csv')
#         if st.button('Download'):
#             data4.to_csv(csv_name,encoding='shift-jis')
        csv = data4.to_csv().encode('shift-jis')
        st.download_button(label="Download",data=csv,file_name=csv_name)


    #D最適計画
    data_for_d_opt = data4.drop(df_qualitative,axis=1)
    do_d_opt = st.checkbox('D最適計画',value=False)
    if do_d_opt:
        number_of_selecting_samples = st.number_input('D最適基準で選択するサンプル数',1,50,10)
        including_obtained_data = st.checkbox('実験済のデータを考慮するか',value=False)
        number_of_random_searches = 1000
        if including_obtained_data==False:
            st.markdown('実験データなしor0からD最適')
            D_selected_samples, d_value = D_optimization(data_for_d_opt, x_obtained=None, number_of_samples=number_of_selecting_samples,number_of_random_searches = number_of_random_searches)
            st.markdown('D最適で選ばれた候補')
            st.dataframe(D_selected_samples)
            st.markdown('D_value: '+str(d_value))
        else:
            st.markdown('実験データをアップロードしてください')
            file_type = st.sidebar.radio('実験データファイル形式',('csv','xlsx'))
            if file_type == 'csv':
                w = st.sidebar.file_uploader("実験データアップロード", type = 'csv')
            elif file_type == 'xlsx':
                w = st.sidebar.file_uploader("実験データアップロード", type = 'xlsx')
            else:
                pass

            if w:
                if file_type == 'csv':
                    data_exp = pd.read_csv(w, index_col=0, encoding='utf-8')
                elif file_type == 'xlsx':
                    data_exp_0 = pd.ExcelFile(w)
                    data_exp = data_exp_0.parse(data_exp_0.sheet_names[0], index_col=0,header=0, encoding='utf-8')
                    data_exp_0.close()
                
                if get_dummy:
                    data_exp2 = pd.get_dummies(data_exp, columns=dummy_var)
                else:
                    data_exp2 = data_exp.copy()
                    
                st.dataframe(data_for_d_opt)
                st.dataframe(data_exp2)
                
        
                D_selected_samples, d_value = D_optimization(data_for_d_opt, x_obtained=data_exp2, number_of_samples=number_of_selecting_samples,number_of_random_searches = number_of_random_searches)
                st.markdown('D最適で選ばれた候補')
                st.dataframe(D_selected_samples)
                st.markdown('D_value: '+str(d_value))
            else:
                pass


            st.markdown('------------------------------------------------------------------------------------------------')
            #描画する説明変数
            st.markdown('描画する説明変数')
            col1, col2, col3 = st.columns(3)
            with col1:
                X_variable = st.selectbox('説明変数1',data_for_d_opt.columns)
            with col2:
                Y_variable = st.selectbox('説明変数2',data_for_d_opt.columns,index=1)
            with col3:
                Z_variable = st.selectbox('説明変数3',data_for_d_opt.columns,index=2)

            #2次元散布図
            plt.rcParams['font.size'] = 6
            fig, ax = plt.subplots(1,2,figsize=(4, 1.7))

            if including_obtained_data:
                if data_exp2:
                    ax[0].scatter(data_exp2[X_variable],data_exp2[Y_variable],s=5, c='blue')
            else:
                pass
            ax[0].scatter(D_selected_samples[X_variable],D_selected_samples[Y_variable],s=5, c='red')
            ax[0].set_xlabel(X_variable,fontsize=6)
            ax[0].set_ylabel(Y_variable,fontsize=6)

            if including_obtained_data:
                if data_exp2:
                    ax[1].scatter(data_exp2[X_variable],data_exp2[Z_variable],s=5, c='blue')
            else:
                pass
            ax[1].scatter(D_selected_samples[X_variable],D_selected_samples[Z_variable],s=5, c='red')
            ax[1].set_xlabel(X_variable,fontsize=6)
            ax[1].set_ylabel(Z_variable,fontsize=6)
            plt.tight_layout()
            st.pyplot(fig)

            if including_obtained_data:
                st.markdown('実験候補：赤, 実験データ: 青')
                st.markdown('実験候補：赤')
            else:
                st.markdown('実験候補：赤')


        #
        #
        # #3次元散布図
        # plt.rcParams['font.size'] = 8
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # # X,Y,Z軸にラベルを設定
        # ax.set_xlabel(X_variable)
        # ax.set_ylabel(Y_variable)
        # ax.set_zlabel(Z_variable)
        #
        # ax.plot(df[X_variable],df[Y_variable],df[Z_variable],c='blue',marker="o",linestyle='None')  #実験データ
        # ax.plot(D_selected_samples[X_variable],D_selected_samples[Y_variable],D_selected_samples[Z_variable],c='red',marker="o",linestyle='None') #実験候補
        # st.pyplot(fig)
