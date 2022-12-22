# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:13:41 2022

@author: 706028
"""

import streamlit as st
from PIL import Image

st.markdown("###  Materials Infomatics Web app")  #🦙
# #st.sidebar.markdown("# Main page 🎈")
# st.write('可視化')
# st.write('仮想候補の生成')
# st.write('回帰と予測')
# st.write('ベイズ最適化')

#image = Image.open('Analysis_flow.png')

st.image(image, caption='解析フロー',use_column_width=True)
