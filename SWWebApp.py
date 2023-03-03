#!/usr/bin/env python
# coding: utf-8

# In[12]:

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import sqrt
from streamlit_option_menu import option_menu
import csv
import io
import ast

def tryAsNumber(string):
    try:
        return int(string)
    except:
        try:
            return float(string)
        except:
            return string

def loadCSVDataFile(filename):   
    data = []
    with open(filename) as csvfile:
        datareader = csv.reader(csvfile)
        for record in datareader:
            if len(record)>0:
                data.append([tryAsNumber(val) for val in record]) 
    return data

from math import sqrt

def euclidean_distance(row1, row2):
    distance = 0.0                            
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2  
    return sqrt(distance)               
    
def get_neighbors(train, test_row, num_neighbors):
    distances = []
    for i, train_row in enumerate(train):
        if train_row != test_row:  # exclude test row from neighbors
            dist = euclidean_distance(test_row, train_row)
            distances.append((i, dist))  # append index of train row instead of row itself
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
    pass

data =loadCSVDataFile("product_images.csv")

import matplotlib.pyplot as plt


def read_clusters_from_csv(csv_file_name):
    Cluster = {}
    ClusterIndice = {}

    with open(csv_file_name, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = int(row['Cluster'])
            coord = tuple(map(float, row['Coordinate'][1:-1].split(',')))
            indice = int(row['Indice'])

            if label not in Cluster:
                Cluster[label] = []
                ClusterIndice[label] = []

            Cluster[label].append(coord)
            ClusterIndice[label].append(indice)

    return Cluster, ClusterIndice
FinalCluster, FinalClusterIndice = csvToClusters("clusters.csv")

def graphImage(imageNumber):

    row_idx = imageNumber - 1
    row_pixels = clusterset[row_idx]
    image = np.array(row_pixels).reshape((28, 28))
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.savefig('my_plot.png')
    return 'my_plot.png'

def graphImageN(imageNumber):

    row_idx = imageNumber - 1
    row_pixels = neighbors[row_idx]
    image = np.array(row_pixels).reshape((28, 28))
    plt.imshow(image, cmap='binary')
    plt.axis('off')
    plt.savefig('my_plot.png')
    return 'my_plot.png'


# Define the icons to use for the sidebar menu
icons = ["house", "basket3", "globe", "star", "chat"]

# Set up the Streamlit app
st.set_page_config(page_title="Sapphire Clothing")

# Add the Bootstrap CSS stylesheet
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">', unsafe_allow_html=True)

# Set up the sidebar menu
with st.sidebar:
    # Display the icons for each menu option
    for icon in icons:
        st.markdown(f'<i class="bi bi-{icon}"></i>', unsafe_allow_html=True)
    
    # Define the menu options and default index
    options = ["Home", "Products", "About Us", "Reviews", "Contact Us"]
    default_index = 0
    
    # Display the menu and get the selected option
    selected = option_menu(
        menu_title="Main Menu",
        options=options,
        icons=icons,
        default_index=default_index
    )

logo1=Image.open("logo1.png")

# Display information based on the selected option
if selected == "Home":
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local("landing.png")    



elif selected == "Products":
    st.image(logo1)
    st.title("Our Products")
    product_options = ["T-shirts", "Trousers", "Outerwear", "Sneaker/Sandal", "Boots", "Bags", "Dresses"]
    selected_product = st.selectbox("Select a product", product_options)
    if selected_product == "T-shirts":

        @st.cache_resource
        def get_image0(image_idx):
            row_pixels = clusterset0[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
        
        st.write("Here are our T-shirts:")
        clusterset0 = FinalCluster[0]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset0)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image0(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset0, clusterset0[n], 5)
                    for i in range(5):
                        st.image(get_image0(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
            
    elif selected_product == "Trousers":
        @st.cache_resource
        def get_image1(image_idx):
            row_pixels = clusterset1[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
        
        Trousers=Image.open("trousers.png")
        st.image(Trousers)
        st.write("Here are our Trousers:")
        clusterset1 = FinalCluster[1]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset1)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image1(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset1, clusterset1[n], 5)
                    for i in range(5):
                        st.image(get_image1(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
            
    elif selected_product == "Outerwear":
        @st.cache_resource
        def get_image2(image_idx):
            row_pixels = clusterset2[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
        
        st.write("Here are our Outerwear products:")
        clusterset2 = FinalCluster[2]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset2)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image2(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset2, clusterset2[n], 5)
                    for i in range(5):
                        st.image(get_image2(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
          
    elif selected_product == "Sneakers/Sandals":
        @st.cache_resource
        def get_image3(image_idx):
            row_pixels = clusterset3[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
    
        
        st.write("Here are our Sneakers and Sandals:")
        clusterset3 = FinalCluster[3]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset3)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image3(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset3, clusterset3[n], 5)
                    for i in range(5):
                        st.image(get_image3(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
          
    elif selected_product == "Boots":
        @st.cache_resource
        def get_image4(image_idx):
            row_pixels = clusterset4[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
        
        
        st.write("Here are our Boots:")
        clusterset4 = FinalCluster[4]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset4)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image4(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset4, clusterset4[n], 5)
                    for i in range(5):
                        st.image(get_image4(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
          
    elif selected_product == "Bags":
        @st.cache_resource
        def get_image5(image_idx):
            row_pixels = clusterset5[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
       
        Bag=Image.open("bags.png")
        st.image(Bag)
        st.write("Here are our Bags:")
        clusterset5 = FinalCluster[5]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset5)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image5(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset5, clusterset5[n], 5)
                    for i in range(5):
                        st.image(get_image5(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
          
    elif selected_product == "Dresses":
        @st.cache_resource
        def get_image6(image_idx):
            row_pixels = clusterset6[image_idx]
            image = np.array(row_pixels).reshape((28, 28))
            plt.imshow(image, cmap='binary')
            plt.axis('off')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            return buf.getvalue()
            pass
        
        Dressess=Image.open("dress.png")
        st.image(Dressess)
        st.write("Here are our Dresses:")
        clusterset6 = FinalCluster[6]
        START_INDEX = int(st.experimental_get_query_params().get("start_index", "0")[0])

        num_items = len(clusterset6)

        cols = st.columns(4)
        for n in range(START_INDEX, min(num_items, START_INDEX + 40)):
            with cols[n % 4]:
                st.image(get_image6(n))
                if st.button('Show similar items', n):
                    neighbors = get_neighbors(clusterset6, clusterset6[n], 5)
                    for i in range(5):
                        st.image(get_image6(neighbors[i]), use_column_width=True)
                    st.write(":red[Continue Browsing]")

        if START_INDEX + 40 < num_items:
            next_index = START_INDEX + 40
            st.button("Load More", on_click=lambda: st.experimental_set_query_params(start_index=str(next_index)))
        else:
            st.write("End of results")
          
    
          
    

elif selected == "About Us":
    st.image(logo1)
    st.title("About Sapphire")
    st.write("We are a clothing store that specializes in high-quality sustainable clothing at affordable prices. We aim to give customers a wider range of options because of our built in reccomendation system.")
    Aboutusimage=Image.open('rename.png')
    st.image(Aboutusimage)


elif selected == "Reviews":
    st.image(logo1)
    st.title("Customer Reviews")
    st.write("Here are what our wonderful customers have to say about us:")
    st.write (" Ethan Wang - I didn't know what to buy my mum for her birthday but this amazing k means algorithm helped me find the perfect gift !!!") 
    st.write ("Eileen - the product reccomendation on this website is amazing")
    st.write ("Joe Banks - I wanted to buy a t-shirt from this website but thanks to the product reccomendation system I ended up buying 4 t-shirts, 3 bags, 2 pairs of shoes and a pair of trousers")

elif selected == "Contact Us":
    st.image(logo1)
    st.title("Contact Us")
    st.write("You can contact us at the following email address:")
    st.write("contact@sapphireclothing.com")



# In[ ]:




