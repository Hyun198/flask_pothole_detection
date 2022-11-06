import base64
import csv
import folium
from folium import IFrame
from matplotlib import image
import pandas as pd


path = 'C:/Users/user-pc/Desktop/visual/project/static/csv/route_df.csv'

def read_csv(file=path):
    route_df = pd.read_csv(file)
    route_df.head()
    m = folium.Map([37.6292,126.7295],zoom_start=16)

    #image = list(hab_df["image"])
    
    #for img in (image):
        #encoded = base64.b64encode(open(img,'rb').read()).decode()
       # html = f'''
        #        <img src="data:image/jpeg;base64,{encoded}">
       # '''
    #iframe = IFrame(html, width=300, height=300)
    #popup = folium.Popup(iframe, max_width=650)


    coordinates = [tuple(x) for x in route_df[['GPSlatitude','GPSlongitude']].to_numpy()]
    folium.PolyLine(coordinates, weight = 10).add_to(m)


    m.add_child(folium.LatLngPopup())
    m.add_child(folium.ClickForMarker())

    #for index, row in route_df.iterrows():
       # folium.Marker([row['GPSlatitude'], row['GPSlongitude']]).add_to(m)

    m.save('templates/map.html') 

if __name__ =='__main__':
    read_csv()