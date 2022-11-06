from fileinput import filename
from importlib.resources import path
import pandas as pd
import folium
import os
import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
plt.rcParams['axes.spines.top'] =False
plt.rcParams['axes.spines.right'] =False


#path="C:/Users/user-pc/Desktop/visual/project/static/gpx/AnyConv.com__iPassBlack_GPS_Data_.gpx"

def read_gpx(my_file):
    with open(my_file, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    route_info = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_info.append({'GPSlatitude':point.latitude,'GPSlongitude':point.longitude})
    #print(route_info)
    route_df = pd.DataFrame(route_info)
    #print(route_df)
    route_df.to_csv('static/csv/route_df.csv',index=False)
    

if __name__ == "__main__":
    read_gpx()