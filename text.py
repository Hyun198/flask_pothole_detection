import os
import pandas as pd
import natsort
import matplotlib.pyplot as plt
import gpxpy
import gpxpy.gpx
plt.rcParams['axes.spines.top'] =False
plt.rcParams['axes.spines.right'] =False



path1 = './frames'
path2="C:/Users/user-pc/Desktop/visual/project/static/gpx/AnyConv.com__iPassBlack_GPS_Data_.gpx"

file_list = os.listdir(path1)
file_list=natsort.natsorted(file_list)


name_info = []

for file in file_list:
        name_info.append(path1+'/'+file)

name_df = pd.DataFrame(name_info)


my_file=path2
with open(my_file, 'r', encoding='utf-8') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

    route_info = []

    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_info.append({'GPSlatitude':point.latitude,'GPSlongitude':point.longitude})
    

route_df = pd.DataFrame(route_info)

#이미지 파일이랑 좌표 합치기
hab_df=pd.concat([name_df,route_df],axis=1).fillna(0)

hab_df.rename(columns = {0:'image'}, inplace=True)

print(hab_df)

hab_df.to_csv('static/csv/hab_df.csv',index=False)


#df1 = pd.read_csv('static/csv/hab_df.csv')
#print(df1.head(), encoding='utf-8')