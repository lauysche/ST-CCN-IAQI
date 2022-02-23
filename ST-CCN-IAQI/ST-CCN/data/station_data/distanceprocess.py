import pandas as pd
from geopy.distance import geodesic
'''
data = pd.read_csv('shanghai_Station.csv')
df = pd.read_csv('distance_.csv')
for i in range(0,9):
    l=[]
    for j in range(0,9):
        lat1 = data['lat'][i]
        lon1 = data['lon'][i]
        lat2 = data['lat'][j]
        lon2 = data['lon'][j]

        distance = geodesic((lat1,lon1),(lat2,lon2)).km
        #print("距离：{:.3f}km".format(distance))

        l.append(distance)
    data1 = pd.Series(pd.DataFrame(l)[0])
    df['{}'.format(i+1)]=data1
df.to_csv('distance__.csv')
'''
data = pd.read_csv('distance__.csv')
data1 = data['2'][0]
print(data1)

