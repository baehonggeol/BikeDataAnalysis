import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import re
import sys
import datetime as dt
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import folium

#############################################################
#데이터가 모두 날짜별 데이터로 txt파일로 저장이 되어 있어 하나의 데이테로 합치는 구간
# 데이터 가지고 오기
path = r'path\bikedata'  # use your path
all_files = glob.glob(os.path.join(path + "\*.txt"))
f_from_each_file = (
pd.read_csv(f, sep="\t", lineterminator="\r", names=['latlong', 'stack', 'maxBike', 'Cbike', 'addr']) for f in all_files)

# combining entire data into a single series of data
concatenated_df = pd.concat(f_from_each_file, ignore_index=True)
# empty series
time = pd.Series([])
# getting the time from files title
regex = re.compile(r'\d\d\d\d\d\d \d\d\d\d\d\d') #파일 이름을 날짜 시간 포멧 변경을 위한 작업
# combining entire data with time in its final row
for i in all_files:
    dd = pd.read_csv(i, sep="\t", lineterminator="\r", names=['latlong', 'stack', 'maxBike', 'Cbike', 'addr'])
    dc = regex.search(i)
    da = dc.group()
    mm = pd.Series([da] * len(dd))
    time = time.append(mm, ignore_index=True)

# combining list of date into a single row
concatenated_df['time'] = time
# combining into big chunk
bike = concatenated_df
# fileout to csv for faster data call
bike.to_csv("bikedata.csv", index=False)

#############################################################################
'''
데이터를 본격 분석하기 위해서 데이터 마지막 정제 작업
'''

# graph 한글 깨짐 해소
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
# 합친 데이터 가지고 오기
bike = pd.read_csv('bikedata.csv')
# dropping nan from data
bike = bike.dropna()

# data type 변환
bike['stack'] = bike['stack'].astype('category')
date_format = '%y%m%d %H%M%S'
bike['time'] = pd.to_datetime(bike['time'], format=date_format)

# dropping testing sites and outliers
# 밑에 명시된 대여소는 테스트 중이거나 아예 다른 영역으로 운영적인 곳이라서 제외
indexNames = bike[(bike['stack'] == '1084. 윤선생빌딩(JYP사옥)') | (bike['stack'] == '1309. 보문3교 옆') |
                  (bike['stack'] == '99998. 상암단말정비') | (bike['stack'] == '9996. 시설2')
                  | (bike['stack'] == '1687. 서울월드컵경기장 테스트')
                  | (bike['stack'] == '132. 창천문화공원')
                  | (bike['stack'] == '위트콤') | (bike['stack'] == '위트콤공장')].index
bike.drop(indexNames, inplace=True)

# getting a list of all stations in the data
stationlist = bike['stack'].unique()

mama = pd.DataFrame(columns=['latlong', 'stationname', 'understacked', 'overstacked', 'stable'])


# calculation all the overstack and understack for each station
for i in stationlist:
    name = i
    be = bike[bike['stack'] == i]
    la = be.iloc[0][0]
    os = be[be['Cbike'] >= be['maxBike'].iloc[0]].shape[0]
    us = be[be['Cbike'] < (be['maxBike'].iloc[0] * 0.25)].shape[0]
    sta = len(be) - (os + us)
    ad = pd.Series([la, name, us, os, sta], index=['latlong', 'stationname', 'understacked', 'overstacked', 'stable'])
    mama = mama.append(ad, ignore_index=True)

# 보고를 위한 저장은 엑셀로 하는 편입니다.
mama.to_excel("balanceofsta.xls", index=False)
mama['understacked'] = mama['understacked'].astype('float')
mama['overstacked'] = mama['overstacked'].astype('float')
mama['stable'] = mama['stable'].astype('float')

# getting the stations status for each
'''
각 스테이션 마다의 제고 유지 상태를 보기 위한 코드
'''
mama['status'] = mama[['understacked', 'overstacked', 'stable']].idxmax(axis=1)
# getting the overview of the station status
# stable = 868
# understacked = 614
# overstacked = 71
# total = 1553


# making pie graph for each station
for i in mama.index:
    labels = mama.columns.values[1:4]
    sizes = mama.iloc[i][1:4]
    explode = ([0, 0, 0])
    max_value = max(sizes)
    max_index = sizes[0:3].values.tolist().index(max_value)
    explode[max_index] = 0.1
    colors = ['#C0392B', '#2E86C1', '#2ECC71']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title(mama.iloc[i][0])
    plt.tight_layout()
    plt.show()
    fig1.savefig((mama.iloc[i][0] + '.png'), bbox_inches='tight')

# making a boxplot for each station
for i in stationlist:
    b1 = bike[bike['stack'] == i]
    fig1, ax1 = plt.subplots()
    ax1.boxplot(b1["Cbike"])
    plt.title(b1.iloc[0][1])
    plt.axhline(y=b1.iloc[0][2], color='red', linestyle='--')
    plt.tight_layout()
    plt.show()
    fig1.savefig(b1.iloc[0][1] + '.png')

# map visualization\
m = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='Stamen Terrain')
# folium.Marker(location=[37.5502, 126.982], popup="Marker A",
#             icon=folium.Icon(icon='cloud')).add_to(m)

# 맵 위에 핀 포인트 만들기
for i in range(len(mama)):
    if mama.iloc[i]['status'] == 'understacked':
        folium.Marker(location=tuple(map(float, mama.iloc[i]['latlong'].split(',')))
                      , popup=mama.iloc[i]['stationname'],
                      icon=folium.Icon(color='red', icon='ok'), ).add_to(m)
    elif mama.iloc[i]['status'] == 'overstacked':
        folium.Marker(location=tuple(map(float, mama.iloc[i]['latlong'].split(',')))
                      , popup=mama.iloc[i]['stationname'],
                      icon=folium.Icon(color='blue', icon='ok'), ).add_to(m)
        folium.Circle(location=tuple(map(float, mama.iloc[i]['latlong'].split(','))),
                      radius=1500, color='crimson').add_to(m)
    else:
        folium.Marker(location=tuple(map(float, mama.iloc[i]['latlong'].split(',')))
                      , popup=mama.iloc[i]['stationname'],
                      icon=folium.Icon(color='green', icon='ok'), ).add_to(m)

m.save('map2.html')

# 군집형 맵 만들기
# cluster method
m.save('map.html')
from folium.plugins import MarkerCluster

m = folium.Map(location=[37.5502, 126.982], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)
mama.iloc[0]['latlong']
res = tuple(map(float, mama.iloc[0]['latlong'].split(',')))

for i in range(len(mama)):
    if mama.iloc[i]['status'] == 'understacked':
        folium.Marker(location=tuple(map(float, mama.iloc[i]['latlong'].split(',')))
                      , popup=mama.iloc[i]['stationname'],
                      icon=folium.Icon(color='red', icon='ok'), ).add_to(marker_cluster)
    elif mama.iloc[i]['status'] == 'overstacked':
        folium.Marker(location=tuple(map(float, mama.iloc[i]['latlong'].split(',')))
                      , popup=mama.iloc[i]['stationname'],
                      icon=folium.Icon(color='blue', icon='ok'), ).add_to(marker_cluster)
        folium.Circle(location=tuple(map(float, mama.iloc[i]['latlong'].split(','))),
                      radius=1500, color='crimson').add_to(marker_cluster)
    else:
        folium.Marker(location=tuple(map(float, mama.iloc[i]['latlong'].split(',')))
                      , popup=mama.iloc[i]['stationname'],
                      icon=folium.Icon(color='green', icon='ok'), ).add_to(marker_cluster)

m.save('map3.html')

'''
관리가 제일 안되는 스테이션 탑 4를 알아보기
'''
# sorting understacked
unde = mama[mama['status'] == 'understacked'].sort_values(['understacked'], ascending=[False])
unde.head(4)
# 2242.양재역 11번 출구 앞
# 580. 신금호역 3번출구 뒤
# 2175. 신림동걷고싶은문화의거리입구
# 1513. 강북구청 뒷편

# sorting overstacked
ovde = mama[mama['status'] == 'overstacked'].sort_values(['overstacked'], ascending=[False])
ovde.head(4)
# 593.자양중앙나들목
# 931. 역촌파출소
# 286. 우성아파트 교차로
# 954. 은평뉴타운구파발9단지

# 특정 데이터를 보기 위한 코드
yang = bike[bike['stack'] == '2242.양재역 11번 출구 앞']
yang.shape
ma = yang[yang['time'].between('2019-08-29 09:00', '2019-08-29 22:00')]
ma.shape
us = ma[ma['Cbike'] < (ma['maxBike'].iloc[0] * 0.25)]
us['time']

#########
# 송파 적합성 평가
'''
송파 지역이 서비스를 시작하기 좋은 지점인지 적합성을 따지기 위한 분석
'''
# 위 방법을 이용한 후 송파 지역만 subset을 하여 분류함
song = pd.read_excel('songpa.xlsx')
# 대여소 99개

# 시작지점 명시
m = folium.Map(location=[37.5502, 126.982], zoom_start=12)

for i in range(len(song)):
    folium.Marker(location=[song.iloc[i]['위도'], song.iloc[i]['경도']]
                  , popup=song.iloc[i]['대여소명'],
                  icon=folium.Icon(color='blue', icon='ok'), ).add_to(m)
m.save('songpa.html')

#############
# 종로 적합성 평가
# 대여소 111개
# 62개 운영
jong = pd.read_excel('jong.xlsx')
jong.dtypes

m = folium.Map(location=[37.5502, 126.982], zoom_start=12)
jong.iloc[1]['대여소명']

for i in range(len(jong)):
    folium.Marker(location=[jong.iloc[i]['위도'], jong.iloc[i]['경도']]
                  , popup=jong.iloc[i]['대여소명'],
                  icon=folium.Icon(color='blue', icon='ok'), ).add_to(m)

m.save('mapjong.html')
# 종로 아닌 대여소가 있는지 확인하는 작업
da = pd.DataFrame()
for i in jong['대여소명']:
    k = mama[mama['stationname'].str.contains(i)]
    if len(k) == 0:
        print(i)
    else:
        da = da.append(k)

da = da.append(k)
mama[mama['latlong'] == str(jong.iloc[0]['위도']) + ',' + str(jong.iloc[0]['경도'])]

# mapping for jongro
m = folium.Map(location=[37.5502, 126.982], zoom_start=12)
# folium.Marker(location=[37.5502, 126.982], popup="Marker A",
#             icon=folium.Icon(icon='cloud')).add_to(m)

for i in range(len(da)):
    if da.iloc[i]['status'] == 'understacked':
        folium.Marker(location=tuple(map(float, da.iloc[i]['latlong'].split(',')))
                      , popup=da.iloc[i]['stationname'],
                      icon=folium.Icon(color='red', icon='ok'), ).add_to(m)
    elif da.iloc[i]['status'] == 'overstacked':
        folium.Marker(location=tuple(map(float, da.iloc[i]['latlong'].split(',')))
                      , popup=da.iloc[i]['stationname'],
                      icon=folium.Icon(color='blue', icon='ok'), ).add_to(m)
        folium.Circle(location=tuple(map(float, da.iloc[i]['latlong'].split(','))),
                      radius=1500, color='crimson').add_to(m)
    else:
        folium.Marker(location=tuple(map(float, da.iloc[i]['latlong'].split(',')))
                      , popup=da.iloc[i]['stationname'],
                      icon=folium.Icon(color='green', icon='ok'), ).add_to(m)

m.save('mapjon1.html')

# separation of the jongro station from main data
km = pd.DataFrame()
for i in da['stationname']:
    dss = bike[bike['stack'] == i]
    if len(dss) == 0:
        print(i)
    else:
        km = km.append(dss)


jstationlist = km['stack'].unique()

# 총거치대수
# 736
gk = list()
for i, k in enumerate(jstationlist):
    qq = km[km['stack'] == k]
    gk.append(qq.iloc[i]['maxBike'])

jtime = km['time'].unique()

# getting the entire bike change for jongro
# 종로 안에 대여 되는 자전거의 총 움직임과 사용시간을 정리하는 코드
ted = pd.DataFrame(columns=['time', 'Tbike', 'length'])

for i in jtime:
    time = i
    be = km[km['time'] == i]
    le = len(be)
    su = np.sum(be['Cbike'])
    ad = pd.Series([time, su, le], index=['time', 'Tbike', 'length'])
    ted = ted.append(ad, ignore_index=True)

# 데이터 길이 맥스 63 미니멈 59
max(ted['length'])
min(ted['length'])

# 자전거 유동이 많은 날과 적은 날
# 829개 max 2019-12-05 16:50 날씨 맑음
# 5개 min  2019-09-21 01:10
max(ted['Tbike'])
min(ted['Tbike'])

ted[ted['Tbike'] == 829]

ma = ted[ted['time'].between('2019-12-02 09:00', '2020-01-15 22:00')]
np.average(ma['Tbike'])

### line graph
g = sns.relplot(x="time", y="Tbike", kind="line", data=ted)
g.fig.autofmt_xdate()


def Average(lst):
    return sum(lst) / len(lst)

Average(ted['Tbike'])

# boxplot
fig1, ax1 = plt.subplots()
ax1.boxplot(ted["Tbike"])
plt.tight_layout()
plt.show()
fig1.savefig(b1.iloc[0][1] + '.png')

d = km[km['time'] == km.iloc[0]['time']]
np.sum(d['Cbike'])

# classification of jongro stations
# 각 대여소 별 상태 데이터 만들기
jongst = pd.DataFrame(columns=['latlong', 'stationname', 'understacked', 'overstacked', 'stable'])
for i in jstationlist:
    name = i
    be = km[km['stack'] == i]
    la = be.iloc[0][0]
    os = be[be['Cbike'] >= be['maxBike'].iloc[0]].shape[0]
    us = be[be['Cbike'] < (be['maxBike'].iloc[0] * 0.25)].shape[0]
    sta = len(be) - (os + us)
    ad = pd.Series([la, name, us, os, sta], index=['latlong', 'stationname', 'understacked', 'overstacked', 'stable'])
    jongst = jongst.append(ad, ignore_index=True)

#데이터 타입 맞추기
jongst['understacked'] = jongst['understacked'].astype('float')
jongst['overstacked'] = jongst['overstacked'].astype('float')
jongst['stable'] = jongst['stable'].astype('float')

# getting the stations status for each
jongst['status'] = jongst[['understacked', 'overstacked', 'stable']].idxmax(axis=1)

jongst['status'].value_counts()
# 38 us
# 22 st
# 2 ov


for i in jongst.index:
    labels = jongst.columns.values[2:5]
    sizes = jongst.iloc[i][2:5]
    explode = ([0, 0, 0])
    max_value = max(sizes)
    max_index = sizes[0:3].values.tolist().index(max_value)
    explode[max_index] = 0.1
    colors = ['#C0392B', '#2E86C1', '#2ECC71']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title(jongst.iloc[i][1])
    plt.tight_layout()
    plt.show()
    fig1.savefig((jongst.iloc[i][1] + '.png'), bbox_inches='tight')

# 2 overstacked station analysis
ovs = km[km['stack'] == '359. 원남동사거리']
ovs1 = km[km['stack'] == '312. 시청역 1번출구 뒤']
Average(ovs1['Cbike'])
max(ovs1['Cbike'])
min(ovs1['Cbike'])

ovs1['maxBike']

#############

# 송파 적합성 평가
# 대여소 99개
# 47개 운영
'''
송파 지역의 반만 적합성 평가를 진행
위 코드와 거의 비슷함
'''
song = pd.read_excel('songpa.xlsx')
song.dtypes
len(song)
m = folium.Map(location=[37.5502, 126.982], zoom_start=12)
song.iloc[1]['대여소명']

for i in range(len(song)):
    folium.Marker(location=[song.iloc[i]['위도'], song.iloc[i]['경도']]
                  , popup=song.iloc[i]['대여소명'],
                  icon=folium.Icon(color='blue', icon='ok'), ).add_to(m)

m.save('mapsong.html')

da = pd.DataFrame()
for i in song['대여소명']:
    k = mama[mama['stationname'].str.contains(i)]
    if len(k) == 0:
        print(i)
    else:
        da = da.append(k)

k = mama[mama['stationname'].str.contains('1019.')]
k['stationname']
f1 = mama[mama['stationname'].str.contains(jong.iloc[0]['대여소명'])]

da = da.append(k)
len(da)
mama[mama['latlong'] == str(jong.iloc[0]['위도']) + ',' + str(jong.iloc[0]['경도'])]

mama.iloc[0]['latlong']
str(jong.iloc[0]['위도']) + ',' + str(jong.iloc[0]['경도'])
# mapping for 송파
m = folium.Map(location=[37.5502, 126.982], zoom_start=12)
# folium.Marker(location=[37.5502, 126.982], popup="Marker A",
#             icon=folium.Icon(icon='cloud')).add_to(m)

for i in range(len(da)):
    if da.iloc[i]['status'] == 'understacked':
        folium.Marker(location=tuple(map(float, da.iloc[i]['latlong'].split(',')))
                      , popup=da.iloc[i]['stationname'],
                      icon=folium.Icon(color='red', icon='ok'), ).add_to(m)
    elif da.iloc[i]['status'] == 'overstacked':
        folium.Marker(location=tuple(map(float, da.iloc[i]['latlong'].split(',')))
                      , popup=da.iloc[i]['stationname'],
                      icon=folium.Icon(color='blue', icon='ok'), ).add_to(m)
        folium.Circle(location=tuple(map(float, da.iloc[i]['latlong'].split(','))),
                      radius=1500, color='crimson').add_to(m)
    else:
        folium.Marker(location=tuple(map(float, da.iloc[i]['latlong'].split(',')))
                      , popup=da.iloc[i]['stationname'],
                      icon=folium.Icon(color='green', icon='ok'), ).add_to(m)

m.save('mapson1.html')

# separation of the jongro station from main data
km = pd.DataFrame()
for i in da['stationname']:
    dss = bike[bike['stack'] == i]
    if len(dss) == 0:
        print(i)
    else:
        km = km.append(dss)

km
qq = km[km['stack'] == jstationlist[0]]
qq.iloc[0]['maxBike']
jstationlist = km['stack'].unique()
# 총거치대수
# 582
gk = list()
np.sum(gk)
for i, k in enumerate(jstationlist):
    qq = km[km['stack'] == k]
    gk.append(qq.iloc[i]['maxBike'])

jtime = km['time'].unique()

# getting the entire bike change for songpa
ted = pd.DataFrame(columns=['time', 'Tbike', 'length'])

for i in jtime:
    time = i
    be = km[km['time'] == i]
    le = len(be)
    su = np.sum(be['Cbike'])
    ad = pd.Series([time, su, le], index=['time', 'Tbike', 'length'])
    ted = ted.append(ad, ignore_index=True)

ted.head()
km.dtypes

# 데이터 길이 맥스 47 미니멈 43
max(ted['length'])
min(ted['length'])

# 자전거수
# 829개 max 2019-12-05 16:50 날씨 맑음
# 5개 min  2019-09-21 01:10
max(ted['Tbike'])
min(ted['Tbike'])

ted[ted['Tbike'] == 829]

# average 302대
ma = ted[ted['time'].between('2019-12-02 09:00', '2020-01-15 22:00')]
Average(ma['Tbike'])

### line graph
g = sns.relplot(x="time", y="Tbike", kind="line", data=ted)
g.fig.autofmt_xdate()


def Average(lst):
    return sum(lst) / len(lst)


Average(ted['Tbike'])

# boxplot
fig1, ax1 = plt.subplots()
ax1.boxplot(ted["Tbike"])
plt.tight_layout()
plt.show()
fig1.savefig(b1.iloc[0][1] + '.png')

d = km[km['time'] == km.iloc[0]['time']]
np.sum(d['Cbike'])

# classification of jongro stations
songst = pd.DataFrame(columns=['latlong', 'stationname', 'understacked', 'overstacked', 'stable'])
for i in jstationlist:
    name = i
    be = km[km['stack'] == i]
    la = be.iloc[0][0]
    os = be[be['Cbike'] >= be['maxBike'].iloc[0]].shape[0]
    us = be[be['Cbike'] < (be['maxBike'].iloc[0] * 0.25)].shape[0]
    sta = len(be) - (os + us)
    ad = pd.Series([la, name, us, os, sta], index=['latlong', 'stationname', 'understacked', 'overstacked', 'stable'])
    songst = songst.append(ad, ignore_index=True)

songst.dtypes
songst['understacked'] = songst['understacked'].astype('float')
songst['overstacked'] = songst['overstacked'].astype('float')
songst['stable'] = songst['stable'].astype('float')

# getting the stations status for each
songst['status'] = songst[['understacked', 'overstacked', 'stable']].idxmax(axis=1)

songst['status'].value_counts()
# 20 us
# 20 st
# 5 ov

for i in songst.index:
    labels = songst.columns.values[2:5]
    sizes = songst.iloc[i][2:5]
    explode = ([0, 0, 0])
    max_value = max(sizes)
    max_index = sizes[0:3].values.tolist().index(max_value)
    explode[max_index] = 0.1
    colors = ['#C0392B', '#2E86C1', '#2ECC71']
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title(songst.iloc[i][1])
    plt.tight_layout()
    plt.show()
    fig1.savefig((songst.iloc[i][1] + '.png'), bbox_inches='tight')

# 5 overstacked station analysis
ss = songst[songst['status'] == 'overstacked']
s = bike[bike['stack'] == '2619. 석촌고분역 4번출구']
max(s['Cbike'])

# 2019-11-09 5시30분
s[s['Cbike'] == 71]

for i in ss['stationname'].to_list():
    s = bike[bike['stack'] == i]
    print(i, round(Average(s['Cbike']), 2), max(s['Cbike']), min(s['Cbike']))

# 1212. 송파역 2번 출구앞, 9.6, 44, 0
# 1221. 삼전사거리 포스코더샵, 10.12, 48, 0
# 1249. 아주중학교건너편, 8.8, 37, 0
# 2619. 석촌고분역 4번출구, 12.4, 71, 0
# 1019. 다성이즈빌아파트(호원대 대각선 맞은편) 10.32, 41, 0




