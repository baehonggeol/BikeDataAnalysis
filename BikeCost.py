import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
'''
이 코드는 서비스를 재배치 서비스를 진행 했을때 리워드로 나가야 할 금액과 할당양을 알아 보기 위한 것입니다.

'''
# overstack mission must sent out when bike is stacked over the station capacity
# understacked happens when it's below 25% and it;s to fill the bike upto recommended status



bike = pd.read_csv('bikedata.csv')#데이터
# dropping nan from data
bike = bike.dropna()

# data type 변환
bike['stack'] = bike['stack'].astype('category')
date_format = '%y%m%d %H%M%S'
bike['time'] = pd.to_datetime(bike['time'], format=date_format)

# dropping testing sites and outliers
indexNames = bike[(bike['stack'] == '1084. 윤선생빌딩(JYP사옥)') | (bike['stack'] == '1309. 보문3교 옆') |
                  (bike['stack'] == '99998. 상암단말정비') | (bike['stack'] == '9996. 시설2')
                  | (bike['stack'] == '1687. 서울월드컵경기장 테스트')
                  | (bike['stack'] == '132. 창천문화공원')
                  | (bike['stack'] == '위트콤') | (bike['stack'] == '위트콤공장')].index
bike.drop(indexNames, inplace=True)

# getting a list of all stations in the data
stationlist = bike['stack'].unique()

# 재배치 서비스가 가능한 시간만 가지고 오기
slc = bike[bike['time'].dt.hour.isin([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])]

cost = pd.DataFrame(columns=['stationname', 'UNR', 'OVR', 'Tot', 'cost'])

starttime = slc.iloc[0]['time']
# 두시간 후의 상황을 잘 예측 했을경우 서비스 cost 계산을 하는 반복문이다.
for i in stationlist:
    ma = slc[slc['stack'] == i]
    k = round(ma['maxBike'][0] * 0.25, 0)
    un = list()
    ov = list()
    while starttime < ma.iloc[-1]['time']:
        endtime = starttime + dt.timedelta(hours=2)
        m = ma[ma['time'].between(starttime, endtime)]
        if m.iloc[-1]['Cbike'] < k:
            l = k - m.iloc[-1]
            un.append(l)
        elif m.iloc[-1]['Cbike'] >= ma['maxBike'][0]:
            d = m.iloc[-1]['Cbike'] - k
            ov.append(d)
        else:
            continue
        starttime += dt.timedelta(hours=2)
    na = i
    uu = sum(un)
    ovv = sum(ov)
    too = uu + ovv
    coo = too * 4000
    ad = pd.Series([na, uu, ovv, too, coo], index=['stationname', 'UNR', 'OVR', 'Tot', 'cost'])
    cost = cost.append(ad, ignore_index=True)
    starttime = slc.iloc[0]['time']

starttime += dt.timedelta(hours=2)
endtime = starttime + dt.timedelta(hours=2)




