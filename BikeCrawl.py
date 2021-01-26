from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
import pymysql
import numpy as np
import datetime

'''
크롬 드라이버를 사용하는 이유는 작동을 직접보며 고칠수 있는 장점이 있다. 하지만 추후에 코드가 안정적으로 운행되면 PhantomJS 드라이버를 사용하는것을 추천한다
# 이 크롤링은 2018년에 진행 되었습니다
# 초기엔 API를 제공 되지 않은 관계로 맵에 표기 되어 있는 데이터를 수집하기 위해 사용 되었습니다.
# 크롬 드라이버를 다우로드 한후
# 저장되어있는 path를 설정한다
# 드라이버를 이용해 웹사이트를 불러온다
# 당시 스테이션은 약 1000개소 이었으며 크롬 드라이버는 한 싸이클당 20분이 걸린 반면 PhantomJs는 10분정도 걸렸다.
'''


def bike_crawl():
    imagenum = np.arange(10, 30) # 스테이션의 숫자에 따라 조정한다.
    k = ["img[%i]" % i for i in imagenum]
    driver = webdriver.Chrome(executable_path="path\\chromedriver.exe")
    wait = WebDriverWait(driver, 30)
    driver.get("https://www.bikeseoul.com/app/station/moveStationRealtimeStatus.do")
    time.sleep(5)
    for i in k:
        num_list = driver.find_element_by_xpath('//*[@id="mapDiv"]/div/div[1]/div[5]/' + str(i)) #제일 안정적으로 특정 위치를 찾는것은 xPath인것 같아서 사용했다.
        driver.execute_script("arguments[0].click();", num_list) # 해당 핀 포인트를 클릭하여 스크립트를 활성화한다. 그냥 클릭은 안된다
        date = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="currentDate"]'))).text #wait.until을 사용하는 이유는 로딩 타임을 주기위해서 이다.
        station_name = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="infoBox"]/p[1]/span'))).text
        dock_available = wait.until(
            EC.visibility_of_element_located((By.XPATH, '//*[@id="infoBox"]/ul/li[1]/span'))).text
        bike_available = wait.until(
            EC.visibility_of_element_located((By.XPATH, '//*[@id="infoBox"]/ul/li[2]/span'))).text
        conn = pymysql.connect(host='localhost', port=1111, user='user', passwd='pwd', db='sqltest', charset='utf8',
                               autocommit=True) #sql database에 연결하여 저장
        cur = conn.cursor()
        sql = 'INSERT INTO bikeseoul (`date`,`station_name`,`dock_available`,`bike_available`) VALUES ("%s","%s","%s","%s")' % (
        date, station_name, dock_available, bike_available) #데이터를 차례대로 넣는다.
        cur.execute(sql)
        time.sleep(0.5)
    conn.close()
    driver.close()

#마지막 날짜를 설정한다.
endTime = datetime.datetime.now() + datetime.timedelta(days=14)
while True:
    if datetime.datetime.now() >= endTime:
        break
    else:
        bike_crawl()
