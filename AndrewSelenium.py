#opens most recent race tip sheet for CTX
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Firefox()
driver.get("http://www.guaranteedtipsheet.com/login.asp")
print 'page name is', '"' + driver.title + '"'

username = driver.find_element_by_name('txtusername')
username.send_keys("shuda")
password = driver.find_element_by_name('txtpassword')
password.send_keys("gandalf73")
password.send_keys(Keys.RETURN)

time.sleep(1)
past_results = driver.find_element_by_link_text('Past Results')
past_results.click()

list_of_tracks = driver.find_elements_by_link_text('view details')

for idx, track in enumerate(list_of_tracks):
    if idx == 27: #28th race on page is 'Charles Town'
        track.send_keys(Keys.COMMAND + Keys.RETURN)

body = driver.find_element_by_xpath('/html/body')
body.send_keys(Keys.COMMAND + 'w')

time.sleep(2)
race = driver.find_element_by_link_text('view details')
race.click()