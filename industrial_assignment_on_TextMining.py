# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:04:09 2024

@author: SAINATH
"""

'''
Problem Statement: -
In the era of widespread internet use, 
it is necessary for businesses to understand what
 the consumers think of their products. If they 
 can understand what the consumers like or 
 dislike about their products, they can improve 
 them and thereby increase their profits by 
 keeping their customers happy. For this reason,
 they analyze the reviews of their products on 
 websites such as Amazon or Snapdeal by using 
 text mining and sentiment analysis techniques. 


Task 1:
1.	Extract reviews of any product from e-commerce 
website Amazon.
2.	Perform sentiment analysis on this extracted 
data and build a unigram and bigram word cloud. 

Task 2:
1.	Extract reviews for any movie from IMDB and 
perform sentiment analysis.

Task 3: 
1.	Choose any other website on the internet and
do some research on how to extract text and perform sentiment analysis
'''
#task1
from bs4 import BeautifulSoup as bs
import requests
link='https://www.amazon.in/gp/aw/d/B0CZ6V978P/?_encoding=UTF8&pd_rd_plhdr=t&aaxitk=eba92336a77184e57ec4d419882f9bd1&hsa_cr_id=0&qid=1726426803&sr=1-1-e0fa1fdd-d857-4087-adda-5bd576b25987&ref_=sbx_be_s_sparkle_sccd_asin_0_rating&pd_rd_w=0EWa8&content-id=amzn1.sym.df9fe057-524b-4172-ac34-9a1b3c4e647d%3Aamzn1.sym.df9fe057-524b-4172-ac34-9a1b3c4e647d&pf_rd_p=df9fe057-524b-4172-ac34-9a1b3c4e647d&pf_rd_r=CCKA8641E46KMP1XV6R6&pd_rd_wg=AmjRr&pd_rd_r=e058a420-6e38-4f8e-a134-7b83f799efc8&th=1'
page=requests.get(link)
#connection is establish successfully
page.content
#you will get all html source code but very crowdy text
#let us apply html parser
soup=bs(page.content,'html.parser')
soup
#now the text is clean but not upto the expectation
#now let us apply prettify method
print(soup.prettify())
title=soup.find_all(class_='title')
title
review_title=[]
for i in range(0,len(title)):
    review_title.append(title[i].get_text())
review_title
review_title[:]=[title.strip('\n') for title in review_title]
review_title
len(review_title)
review=soup.find_all('span',class_="a-size-base-review-text review-text-content")
review
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
len(review_body)

#Task2
from bs4 import BeautifulSoup as bs
import requests
link='https://www.imdb.com/title/tt11531182/reviews/?ref_=tt_ov_rt'
page=requests.get(link)
#connection is establish successfully
page.content
#you will get all html source code but very crowdy text
#let us apply html parser
soup=bs(page.content,'html.parser')
soup
print(soup.prettify())
title=soup.find_all('a',class_='title')
title
review_title=[]
for i in range(0,len(title)):
    review_title.append(title[i].get_text())
review_title
review_title[:]=[title.strip('\n') for title in review_title]
review_title
len(review_title)
review=soup.find_all('div',class_="text")
review
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
len(review_body)

#task3
from bs4 import BeautifulSoup as bs
import requests
link='https://www.tatamotors.com/careers/'
page=requests.get(link)
#connection is establish successfully
page.content
#you will get all html source code but very crowdy text
#let us apply html parser
soup=bs(page.content,'html.parser')
soup
print(soup.prettify())
title=soup.find_all('p',class_='text-white appearIntro')
title
review_title=[]
for i in range(0,len(title)):
    review_title.append(title[i].get_text())
review_title
review_title[:]=[title.strip('\n') for title in review_title]
review_title
len(review_title)
from textblob import TextBlob
# Assuming page_text contains the text extracted from the website
analysis = TextBlob(page)
# Perform sentiment analysis
sentiment = analysis.sentiment
# Print the results (polarity: -1 to 1, subjectivity: 0 to 1)
print(f'Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}')
