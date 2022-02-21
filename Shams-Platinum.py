import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# In[4]:


def pre_process(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 1)

    edge = cv2.Canny(blur, 135, 200)

    kernal = np.ones((5, 5))

    dilate = cv2.dilate(edge, kernal, iterations=2)

    threshold = cv2.erode(dilate, kernal, iterations=1)

    return threshold


# In[5]:


def get_contours(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    points = cv2.drawContours(img_contour, biggest, -1, (255, 0, 0), 20)
    return biggest


# In[6]:


def reshape(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points


# In[7]:


def fix_image(img, contour):
    contour = reshape(contour)

    pts1 = np.float32(contour)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # get the warp perspective transform
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # apply to the image
    final_img = cv2.warpPerspective(img, matrix, (width, height))

    #crop the image
    cropped = final_img[20:final_img.shape[0] - 20, 20:final_img.shape[1] - 20]

    return cropped


# In[8]:

filename = r"D:\Platinum Project\BrilliantAC.jpg"
img = cv2.imread(filename)
print(img.shape)
img_contour = img.copy()



scale_percent = 100
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

threshold = pre_process(img)
get_contour = get_contours(threshold)
fixed_image = fix_image(img, get_contour)

# extract text from image using tesseract

text = pytesseract.image_to_string(img)
print('text detected: \n' + text)

cv2.imshow('img', img)
cv2.imshow('fixed_image', fixed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


text


# In[10]:


import spacy
nlp = spacy.load('en_core_web_sm') 
doc  = nlp(text)


# In[11]:


doc


# In[12]:


for token in doc:
    print(token)


# In[13]:


for sent in doc.sents:
    print(sent)


# In[14]:


for entity in doc.ents:
    print(f"{entity.text:>35} {entity.label_}")


# In[15]:


ruler = nlp.add_pipe('entity_ruler' , before = 'ner')


# In[16]:


patterns = [
    {'label': 'COMPANY' , 'pattern' : 'Brilliant Air Conditioners PTY LTD.'},
    {'label': 'DATE' , 'pattern' : '03 FEB 2022'},
    {'label': 'DATE' , 'pattern' : '15 February 2022'},
    {'label': 'DATE' , 'pattern' : '2022/03/05'},
    {'label': 'EMAIL' , 'pattern' : 'sophiew@brilliantac.com'},
    {'label': 'PHONE' , 'pattern' : '082 221 1549'},
    
]


# In[17]:


ruler.add_patterns(patterns)


# In[18]:


doc  = nlp(text)


# In[19]:


for entity in doc.ents:
    print(f"{entity.text:>35} {entity.label_}")


# In[20]:


from spacy.matcher import Matcher


# In[21]:


matcher = Matcher(nlp.vocab)


# In[22]:


pattern1 = [
    {"IS_DIGIT":True},
    {"IS_ALPHA":True},
    {"IS_DIGIT":True},
    {"IS_ALPHA":True},
    {"LENGTH":10 , 'OP':'+'}
]



matcher.add("dates_between", [pattern1], greedy='LONGEST')


# In[23]:



pattern2 = [
    {"IS_DIGIT":True},
    {"IS_ALPHA":True},
    {"IS_DIGIT":True},
    {"ORTH":"."}
]


matcher.add("document_date", [pattern2])


# In[24]:


pattern3 = [
    {"POS":'VERB'},
    {"IS_ALPHA":True},
    {"IS_ALPHA":True},
    {"POS":'ADP'},
    {"IS_DIGIT":True},
    {"IS_DIGIT":True},
    {"IS_DIGIT":True},
    {"ORTH":"."}
]


matcher.add("contact_person", [pattern3])


# In[25]:


pattern4 = [
    {"LIKE_EMAIL":True},
]

matcher.add("contact_email", [pattern4])


# In[26]:


lemmas = ['guarantee']


pattern5 = [
    {"POS":'VERB', 'LEMMA':{'IN' : lemmas}},
    {"POS":'ADP'},
    {"IS_DIGIT":True}

]


matcher.add("guaranteed", [pattern5] )


# In[27]:


pattern6 = [
    {"IS_DIGIT":True},

    {"IS_DIGIT":True},

    {"IS_DIGIT":True}

]


matcher.add("contact_number", [pattern6])


# In[28]:


matches = matcher(doc)


# In[29]:


matches


# In[30]:


for match in matches:
    print(match, doc[match[1]:match[2]])


# In[31]:


for i in range(0 , len(matches)):
    index = matches[i]
    tag = nlp.vocab[matches[i][0]].text
    text2 = doc[matches[i][1]:matches[i][2]]
    
    print(f'{tag}: {text2}')


# In[32]:


for entity in doc.ents:
    if entity.label_ == 'COMPANY':
        print(f'company_name:{entity.text}')
        
    if entity.label_ == 'GPE':
        print(f'locatoin:{entity.text}') 
        

for i in range(0 , len(matches)):
    index = matches[i]
    tag = nlp.vocab[matches[i][0]].text
    text2 = doc[matches[i][1]:matches[i][2]]
    
    print(f'{tag}: {text2}')
                


# In[33]:


for entity in doc.ents:
    if entity.label_ == 'COMPANY':
        company_name = entity.text
    if entity.label_ == 'GPE':
        location = entity.text


# In[34]:


for match in matches[0:10]:
    print(nlp.vocab[match[0]].text)
    


# In[35]:


for match in matches[0:10]:
    if nlp.vocab[match[0]].text == 'document_date':
        document_date= doc[match[1]:match[2]]
        
    if nlp.vocab[match[0]].text == 'contact_number':
        contact_number= doc[match[1]:match[2]]
        
    if nlp.vocab[match[0]].text == 'contact_person':
        contact_person= doc[match[1]:match[2]][1:3]
        
    if nlp.vocab[match[0]].text == 'contact_email':
        contact_email= doc[match[1]:match[2]]
        
    if nlp.vocab[match[0]].text == 'guaranteed':
        guaranteed= doc[match[1]:match[2]][2:3]
        
    if nlp.vocab[match[0]].text == 'dates_between':
        dates_between= doc[match[1]:match[2]]


# In[484]:


print(f"company_name: {company_name}")
print(f"document_date: {document_date}")
print(f"location: {location}")
print(f"dates_between: {dates_between}")
print(f"contact_person: {contact_person}")
print(f"contact_email: {contact_email}")
print(f"contact_number: {contact_number}")
print(f"guaranteed: {guaranteed}")

