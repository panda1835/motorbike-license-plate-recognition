import os
from flask import Flask, render_template, request
from flask import send_from_directory
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
TEMPLATES_FOLDER = 'templates'


# Preprocess an image
def extract_number(file_name):
    plates = cv2.imread(file_name)

    # convert greyscale
    grey_plates = cv2.cvtColor(plates, cv2.COLOR_BGR2GRAY)

    # blur to redure noise
    blur_plates = cv2.GaussianBlur(grey_plates,(5,5),0)

    cv2.imwrite('uploads/blur.jpg', blur_plates) #///////////////////////

    # threshold
    thresh_plates_binary = cv2.threshold(blur_plates, 150, 255, cv2.THRESH_BINARY)[1]
    thresh_plates_adaptive = cv2.adaptiveThreshold(blur_plates, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 11, 8)
    # dilate to fill some little holes in numbers

    thresh_plates = thresh_plates_binary
    
    cv2.imwrite('uploads/thresh.jpg', thresh_plates) #///////////////////////

    # find contours
    contours_plates = cv2.findContours(thresh_plates,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours_plates_out = thresh_plates.copy()

    # find plates in contours
    plates_array = [] # hold the coordinate of the extracted plates
    for i in range(len(contours_plates)):
            x, y, w, h = cv2.boundingRect(contours_plates[i])
            if (w/h>1) & (w/h<2) :#& (w>plates.shape[1]/5) & (h>plates.shape[0]/5):
                plates_array.append((x,y,w,h))
                cv2.rectangle(contours_plates_out, (x,y), (x+w,y+h), (160, 160, 160), 5)

    cv2.imwrite('uploads/plates_first.jpg', contours_plates_out) #///////////////////////

    extracted_plates = []
    extracted_plates = extract(blur_plates, plates_array)
    
    for i in range(len(extracted_plates)):
        extracted_plates[i] = cv2.adaptiveThreshold(extracted_plates[i], 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 11, 8)

    extracted_numbers = []
    extracted_numbers_array = []
    for j in range(len(extracted_plates)):
        try:
            numbers_array = [] # hold the coordinate of the extracted numbers
            extract_plate_contour = cv2.findContours(extracted_plates[j], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            draw_extract_plate_contour = extracted_plates[j].copy()

            for i in range(len(extract_plate_contour)):
                x,y,w, h = cv2.boundingRect(extract_plate_contour[i])
                if (h/w>1.5) & (h/w<5.5) & (h>1/4*extracted_plates[j].shape[0])&(h<1/2*extracted_plates[j].shape[0]):
                    cv2.rectangle(draw_extract_plate_contour,(x-1,y-1), (x+w+1,y+h+1), (160,160,160), 2)
                    numbers_array.append([x,y,w,h])
                    
            numbers_array = removeInnerBound(numbers_array)
            # place the number in order left -> right, up -> down
            numbers_array = order(numbers_array)
            cv2.imwrite('uploads/bound_numbers.jpg', cv2.resize(draw_extract_plate_contour,(300,200))) #///////////////////////
            # map the bounding of numbers in original grey image
            for i in range(len(numbers_array)):
                numbers_array[i][0] += plates_array[j][0]
                numbers_array[i][1] += plates_array[j][1]

            extracted_numbers = extract(grey_plates, numbers_array)

            for i in range(len(extracted_numbers)):
                extracted_numbers[i] = 255 - extracted_numbers[i]
                # normalization
                extracted_numbers[i] = extracted_numbers[i]/255   
            extracted_numbers_array.append(extracted_numbers)
        except Exception as e:
            pass
    return extracted_numbers_array

from sklearn.linear_model import LinearRegression
def order(xy):
    '''sort the list of point in order from top to bottom, left to right
       i.e. 1  2  3
            4  5  6'''
    data = np.array(xy)
    # make a hyperplane using linear regression
    lr = LinearRegression().fit(data.reshape(len(xy),-1)[:,0].reshape(-1,1),data.reshape(len(xy),-1)[:,1])
    w = lr.coef_[0]*data.reshape(len(xy),-1)[:,0]+lr.intercept_-data.reshape(len(xy),-1)[:,1]
   
    upper = data[w > 0]
    lower = data[w < 0]
    upper = upper.reshape(-1,len(upper[0]))
    upper = upper[np.argsort(upper[:,0])]
    lower = lower.reshape(-1,len(upper[0]))
    lower = lower[np.argsort(lower[:,0])]
    
    order = np.concatenate((upper, lower))
        
    return order

def removeInnerBound(number_array_):
    number_array = number_array_.copy()
    while True:
        arr = np.array(number_array)
        area = arr[:,2]*arr[:,3]
        smallrect = arr[np.argmin(area)]
        for bigrect in arr:
            if (bigrect != smallrect).any() & (isContained(bigrect, smallrect)):
                number_array.pop(np.argmin(area))
                continue
        break
    return number_array

def isContained(bigrect, smallrect):
    return ((smallrect[0]>bigrect[0]) & ((smallrect[0]+smallrect[2])<(bigrect[0]+bigrect[2])) \
          &(smallrect[1]>bigrect[1]) & ((smallrect[1]+smallrect[3])<(bigrect[1]+bigrect[3])))

def extract(image, bounding):
    '''extract objects with rectangle bounding coordinates from an image
        @return list of extracted image
        @param bounding: list of tuples (x,y,w,h) of the rectangle bound
        @param image: image source, gray scale format'''
    extracted_array = []
    for i in range(len(bounding)):
        x, y, w, h = bounding[i]
        extracted = np.zeros((h,w), np.uint8)
        extracted[:,:] = image[y:y+h,x:x+w] 
        extracted_array.append(extracted)
    return extracted_array

# Predict & classify image
def predict(image_path):
    plates = extract_number(image_path)
    model_number = pickle.load(open('number.pkl', 'rb'))
    model_character = pickle.load(open('character.pkl', 'rb'))
    predictions = []
    res = []
    result = ''
    prob = 1
    img_out = np.ones((140,1))
    for extracted_numbers in plates:
        prediction = []
        for pos in range(len(extracted_numbers)):
            if pos != 2: 
                value = 0.15
                lim = 1 - value
                extracted_numbers[pos][extracted_numbers[pos] > lim] = 1
                extracted_numbers[pos][extracted_numbers[pos] <= lim] += value
                extracted_numbers[pos][extracted_numbers[pos]<0.5] /= 2
                # resize (12,28)
                extracted_numbers[pos] = cv2.resize(extracted_numbers[pos],(12,28))
                extracted_numbers[pos] = cv2.erode(extracted_numbers[pos], np.ones((3,3), 'uint8'), iterations = 1)
                flatten_img = np.array(extracted_numbers[pos]).reshape(1, -1)
                pred = model_number.predict(flatten_img)[0]
                prediction.append(pred)
                prob = round(model_number.predict_proba(flatten_img).max(),2)
                text = f'{pred}-{int(prob*100)}%'
                img = np.zeros((140,60))
                img[20:,:] = cv2.resize(extracted_numbers[pos],(60,120))
                img = cv2.putText(img,text, (2,15) ,cv2.FONT_HERSHEY_PLAIN,1,(1,1,1))
                img_out = np.concatenate((img_out, img), axis = 1)
            else:
                value = 0.1
                lim = 1 - value
                extracted_numbers[pos][extracted_numbers[pos] > lim] = 1
                extracted_numbers[pos][extracted_numbers[pos] <= lim] += value
                # darkening, for some dim image, make the dark part around numbers darker
                extracted_numbers[pos][extracted_numbers[pos]<0.5] /= 2
                # resize (30,60)
                extracted_numbers[pos] = cv2.resize(extracted_numbers[pos],(30,60))
                extracted_numbers[pos] = cv2.erode(extracted_numbers[pos], np.ones((2,2), 'uint8'), iterations = 1)
                flatten_img = np.array(extracted_numbers[pos]).reshape(1, -1)
                pred = model_character.predict(flatten_img)[0]
                prediction.append(pred) 
                prob = round(model_character.predict_proba(flatten_img).max(),2)
                text = f'{pred}-{int(prob*100)}%'
                img = np.zeros((140,60))
                img[20:,:] = cv2.resize(extracted_numbers[pos],(60,120))
                img = cv2.putText(img,text, (2,15) ,cv2.FONT_HERSHEY_PLAIN,1,(1,1,1))
                img_out = np.concatenate((img_out, img), axis = 1)
                
        predictions.append(prediction)

    cv2.imwrite('uploads/predictions.jpg', img_out*255)#///////////////////////

    for i in predictions:
        re = ""
        for j in i:
            re += j
        re = re[:2]+'-'+re[2:]
        res.append(re)
    for i in res:
        result += i+"\n" 
    return result, len(plates)

# home page
@app.route('/')
def home():
   return render_template('home.html')

@app.route('/classify', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('home.html')
    else:
        file = request.files["image"]
        file.filename = "original.jpg"
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)

        label, number_of_plate = predict(upload_image_path)
    
    return render_template('classify.html', image_file_name = file.filename, label = label, \
    number_of_plate = number_of_plate, blur = 'blur.jpg', bound_numbers = 'bound_numbers.jpg', \
    plates_first = 'plates_first.jpg', predictions = 'predictions.jpg', thresh = 'thresh.jpg')

@app.route('/classify/<filename>')
def send_file(filename):
    response = send_from_directory(UPLOAD_FOLDER, filename)
    response.cache_control.no_cache = True
    return response

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True