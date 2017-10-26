import os
import cv2
import time
import numpy as np
import tensorflow as tf
import pickle
from matplotlib import pyplot as plt
import copy
#???works only in ipython notebook from . import pedestrianDetection

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)


###############################################
def get_light_color(imgbox, imgo, tmpBox, lower_HSV, upper_HSV):
    # retain the orignal image
    """
    print("imgbox image:")
    plt.imshow(cv2.cvtColor(imgbox, cv2.COLOR_RGB2BGR))
    plt.show()
    print("imgo image:")
    plt.imshow(imgo)
    plt.show()
    """

    #mgbox = cv2.cvtColor(imgbox, cv2.COLOR_RGB2BGR)
    imgOrig = imgbox

    # use initial image without any bounding boxes to classify traffic light
    # color since the bounding boxes sometimes totally blocking the traffic light color
    img = imgo    
    #convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    """
    print("make sure the original image in BGR format")
    plt.imshow(imgOrig)
    plt.show()
    print("make sure the new image in BGR format")
    plt.imshow(img)
    plt.show()
    """

    colorID = "UNKNOWN"
    # median blur the image
    img = cv2.medianBlur(img, 5)
    # Convert image to HSV
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    # Threshold the HSV image to get only selected(red, green, or yellow) colors
    mask = cv2.inRange(hsvImg, lower_HSV, upper_HSV) 
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    
    #mask out the area in image that has no traffic lights
    #create a black image
    polygon_img = np.zeros(img.shape, np.uint8)
    
    left_x = tmpBox[0] 
    top_y = tmpBox[2] 
    right_x = tmpBox[1] 
    bot_y = tmpBox[3]
    #print("left_x: {}, top_y:  {}, right_x: {}, bot_y: {}".format(left_x, top_y, right_x, bot_y))
    pts = np.array([[left_x, top_y], [right_x, top_y], [right_x, bot_y], [left_x, bot_y]])
    cv2.fillPoly(polygon_img, pts=[pts], color=(255,255,255))
    res1 = cv2.bitwise_and(res,res,mask=polygon_img[:,:,1])
         
    # Debug.
    #cv2.imwrite('img.png',img)
    #cv2.imwrite('poly.png',polygon_img)

    #cv2.imwrite('combined.jpg',cv2.bitwise_and(
    #img,img,mask=polygon_img[:,:,1]))
    #plt.imshow(img)
    #plt.show()
    #print("mask image")
    #plt.imshow(mask)
    #plt.show()
    #print("res1 image")
    #plt.imshow(res1)
    #plt.show()
        
    #brightest spot
    a = np.array(res1)
    #print(a.max(), np.unravel_index(a.argmax(), a.shape))
    brighty = np.unravel_index(a.argmax(), a.shape)[0]
    brightx = np.unravel_index(a.argmax(), a.shape)[1]
    #print("Brightest spot, brightx: {}, birghty: {}".format(brightx, brighty)) 

    #color hsv range boolean
    greenColor = np.all(lower_HSV == np.array([60, 125, 125])) and np.all(upper_HSV == np.array([120,255,255]))
    redColor = np.all(lower_HSV == np.array([170, 125, 125])) and np.all(upper_HSV == np.array([179,255,255]))
    yellowColor = np.all(lower_HSV == np.array([5, 150, 150])) and np.all(upper_HSV == np.array([15,255,255]))

    #divide the bounding box into 3 regions: red, yellow, and green
    upperYellowy = top_y + (bot_y - top_y)/3
    lowerYellowy = bot_y - (bot_y - top_y)/3
    #print("Average height of traffic light: {}".format((bot_y - top_y)/3))
    #print("Width of the traffic light: {}".format(right_x - left_x))
    #print("Ratio of average height over width: {}".format((bot_y - top_y)/(3*(right_x - left_x))))
    avgHWratio = (bot_y - top_y)/(3*(right_x - left_x))
    #print("top_y: {}, upperYellowy: {}, lowerYellowy: {}, bot_y:{}".format(top_y, upperYellowy, lowerYellowy, bot_y))
    #put the original image back
    img = imgOrig
        
    #average height of the traffic light has to be over 10 pixels for red, yellow, and green 3 color box type
    if (((brightx == 0) and (brighty == 0)) == False and avgHWratio > 0.5): #(bot_y - top_y)/3 > 10):
        if (brighty >= lowerYellowy and brighty <= bot_y) and (greenColor == True):
            #print("********* G R E E N *********")
            cv2.rectangle(img, (brightx -15, brighty - 15), (brightx + 15, brighty + 15), (0,255,0),2)
            cv2.putText(img, "green traffic light", (brightx-15, brighty -27), 0, 1.2, (0,255,0),2)
            colorID = "GREEN"
            #print("At time: {} sec, colorID: TrafficLight.GREEN ".format(str(time.clock())))
        elif (brighty >= top_y and brighty < upperYellowy) and (redColor == True):
            #print("*********   R E D   *********")
            cv2.rectangle(img, (brightx -15, brighty - 15), (brightx + 15, brighty + 15), (0,0,255),2)
            cv2.putText(img, "red traffic light", (brightx-15, brighty -27), 0, 1.2, (0,0,255),2)
            colorID = "RED"
            #print("At time: {} sec, colorID: TrafficLight.RED".format(str(time.clock())))
        elif (brighty >= upperYellowy and brighty < lowerYellowy) and (yellowColor == True):
            #print("******** Y E L L O W ********")
            cv2.rectangle(img, (brightx -15, brighty - 15), (brightx + 15, brighty + 15), (255,255,0),2)
            cv2.putText(img, "yellow traffic light", (brightx-15, brighty -27), 0, 1.2, (255,255,0),2)
              
            colorID = "YELLOW"
            #print("At time: {} sec, colorID: TrafficLight.YELLOW".format(str(time.clock())))
        #plt.imshow(img)
        #plt.show()
    return colorID, img
##############################################

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op, self.summary_op] 
        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im, imname=None): #only for testing images , imname):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    #_, h, w, _ = im.shape
    h, w, _ = im.shape
    imgcv = np.copy(im)
    im = self.framework.resize_input(im)
    h2, w2, _ = im.shape
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']

    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        max_indx = tmpBox[5]
        thick = int((h + w) // 300)
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                 "y": tmpBox[3]}
        })

        #print('boxes',tmpBox[0],tmpBox[1],tmpBox[2],tmpBox[3],tmpBox[4])
        if 'traffic light' in tmpBox[4]:
            #print("found traffic light")

            #classify the traffic light color
    
            ##from my code in Udacity CapstoneProject for traffic light color classification########
            #create a deep copy of imgcv
            imgo = copy.deepcopy(imgcv)
            #convert BGR to RGB
            imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)

            #initialize all variables
            yellowLight = False
            redLight = False
            greenLight = False
            yellowImg = imgcv
            redImg = imgcv
            greenImg = imgcv
            result = imgcv
            #plt.imshow(imgcv)
            #plt.show()

            ###################green color detection##########
            # define range of green color in HSV
            lower_green = np.array([60,125,125]) #100,100])
            upper_green = np.array([120,255,255])
            #print("start green classification")
            [clr_ID, greenImg] = get_light_color(imgcv, imgo, tmpBox, lower_green, upper_green)
            if (clr_ID == "GREEN"):
                greenLight = True
            ##################red color detection#################
            # define range of red color in HSV
            lower_red = np.array([170,125,125]) 
            upper_red = np.array([179,255,255])
            #print("start red classification")
            [clr_ID, redImg] = get_light_color(imgcv, imgo, tmpBox, lower_red, upper_red)
            if (clr_ID == "RED"):
                redLight = True


            ###########yellow traffic light detection###########
            # define range of orange color in HSV
            lower_yellow = np.array([5,150,150]) 
            upper_yellow = np.array([15,255,255]) #40,255,255]) #real amber traffic light works 15,255,255])
            #print("start yellow classification")
            [clr_ID, yellowImg] = get_light_color(imgcv, imgo, tmpBox, lower_yellow, upper_yellow)
            if (clr_ID == "YELLOW"):
                yellowLight = True
            	
            if ((yellowLight == True) and (redLight == False) 
                 and (greenLight == False)):
                clr_ID = "YELLOW"
                result = yellowImg

            elif ((yellowLight == False) and (redLight == True) 
                and (yellowLight == False)):
                clr_ID = "RED"
                result = redImg

            elif ((yellowLight == False) and (redLight == False) 
                 and (greenLight == True)):
                clr_ID = "GREEN" 
                result = greenImg
            else:
                clr_ID = "UNKNOWN"
                result = imgcv
        
            #plt.imshow(result)
            #plt.show()
            #print("Traffic Light color_ID: {}".format(clr_ID))
            imgcv = result        

            ###################################################################################
        if (tmpBox[4] == 'traffic light') or (tmpBox[4] == 'car') or (tmpBox[4] == 'truck') or (tmpBox[4] == 'pedestrian') or (tmpBox[4] == 'cyclist'): #  in tmpBox[4]:
            cv2.rectangle(imgcv,(tmpBox[0], tmpBox[2]), (tmpBox[1], tmpBox[3]), colors[max_indx], thick)
            cv2.putText(imgcv, tmpBox[4], (tmpBox[0], tmpBox[2] - 12), 0, 1e-3 * h, colors[max_indx],thick//3)
    if (imname is not None):        
        outfolder = './test/out/'
        img_name = os.path.join(outfolder, imname.split('/')[-1]) 
        print("writing file: {}".format(img_name))
        cv2.imwrite(img_name, imgcv)

    """
    #since this yolow cfg file is not good at pedestrian    classification, I will try one that works well in the past 
    #Now I realize it is a software issue, only works well in ipython notebook with the same code, but it doesn't work in ubuntu for now???
    imgcv = pedestrianDetection.frame_func_person(
        cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB))
    imgcv = cv2.cvtColor(imgcv, cv2.COLOR_RGB2BGR)
    """

    return imgcv, boxesInfo

import math

def predict(self):
    inp_path = self.FLAGS.test
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any test files in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            self.framework.postprocess(prediction,
                os.path.join(inp_path, this_batch[i]))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
