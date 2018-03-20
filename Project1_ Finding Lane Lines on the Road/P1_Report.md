
# **Finding Lane Lines on the Road** 

## Goals: Finding Lane Lines on the Road

First, extract lane lines from different images. Then, apply our methods to videos.
The main technologies in this project are the following:
* HLS color conversion
* Canny edges detection
* cv2.fillPoly to select area of interest
* HoughLinesP to find lines
* Divide lines into left  and right parts by slope
* Smooth left and right lines

---
## Steps
### Step 1: color selection
To select yellow and white colors, we first apply HLS to convert the original image. Then set the threshold to get only yellow/white.
``` python
def filter_white_yellow_hls(image):
    '''
    Identify colors of yellow and white by HLS conversion and filtering. 
    Input and output are image arrays with the same size. eg. (720, 1280, 3) -->(720, 1280, 3).
    '''
    hls_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    # define range of color in HLS
    lower_yel = np.array([10,0,100])
    upper_yel = np.array([40,255,255])
    lower_wht = np.array([0,200,0])
    upper_wht = np.array([255,255,255])
    
    # Threshold the HLS image to get only yellow/white 
    yellow_mask = cv2.inRange(hls_img, lower_yel, upper_yel)
    white_mask = cv2.inRange(hls_img, lower_wht, upper_wht)
    
    # combine the mask
    full_mask = cv2.bitwise_or(white_mask, yellow_mask)
    boosted_lanes = cv2.bitwise_and(image, image, mask = full_mask)
    return hls_img, boosted_lanes
```
Here are results:
IMG1

SolidWhiteRight| SolidYellowLeft
------------ | -------------
![solidWhiteRight7-8.jpeg](http://upload-images.jianshu.io/upload_images/6982894-37889ed115d8ada6.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) |  ![solidYellowLeft14-8.jpeg](http://upload-images.jianshu.io/upload_images/6982894-ae40900ea9c66bdb.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)
![solidWhiteRight7-8_0hls_img.png](http://upload-images.jianshu.io/upload_images/6982894-1b1211629163bf6f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) | ![solidYellowLeft14-8_0hls_img.png](http://upload-images.jianshu.io/upload_images/6982894-e1ad0594d0dc8820.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)
![solidWhiteRight7-8_1hls.png](http://upload-images.jianshu.io/upload_images/6982894-a5291f28d5c4f9b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)| ![solidYellowLeft14-8_1hls.png](http://upload-images.jianshu.io/upload_images/6982894-f703a338bcbd0e28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)

Challenge | Challenge
------------ | -------------
![challenge3-6.jpeg](http://upload-images.jianshu.io/upload_images/6982894-b6bd984e5ec5d68a.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) | ![challenge4-1.jpeg](http://upload-images.jianshu.io/upload_images/6982894-e4322e8c48ee95e3.jpeg?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)
![challenge3-6_0hls_img.png](http://upload-images.jianshu.io/upload_images/6982894-d5c56917d6d3b033.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) | ![challenge4-1_0hls_img.png](http://upload-images.jianshu.io/upload_images/6982894-ba283871db93da33.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)
![challenge3-6_1hls.png](http://upload-images.jianshu.io/upload_images/6982894-2c282a1ab2592ae2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) | ![challenge4-1_1hls.png](http://upload-images.jianshu.io/upload_images/6982894-5a2e41d7d430de3c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)



### Step 2: edges detection
After identifying the colors, we make a smoothy of the image by `cv2.GaussianBlur`. Then, Canny Edge Detection is applied to find the edges of the lane lines in an image of the road. As far as a ratio of low_threshold to high_threshold, [John Canny himself recommended](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html#steps) a low to high ratio of 1:2 or 1:3.
``` python
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_hls, low_threshold, high_threshold)
```
IMG2

SolidWhiteRight| SolidYellowLeft
------------ | -------------
![solidWhiteRight7-8_2blur_hls.png](http://upload-images.jianshu.io/upload_images/6982894-a919b12674a127a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)| ![solidYellowLeft14-8_2blur_hls.png](http://upload-images.jianshu.io/upload_images/6982894-4d6b04af1fdd6d29.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![solidWhiteRight7-8_3edges.png](http://upload-images.jianshu.io/upload_images/6982894-d33dbbba5ddf8903.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)| ![solidYellowLeft14-8_3edges.png](http://upload-images.jianshu.io/upload_images/6982894-9d3c90aefa9bdeda.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Challenge | Challenge
------------ | -------------
![challenge3-6_2blur_hls.png](http://upload-images.jianshu.io/upload_images/6982894-9abc2a1d6b78be95.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![challenge4-1_2blur_hls.png](http://upload-images.jianshu.io/upload_images/6982894-99723f8dad4c6eef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![challenge3-6_3edges.png](http://upload-images.jianshu.io/upload_images/6982894-0f234ebe257bb7c3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![challenge4-1_3edges.png](http://upload-images.jianshu.io/upload_images/6982894-02a3b886639a441c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### Step 3: region masking
According to the shape of two lane lines, we take a four sided polygon to mask. 
Tips: the coordinate origin (x=0, y=0) is in the upper left in image processing. The region of the mask should not be too small. Otherwise, mistakes will occur after images change. Models are not stable as well.
``` python
    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    left_bottom = (0.10*imshape[1],imshape[0])
    left_top = (0.45*imshape[1], 0.6*imshape[0])
    right_top = (0.55*imshape[1], 0.6*imshape[0])
    right_bottom = (0.9*imshape[1],0.95*imshape[0])
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    mask_zone = cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)  
```
IMG3

SolidWhiteRight| SolidYellowLeft
------------ | -------------
![solidWhiteRight7-8_4masked_edges.png](http://upload-images.jianshu.io/upload_images/6982894-a14f561f31a2f763.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![solidYellowLeft14-8_4masked_edges.png](http://upload-images.jianshu.io/upload_images/6982894-d5fa0c6cea8c974b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Challenge | Challenge
------------ | -------------
![challenge3-6_4masked_edges.png](http://upload-images.jianshu.io/upload_images/6982894-53ee637425457c2b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![challenge4-1_4masked_edges.png](http://upload-images.jianshu.io/upload_images/6982894-9630a01197bd766e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Step 4: Hough lines detection
Run Hough on edge detected image. Output "lines" is an array containing endpoints of detected line segments. Parameter rho takes a minimum value of 1, and a reasonable starting place for theta is 1 degree (pi/180 in radians). A good parameter set can clear noisy lines efficiently. The output of this part matters the final result a lot.
``` python
    # Define the Hough transform parameters    
    rho = 3 # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15      # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 25    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
```
IMG4

SolidWhiteRight| SolidYellowLeft
------------ | -------------
![solidWhiteRight7-8_5line_img.png](http://upload-images.jianshu.io/upload_images/6982894-e899e42c6e47989a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![solidYellowLeft14-8_5line_img.png](http://upload-images.jianshu.io/upload_images/6982894-2d7cda9cdb4a5b54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Challenge | Challenge
------------ | -------------
![challenge3-6_5line_img.png](http://upload-images.jianshu.io/upload_images/6982894-72f1680060d058cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![challenge4-1_5line_img.png](http://upload-images.jianshu.io/upload_images/6982894-241edbfd75b35f6e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Step 5: Drawing lines

**Divide line segments into left and right parts**
Separating line segments by their slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left line vs. the right line. 

```python
def devide_left_right_lines(lines, line_threshold):
    #Calculate slopes of each line, then devide lines into left and right by slope and lines' location(line_threshold) 
    #Collect lines' start and end points into four lists.
    
    x_l,y_l,x_r,y_r = [],[],[],[]
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2-x1)!=0:
                slope = (y2-y1)/(x2-x1)
                if slope<-0.5 and x1<line_threshold and x2<line_threshold:
                    x_l.append(x1)
                    x_l.append(x2)
                    y_l.append(y1)
                    y_l.append(y2)
                if slope> 0.5 and x1>line_threshold and x2>line_threshold:
                    x_r.append(x1)
                    x_r.append(x2)
                    y_r.append(y1)
                    y_r.append(y2)
    return x_l, y_l, x_r, y_r
```
**Find start and end points of left and right lines**
Then, we adopt 1D polynomial function `np.polyfit ` to fit two segments and extrapolate to the top and bottom of the lane.
```python
def find_start_end_points(x_list, y_list, alpha, imshape):
    '''
    After segregated lines into left/right baskets(x_list, y_list), 
    combine the lines in each basket to come up with a single line.
    Using 1D polynomial np.polyfit to fit x_list, y_list and 
    compute the average slope m and c of line y = mx + c.
    Finally, calculate (x, y) coordinates for the extreme points of the lines 
    '''
    # compute the average slope m and c 
    als_xy = np.polyfit(x_list,y_list,1)
    als_yx = np.polyfit(y_list,x_list,1)
    
    # 1D polynomial
    p_yx = np.poly1d(als_yx)
    p_xy = np.poly1d(als_xy)
    
    # compute x according to y and compute y according to x
    yy = imshape[0]
    xx = alpha*imshape[1]
    start_xy = (int(p_yx(yy)),yy)
    end_xy = (int(xx),int(p_xy(xx)))
    return start_xy, end_xy
```
**Smooth left and right lines for image stream (video)**
* Using a queue to smooth the lines. Each line's start and end points are an average of `qsize` (we set 3) frames, one is itself and others are previous frames. 
* If the current line has a large deviation from the average of previous frames, it will be abandoned and replaced by the average. This tactic is used for avoiding abnormal lines. It also benefits to the stability of the model.
```python
    def smoother(arr, dev_threshold):
        if smooth_queue.full():
            smooth_queue.get()
        q_size = smooth_queue.qsize()
        if q_size==0:
            smooth_queue.put(arr)
            line_image_avg = arr
        else:
            q_arr = np.asarray(smooth_queue.queue)
            last_line_image_avg = (sum(q_arr)/(q_size)).astype('int32')
            
            gap = np.sum(abs(arr-last_line_image_avg))
            if gap>dev_threshold*8:
                line_image_avg = last_line_image_avg
            else:
                smooth_queue.put(arr)
                new_q_arr = np.asarray(smooth_queue.queue)
                line_image_avg = (sum(new_q_arr)/(q_size+1)).astype('int32')                       
        return line_image_avg 
```

SolidWhiteRight| SolidYellowLeft
------------ | -------------
![solidWhiteRight7-8_6line_image_lr.png](http://upload-images.jianshu.io/upload_images/6982894-5ee1d77a02646e3f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![solidYellowLeft14-8_6line_image_lr.png](http://upload-images.jianshu.io/upload_images/6982894-b08e3d9a7be16119.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Challenge | Challenge
------------ | -------------
![challenge3-6_6line_image_lr.png](http://upload-images.jianshu.io/upload_images/6982894-01d218ace3dd5f5a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | ![challenge4-1_6line_image_lr.png](http://upload-images.jianshu.io/upload_images/6982894-3ac49557a5b041a9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


* Drawing the smooth-after lines on the image.
```python
    #Make a smoothing, filter abnormal points(lines)
    points = np.array([np.asarray(left_start_xy), np.asarray(left_end_xy), np.asarray(right_start_xy), np.asarray(right_end_xy)])
    points_avg = smoother(points, dev_threshold=10)
    
    #Convert arrays to tuples, then draw the lines
    lx = totuple(points_avg[0])
    ly = totuple(points_avg[1])
    rx = totuple(points_avg[2])
    ry = totuple(points_avg[3])
    cv2.line(line_image_left, lx, ly, (255,0,0),6)
    cv2.line(line_image_right, rx,ry,(255,0,0),6)
    
    #Combine left and right line, then add it to original image
    line_image_lr = cv2.addWeighted(line_image_right, 1, line_image_left, 1, 0)
    line_image = cv2.addWeighted(image, 1, line_image_lr, 1, 0)       
```

SolidWhiteRight| SolidYellowLeft
------------ | -------------
![solidWhiteRight7-8_7line_image.png](http://upload-images.jianshu.io/upload_images/6982894-ac765d65d165c2db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) | ![solidYellowLeft14-8_7line_image.png](http://upload-images.jianshu.io/upload_images/6982894-30ad8864944d7f3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)



Challenge | Challenge
------------ | -------------
![challenge3-6_7line_image.png](http://upload-images.jianshu.io/upload_images/6982894-676cb03149e3179c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400) | ![challenge4-1_7line_image.png](http://upload-images.jianshu.io/upload_images/6982894-5bdde1a02c048634.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/400)


---
## Potential Shortcomings
* One potential shortcoming would be what would happen when road situation is so bad. For example, when the road has a sudden turn.The surface of the road is so bright that the lanes are hard to be recognized. 
* Another thing that the model may make mistake is that when a white/yellow car in front our camera closely.
* Finally, the shortcoming could be the lines' vibration when the scenes of road change fast, eg. shadows, turns. 

## Possible Improvements
* Add a smoother to the slope to reduce the shake of lines.
* Improve the color selection and Hough lines detection in case of a complicated situation.
[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---





