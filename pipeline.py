import os
from glob import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

import tqdm

class LanePipeline:
    def __init__(
        self,
        camera_calibr_images,
        perspective_src,
        perspective_dst
    ):
        print("Calibrating camera. Please wait...")
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self._calibrate_camera(camera_calibr_images)
        print("Camera calibration complete.")
        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.M = cv2.getPerspectiveTransform(self.perspective_src, self.perspective_dst)
        self.Minv = cv2.getPerspectiveTransform(self.perspective_dst, self.perspective_src)
        self.num_frames_to_average = 30
        self.left_x_points = []
        self.left_y_points = []
        self.right_x_points = []
        self.right_y_points = []

    def _calibrate_camera(self, files):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        object_points = []
        image_points = []
        
        for fname in tqdm.tqdm(files):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                object_points.append(objp)
                image_points.append(corners)           
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs
    
    def camera_undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    @staticmethod
    def abs_sobel_thresh(img, orient='x', thresh=(0, 255), ksize=3):
        thresh_min, thresh_max = thresh
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # Return the result
        return binary_output    
    
    @staticmethod
    def sobel_mag_thresh(img, ksize=3, mag_thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        # Return the binary image
        return binary_output    
    
    @staticmethod
    # Define a function to threshold an image for a given range and Sobel kerne
    def dir_threshold(img, ksize=3, thresh=(0, np.pi/2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # Return the binary image
        return binary_output    
    
    @staticmethod
    def combined_threshold(
        img,
        ksize=3, 
        dir_ksize=9,
        gradx_thresh=(20, 255),
        grady_thresh=(20, 255),
        mag_thresh=(30, 100),
        angle_thresh=(0.75*np.pi/2, np.pi/2),
        color_thresh=(40, 255)
    ):
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        
        gradx = LanePipeline.abs_sobel_thresh(gray, orient='x', ksize=ksize, thresh=gradx_thresh)
        grady = LanePipeline.abs_sobel_thresh(gray, orient='y', ksize=ksize, thresh=grady_thresh)
        
        mag_binary = LanePipeline.sobel_mag_thresh(gray, ksize=ksize, mag_thresh=mag_thresh)
        dir_binary = LanePipeline.dir_threshold(gray, ksize=dir_ksize, thresh=angle_thresh)
        grads = np.zeros_like(gray)
        grads[((gradx == 1) & (grady == 1))] = 1 # & (grady == 1) & & (dir_binary == 1)
        
        color_thresh1 = np.zeros_like(S)
        color_thresh1[(
                (S > color_thresh[0]) & (S <= color_thresh[1])
            )] = 1
        
        color_thresh2 = np.zeros_like(S)
        color_thresh2[(
                (S > 70) & (S <= 255) & (L >= 50)
            )] = 1        
        
        combined = np.zeros_like(grads)
        combined[(grads == 1) & (color_thresh1 == 1) | (color_thresh2 == 1)] = 1 
        return combined
    
    def find_lines_histogram(self, img, visualize=False):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        if visualize:
            plt.plot(histogram)
            plt.title('Histogram')
            plt.show()
        out_img = np.dstack((img, img, img))*255 if visualize else None
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        
        midpoint = np.int(histogram.shape[0]//2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
                
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(img.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin
        margin = 150
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
                
        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            if visualize:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
            
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        if len(self.left_x_points) > self.num_frames_to_average:
            self.left_x_points.pop(0)
            
        if len(self.left_y_points) > self.num_frames_to_average:
            self.left_y_points.pop(0)
            
        if len(self.right_x_points) > self.num_frames_to_average:
            self.right_x_points.pop(0)
            
        if len(self.right_y_points) > self.num_frames_to_average:
            self.right_y_points.pop(0)            
        
        curr_leftx = leftx
        curr_lefty = lefty
        curr_rightx = rightx
        curr_righty = righty
        
        for i in range(len(self.left_x_points)):
            curr_leftx = np.concatenate(self.left_x_points)
            curr_lefty = np.concatenate(self.left_y_points)
            curr_rightx = np.concatenate(self.right_x_points)
            curr_righty = np.concatenate(self.right_y_points)
        
        self.left_x_points.append(leftx)
        self.left_y_points.append(lefty)
        
        self.right_x_points.append(rightx)
        self.right_y_points.append(righty)     
            
        # Fit a second order polynomial to each
        left_fit = np.polyfit(curr_lefty, curr_leftx, 2) # lefty, leftx, 2)
        right_fit = np.polyfit(curr_righty, curr_rightx, 2) # righty, rightx, 2)               
            
        # curr_leftx, curr_rightx
        return leftx, rightx, left_fit, right_fit, nonzerox,             nonzeroy, left_lane_inds, right_lane_inds, out_img
    
    def visualize_hist(self, img, left_fit, right_fit,                        nonzerox, nonzeroy, left_lane_inds,                        right_lane_inds, out_img):
        
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title('Finding lane lines')
        plt.show()
        
    def curvature(self, img, left_fit, right_fit):
        
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        
#         # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        
        y_eval = np.max(ploty)

        
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        
        lane_center = ((rightx[-1] - leftx[-1]) / 2) + leftx[-1]
        lane_offset = np.abs(lane_center - 650)
        
        lane_offset_meters = lane_offset * xm_per_pix
        
        return left_curverad, right_curverad, lane_offset_meters
    
    def draw_lane(self, img, warped, left_fit, right_fit, visualize=False):
        
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)        
        if visualize:
            plt.imshow(result)
            plt.title('Final Lane')
            plt.show()            
        return result
    
    def pipeline_apply(self, original, visualize=False):        
        img_size = np.shape(original)        
        rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        if visualize:
            plt.imshow(rgb)
            plt.title('Original Image')
            plt.show()
        undist = self.camera_undistort(rgb)
        if visualize:
            plt.imshow(undist)
            plt.title('Corrected camera distortion')
            plt.show()
        thresholded = LanePipeline.combined_threshold(undist)
        if visualize:
            plt.imshow(thresholded, cmap='gray')
            plt.title('Thresholded image')
            plt.show()            
        warped = cv2.warpPerspective(thresholded, self.M, (img_size[1], img_size[0]))
        if visualize:
            plt.imshow(warped, cmap='gray')
            plt.title('Warped perspective')
            plt.show()
        leftx, rightx, left_fit, right_fit, nonzerox, nonzeroy,         left_lane_inds, right_lane_inds, out_img = self.find_lines_histogram(
            warped, visualize=visualize
        )
        if visualize:
            self.visualize_hist(
                warped, left_fit, right_fit,
                nonzerox, nonzeroy, left_lane_inds,
                right_lane_inds, out_img
            )
        left_curverad, right_curverad, lane_offset_m = self.curvature(warped, left_fit, right_fit)
        
        lane = self.draw_lane(undist, warped, left_fit, right_fit, visualize=visualize)
        
        text = "CURVE RADII: (%.2f m, %.2f m)" % (left_curverad, right_curverad)
        annotated = cv2.putText(lane, text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155), 2, cv2.LINE_AA)
        text = "CENTER DISPLACEMENT: %.2f m" % lane_offset_m
        annotated = cv2.putText(annotated, text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 155), 2, cv2.LINE_AA)
        
        if visualize:
            plt.imshow(annotated)
            plt.title("annotated")
            plt.show()
            
        return rgb, undist, thresholded, warped, lane, annotated

    def pipeline(self, img_path):
        rgb, undist, thresholded, warped, lane, annotated = self.pipeline_apply(img_path, False)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        return annotate
          

# rgb, undist, thresholded, warped, lane, annotated = pipeline.pipeline_apply(
#     cv2.imread('./test_images/test1.jpg'),
#     visualize=True
# )

def find_lines_in_video(video_path, pipeline):
    clip = VideoFileClip(video_path)
    clip_with_lines = clip.fl_image(pipeline.pipeline)
    clip_with_lines.write_videofile('annotated.mp4', audio=False)        


if __name__ == '__main__':

    print('Running pipeline. Please wait.')

    src = np.float32([[555, 481], [734, 481], [299, 655], [1001, 655]])
    dst = np.float32([[226, 96], [1106, 96], [246, 577], [1106, 577]])

    pipeline = LanePipeline(
        camera_calibr_images = glob('./camera_cal/*.jpg'),
        perspective_src = src,
        perspective_dst = dst
    )

    find_lines_in_video('./project_video.mp4', pipeline)

    print('Done.')
