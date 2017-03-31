#include "libfreenect.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;

bool editor=false;
bool writing=false;

//bool isRGBDepth = false, isDynHandSeg = false, isDots = false;
Scalar white = Scalar(255, 255, 255);
bool die=false;

Mat find_fist(Mat image)
{
 // Load Face cascade (.xml file)
 CascadeClassifier face_cascade( "fist.xml" );
 
 // Detect faces
 std::vector<Rect> faces;
 face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
int z=faces.size();
//printf("%d\n",z);
 Mat fist = Mat::zeros(640,840, CV_8UC1);
 // Draw circles on the detected faces
 for( int i = 0; i < faces.size(); i++ )
 {
  Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
  ellipse(fist, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255),-1, 8, 0 );
 } 
 return image;
}


class myMutex
{
	public:
		myMutex()
		{
			pthread_mutex_init( &m_mutex, NULL );
		}
		void lock() 
		{
			pthread_mutex_lock( &m_mutex );
		}
		void unlock() 
		{
			pthread_mutex_unlock( &m_mutex );
		}
	private:
		pthread_mutex_t m_mutex;
};

class MyFreenectDevice : public Freenect::FreenectDevice 
{
	public:
		MyFreenectDevice(freenect_context *_ctx, int _index)
	 		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),
			m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false),
			m_new_depth_frame(false), depthMat(Size(640,480),CV_16UC1),
			rgbMat(Size(640,480), CV_8UC3, Scalar(0)),
			ownMat(Size(640,480),CV_8UC3,Scalar(0)) {
			
			for( unsigned int i = 0 ; i < 2048 ; i++) 
			{
				float v = i/2048.0;
				v = std::pow(v, 3)* 6;
				m_gamma[i] = v*6*256;
			}
		}
		
		// Do not call directly even in child
		void VideoCallback(void* _rgb, uint32_t timestamp) 
		{
			m_rgb_mutex.lock();
			uint8_t* rgb = static_cast<uint8_t*>(_rgb);
			rgbMat.data = rgb;
			m_new_rgb_frame = true;
			m_rgb_mutex.unlock();
		};
		
		// Do not call directly even in child
		void DepthCallback(void* _depth, uint32_t timestamp) 
		{
			m_depth_mutex.lock();
			uint16_t* depth = static_cast<uint16_t*>(_depth);
			depthMat.data = (uchar*) depth;
			m_new_depth_frame = true;
			m_depth_mutex.unlock();
		}
		
		bool getVideo(Mat& output) 
		{
			m_rgb_mutex.lock();
			if(m_new_rgb_frame) 
			{
				cv::cvtColor(rgbMat, output, CV_RGB2BGR);
				m_new_rgb_frame = false;
				m_rgb_mutex.unlock();
				return true;
			} 
			else 
			{
				m_rgb_mutex.unlock();
				return false;
			}
		}
		
		bool getDepth(Mat& output) 
		{
				m_depth_mutex.lock();
				if(m_new_depth_frame) 
				{
					depthMat.copyTo(output);
					m_new_depth_frame = false;
					m_depth_mutex.unlock();
					return true;
				} 
				else 
				{
					m_depth_mutex.unlock();
					return false;
				}
			}
	private:
		std::vector<uint8_t> m_buffer_depth;
		std::vector<uint8_t> m_buffer_rgb;
		std::vector<uint16_t> m_gamma;
		Mat depthMat;
		Mat rgbMat;
		Mat ownMat;
		myMutex m_rgb_mutex;
		myMutex m_depth_mutex;
		bool m_new_rgb_frame;
		bool m_new_depth_frame;
};

int main(int argc, char **argv) 
{
	Mat savedImage;
	Mat depthMat(Size(640,480),CV_16UC1);
	Mat depthf (Size(640,480),CV_8UC1);
	Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
	Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));
	
	// The next two lines must be changed as Freenect::Freenect
	// isn't a template but the method createDevice:
	// Freenect::Freenect<MyFreenectDevice> freenect;
	// MyFreenectDevice& device = freenect.createDevice(0);
	// by these two lines:
	
	Freenect::Freenect freenect;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
	
	device.startVideo();
	device.startDepth();

	int maxdepth=0, flag=0;
	Mat findChar;
	findChar = Mat::zeros(depthf.size(), CV_8UC1);
	savedImage=Mat::zeros(depthf.size(),CV_8UC1);
	//Mat handSegmented = Mat::zeros(depthf.size(), CV_8UC1);
	int iclosep=320;
	int jclosep=240;

	while (!die) 
	{
		Mat handSegmented = Mat::zeros(depthf.size(), CV_8UC1);
		int a=0;
		int b=0;
		device.getVideo(rgbMat);
		if(rgbMat.empty())
			printf("No colour image.\n");
		device.getDepth(depthMat);
		if(depthMat.empty())
			printf("No depth image.\n");

		
		depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
 		Mat flipped = Mat::zeros(depthf.size(), CV_8UC1);
		rectangle(findChar, Point(169,89), Point(470,390), Scalar(255, 255, 255));
		//namedWindow("HandSegment");
		namedWindow("Image Being Written");


		if(true) 
		{
			
			Mat hsv, mask;
			Mat hsvMask1 = Mat::zeros(rgbMat.size(), CV_8UC1);
			Mat hsvMask2 = Mat::zeros(rgbMat.size(), CV_8UC1);
			cvtColor(rgbMat, hsv, CV_RGB2HSV);
			inRange(hsv, Scalar(0, 0, 0), Scalar(30, 255, 255), hsvMask1);
			inRange(hsv, Scalar(150, 0, 0), Scalar(180, 255, 255), hsvMask2);
			add(hsvMask1, hsvMask2, mask);
			int close=255, iclose=0, jclose=0;

			for(int i=0; i<480; i++)
			{
				for(int j=0; j<640; j++)
				{
					uchar value = mask.at<uchar>(i,j);
					uchar intensity = depthf.at<uchar>(i,j);
					if(close > intensity)
					{
						close = intensity;
						iclose = i;
						jclose = j; 
					}
						

				}
			}
			for(int i=0; i<480; i++)
			{
				for(int j=0; j<640; j++)
				{
					uchar depth = depthf.at<uchar>(i,j);
					uchar &intensity = handSegmented.at<uchar>(i,j);
					if(depth>=close&& depth<=close+8)
						intensity = 255;
				}
			}
			//imshow("DepthImage",depthf);
			//imshow("HandSegment", handSegmented);
			/*int maxIntensity=0;
    		int xMax=0;
    		int yMax=0;
			Mat dst;
			distanceTransform(handSegmented,dst,CV_DIST_L2,3);
			for(int i=0;i<dist.rows;i++)
    		{
    			for(int j=0;j<dist.cols;j++)
    			{
    				float t=dist.at<float>(i,j);
    				if(t>maxIntensity)
    				{
    					maxIntensity=t;
    					xMax=i;
    					yMax=j;
    				}
    			}
    		}
			*/
			Mat display;
			flip(rgbMat,display,1);
			imshow("RGB Image",display);
			Mat display2=display.clone();
			Mat fist=Mat::zeros(640,480,CV_8UC1);
			fist=find_fist(display2);
			imshow("fist",fist);
			int num=0;
			for(int i=0;i<480;i++)
			{
				for(int j=0;j<640;j++)
				{
					uchar t1=handSegmented.at<uchar>(i,j);
					uchar t2=fist.at<uchar>(i,j);
					if((t1+t2)==510)
						num++;

				}
			}
			if (num>50)
			{
				if(flag==0)
				{
					flag=1;
				}
			}
			else
			{
				if (flag==1)
				{
					imwrite("Test1.jpg",findChar);
					for(int a = 0; a < 10; a++)
						dilate(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);

					for(int a = 0; a < 10; a++)
						erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);	
				/*dilate(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);
				dilate(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);
				erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);
				erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);*/
				//erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 100);

					int largest_area = 0;
					int largest_contour_index = 0;
					Mat F = findChar.clone();
					Rect bounding_rect,bounding_rect1;
					flag=0;
					Mat save;
					vector<vector<Point> > contours;
					vector<Vec4i> hierarchy;
					findContours(F,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
					for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
					{
    					double a=contourArea( contours[i],false); //  Find the area of contour
    					bounding_rect=boundingRect(contours[i]);
    					if(a>largest_area)
    					{
        					largest_area=a;
        					largest_contour_index=i; 
        					bounding_rect1=boundingRect(contours[i]);
        				}  
    				
    					if ( a < 60.0)
						{
						    	   				
    	   					rectangle(findChar,bounding_rect, Scalar(0) , CV_FILLED , 8 , 0 );

    	   				}	
					}

					save=findChar(bounding_rect1);

				//rectangle(findChar, bounding_rect, 255,5, 8,0);
					imwrite("digit.jpg",save);
					imwrite("Testing.jpg",findChar);
					savedImage=Mat::zeros(depthf.size(),CV_8UC1);
					findChar=Mat::zeros(depthf.size(),CV_8UC1);
					system("./code");
				}
			}
			if(flag==1)
			{
				if(((iclosep-iclose)<100||(iclose-iclose<100))&&((jclosep-jclose)<100||(jclose-jclose)<100))
				{
					circle(savedImage, Point(jclose,iclose),10,255,CV_FILLED, 8,0);
					iclosep=iclose;
					jclosep=jclose;
				}

				flip(savedImage,findChar, 1);
				imshow("Image Being Written",findChar);

			}
			else
			{
				flip(handSegmented,findChar,1);
				imshow("Image Being Written",findChar);

			}
			
		}
		
		char k = waitKey(1);
		if( k == 'q' )
		{
			if(flag==1)
			{
				imwrite("digit.jpg",findChar);
				system("./code");
			}
			destroyAllWindows();
			break;
		}
		if( k == 's' ) 
		{
			if(flag==0)
				flag=1;
			else if(flag==1)
			{	
				imwrite("Test1.jpg",findChar);
				for(int a = 0; a < 10; a++)
					dilate(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);

				for(int a = 0; a < 10; a++)
					erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);	
				/*dilate(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);
				dilate(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);
				erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);
				erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 1);*/
				//erode(findChar,findChar, Mat(), Point(-1, -1), 2, 1, 100);

				int largest_area = 0;
				int largest_contour_index = 0;
				Mat F = findChar.clone();
				Rect bounding_rect,bounding_rect1;
				flag=0;
				Mat save;
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				findContours(F,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
				for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
				{
    				double a=contourArea( contours[i],false); //  Find the area of contour
    				bounding_rect=boundingRect(contours[i]);
    				if(a>largest_area)
    				{
        				largest_area=a;
        				largest_contour_index=i; 
        				bounding_rect1=boundingRect(contours[i]);
        			}  
    				
    				if ( a < 60.0)
					{
						    	   				
    	   				rectangle(findChar,bounding_rect, Scalar(0) , CV_FILLED , 8 , 0 );

    	   			}	
				}

				save=findChar(bounding_rect1);

				//rectangle(findChar, bounding_rect, 255,5, 8,0);
				imwrite("digit.jpg",save);
				imwrite("Testing.jpg",findChar);
				savedImage=Mat::zeros(depthf.size(),CV_8UC1);
				findChar=Mat::zeros(depthf.size(),CV_8UC1);
				system("./code");


			}	
		}
		
	}
	
	device.stopVideo();
	device.stopDepth();
	return 0;
}

/*src=Mat::zeros(final2.x-final1.x,final2.x-final1.x,CV_8UC1);
		test=img(Rect(final1.x,final1.y,(final2.x-final1.x),(final2.y-final1.y)));
		test.copyTo(src(0,0,final2.x-final1.x,final2.y-final1.y));
*/