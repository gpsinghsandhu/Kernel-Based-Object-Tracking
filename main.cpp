#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <math.h>
#include <conio.h>


cv::Mat kernel;
float kernel_sum = 0;
int count=0;
float pi1 = 3.142;
int total_iter = 0;
int frame_count = 0;
int model = 0;
float alpha = 0.99;

int no_of_bins = 16;
int range = 256;
int bin_width = cvRound(float(range)/float(no_of_bins)); 

cv::Rect box,next_box;
bool drawing_box = false;
bool selected = false;

void reduce_color_space(cv::Mat &input_image,cv::Mat &output_image,int div_factor);
void create_kernel(cv:: Mat &kernel);
cv::Mat kernel_on_patch(cv::Mat &input_image,int *patch_centre,cv::Mat input_kernel);
cv::Mat index_function(cv::Mat &input_image,int no_of_bins);
cv::Mat create_target_model(cv::Mat &input_image,int *patch_centre,cv::Mat &input_kernel,int no_of_bins);
cv::Mat detect_object(cv::Mat &input_image,cv::Rect &box);
cv::Mat assign_weight(cv::Mat &input_image,cv::Mat &target_model,cv::Mat &target_candidate,cv::Rect &box);
int check_bin_for_pixel(int pixel, int no_of_bins, int range);
float calc_bhattacharya(cv::Mat &target_model,cv::Mat &target_candidate);

void create_mouse_callback(int event,int x,int y,int flag,void* param);


void main()
{
	cv::Mat orig_img,temp_img,target_model,target_candidate,weight;

	int curr_pos[2] = {0,0};
	int next_pos[2] = {0,0};

	cv::VideoCapture start_capture;
	start_capture = cv::VideoCapture("tracking_video.avi");
	
	for(int i = 0; i<5; i++)
		start_capture.read(orig_img);

	//imshow("test",orig_img);

	int patch_centre[2] = {0,0};
	int patch_centre1[2] = {0,0};

	
	cv::namedWindow("original image");

	temp_img = orig_img.clone();

	cv::setMouseCallback("original image",create_mouse_callback,(void*) &temp_img);

	cv::imshow("original image",orig_img);

	while(selected == false)
	{
		cv::Mat temp;

		temp_img.copyTo(temp);

		if( drawing_box ) 
			cv::rectangle( temp, box,cv::Scalar(0),2);

		cv::imshow("original image", temp );

		if( cv::waitKey( 15 )==27 ) 
			break;
	}

	if(box.width%2==0)
		box.width++;
	if(box.height%2==0)
		box.height++;
	//cv::imshow("gp",orig_img);
	cv::waitKey(0);
	target_model = detect_object(orig_img,box);

	cv::waitKey(0);

	while(1)
	{
		count++;
		if(!start_capture.read(orig_img))
			break;	

		//start_capture.read(orig_img);
		//start_capture.read(orig_img);
		
		frame_count++;

		/*if(frame_count == 10)
		{
			frame_count = 0;
			target_model = target_candidate.clone();
		}*/

		

		for(int k=0;k<20;k++)
		{
			target_candidate = detect_object(orig_img,box);
			weight = assign_weight(orig_img,target_model,target_candidate,box);
		
			float num_x = 0.0;
			float den = 0.0;
			float num_y = 0.0;
			float centre = static_cast<float>((weight.rows-1)/2.0);
			double mult = 0.0;
			float norm_i = 0.0;
			float norm_j = 0.0;
			next_box.x = box.x;
			next_box.y = box.y;
			next_box.width = box.width;
			next_box.height = box.height;

			for(int i=0;i<weight.rows;i++)
			{
				for(int j=0;j<weight.cols;j++)
				{
					norm_i = static_cast<float>(i-centre)/centre;
					norm_j = static_cast<float>(j-centre)/centre;
					mult = pow(norm_i,2)+pow(norm_j,2)>1.0?0.0:1.0;
					num_x += static_cast<float>(alpha*norm_j*weight.at<float>(i,j)*mult);
					num_y += static_cast<float>(alpha*norm_i*weight.at<float>(i,j)*mult);
					den += static_cast<float>(weight.at<float>(i,j)*mult);
				}
			}
			
			next_box.x += static_cast<int>((num_x/den)*centre);
			next_box.y += static_cast<int>((num_y/den)*centre);

			//std::cout << "\n" << k;

			if((next_box.x-box.x)<1 && (next_box.y-box.y)<1)
			{
				//std::cout <<"\n \n success\n" << k;
				total_iter += k;
				break;
			}
			else
			{
				box.x = next_box.x;
				box.y = next_box.y;
			}
			if(box.x + box.width >= orig_img.cols || box.x <= 0 || box.y + box.height >= orig_img.rows || box.y <= 0)
			{
				_getch();
			}
		
		}

		float dist = 0.0;
		dist = calc_bhattacharya(target_model,target_candidate);
		//std::cout << '\n' << "Bhattacharya Distance : " << dist;

		if(dist < 0.6 && frame_count > 10)
		{	
			//target_model = target_candidate.clone();
			frame_count = 0;
			//std::cout << "gp";
		}
		

		cv::rectangle(orig_img,box,cv::Scalar(0));

		cv::imshow("Tracking",orig_img);

		cv::waitKey(5);


	}

	cv::waitKey(0);


}



void reduce_color_space(cv::Mat &input_image,cv::Mat &output_image,int div_factor=64)
{

	output_image = input_image.clone();
	
	int no_rows = input_image.rows;
	int no_cols = input_image.cols*input_image.channels();

	if(input_image.isContinuous())
	{
		no_cols = no_rows*no_cols;
		no_rows = 1;
	}

	int n = static_cast<int>(log(static_cast<double>(div_factor))/log(2.0));

	uchar mask = 0xFF<<n;

	for(int i=0;i<no_rows;i++)
	{
		uchar* data = output_image.ptr<uchar>(i);

		for(int j=0;j<no_cols;j++)
		{
			*data++ = *data&mask + div_factor/2;
		}

	}

}

void create_kernel(cv::Mat &kernel)
{
	int ck_no_rows = kernel.rows;
	int ck_no_cols = kernel.cols;
	float ck_centre[2] = {float((ck_no_cols-1)/2), float((ck_no_rows-1)/2)};

	float parameter_cd = 0.1*pi1*ck_no_rows*ck_no_rows;
	std::cout << '\n' << parameter_cd;

	for(int i=0;i<ck_no_rows;i++)
	{
		for(int j=0;j<ck_no_cols;j++)
		{
			float x = (abs(i-ck_centre[0]));
			float y = (abs(j-ck_centre[1]));

			float n = static_cast<float>(parameter_cd*(1.0-((x*x+y*y)/(ck_centre[0]*ck_centre[0]))));
			float m = n<0?0:n;
			kernel.at<float>(i,j) = m;

		}
	}

	//cv::normalize(kernel, kernel,1.0,0.0);

	for(int i=0;i<ck_no_rows;i++)
	{
		for(int j=0;j<ck_no_cols;j++)
		{
			kernel_sum += kernel.at<float>(i,j);
		}
	}
}

cv::Mat create_target_model(cv::Mat &input_image, int *patch_centre,cv::Mat &input_kernel,int no_of_bins)
{

	//const int size[3] = {no_of_bins,1};
	cv::Mat target_model(3,no_of_bins,CV_32F,cv::Scalar(0));

	int no_of_channels = input_image.channels();

	int ctm_no_cols = input_kernel.cols;
	int ctm_no_rows = input_kernel.rows;

	cv::Vec3f curr_pixel_value;
	cv::Vec3f bin_value;
	int x_img = patch_centre[0]-(ctm_no_cols-1)/2;
	int y_img = patch_centre[1]-(ctm_no_rows-1)/2;
	
	
	for(int i=0;i<target_model.rows;i++)
		target_model.at<float>(i,0) = 1e-10;

	/*std::cout << '\n';
	for(int i=0;i<bgr_planes[0].rows;i++)
		for(int j=0;j<bgr_planes[0].cols;j++)
		{
			std::cout << static_cast<int>(bgr_planes[0].at<uchar>(i,j)) << '\t';
		}
*/

		x_img = patch_centre[0]-(ctm_no_cols-1)/2;

		for(int i=0;i<ctm_no_rows;i++)
		{
			y_img = patch_centre[1]-(ctm_no_rows-1)/2;


			for(int j=0;j<ctm_no_rows;j++)
			{
				curr_pixel_value = input_image.at<cv::Vec3b>(x_img,y_img);
				bin_value[0] = (curr_pixel_value[0]/no_of_bins);
				bin_value[1] = (curr_pixel_value[1]/no_of_bins);
				bin_value[2] = (curr_pixel_value[2]/no_of_bins);
				target_model.at<float>(0,bin_value[0]) += input_kernel.at<float>(i,j)/(kernel_sum);
				target_model.at<float>(1,bin_value[1]) += input_kernel.at<float>(i,j)/(kernel_sum);
				target_model.at<float>(2,bin_value[2]) += input_kernel.at<float>(i,j)/(kernel_sum);
				y_img++;
			}
			x_img++;
		}


	return target_model;
}

void create_mouse_callback(int event,int x,int y,int flag,void* param)
{	
	cv::Mat *image = (cv::Mat*) param;
	switch( event ){
		case CV_EVENT_MOUSEMOVE: 
			if( drawing_box ){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;

		case CV_EVENT_LBUTTONDOWN:
			drawing_box = true;
			box = cv::Rect( x, y, 0, 0 );
			break;

		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			cv::rectangle(*image,box,cv::Scalar(0),2);
			selected = true;
			break;
	}

}

cv::Mat detect_object(cv::Mat &input_image,cv::Rect &box)
{
	int kernel_size = box.height>box.width?box.width:box.height;
	cv::Mat kernel_for_now(kernel_size,kernel_size,CV_32F,cv::Scalar(0));

	create_kernel(kernel_for_now);

	int patch_centre[2] = {box.y+box.height/2,box.x+box.width/2};
	cv::Mat target_model = create_target_model(input_image,patch_centre,kernel_for_now,16);

	return target_model;
}

cv::Mat assign_weight(cv::Mat &input_image,cv::Mat &target_model,cv::Mat &target_candidate,cv::Rect &box)
{
	int aw_no_of_rows = box.height>box.width?box.width:box.height;
	int aw_no_of_cols = box.height>box.width?box.width:box.height;

	cv::Mat weight(aw_no_of_rows,aw_no_of_cols,CV_32F,cv::Scalar(0));

	std::vector<cv::Mat> bgr_planes;

	split(input_image, bgr_planes);

	for(int i = 0; i < weight.rows; i++)
		for(int j = 0; j < weight.cols; j++)
		{
			weight.at<float>(i,j) = 1.0000;
		}


	int i_img = box.y;
	int j_img = box.x;

	int curr_pixel = 0;
	int bin_value = 0;

	for(int k = 0; k < 3;  k++)
	{	
		i_img = box.y;
		for(int i=0;i<aw_no_of_rows;i++)
		{
			j_img = box.x;
			for(int j=0;j<aw_no_of_cols;j++)
			{
				curr_pixel = static_cast<int>(bgr_planes[k].at<uchar>(i_img,j_img));
				bin_value = check_bin_for_pixel(curr_pixel,no_of_bins,range);

				weight.at<float>(i,j) *= static_cast<float>((sqrt(target_model.at<float>(bin_value + (k*no_of_bins),0)/target_candidate.at<float>(bin_value + (k*no_of_bins),0))));
			
			j_img++;		
			}

		i_img++;
		}
	}

	return weight;
}

int check_bin_for_pixel(int pixel, int no_of_bins, int range)
{
	int bin_value = 0;
	int bin_width = cvRound(float(range)/float(no_of_bins));

	if(pixel >= range)
	{
		return no_of_bins+1;
	}

	for(int i = 0; i < no_of_bins; i++)
	{
		if(pixel >= bin_width*i && pixel < bin_width*(i+1))
		{
			bin_value = i;
			break;
		}
	}

	return bin_value;

}

float calc_bhattacharya(cv::Mat &target_model,cv::Mat &target_candidate)
{
	float p_bar = 0.0;
	float dist = 0.0;

	for(int i = 0; i < target_model.rows; i++)
	{
		p_bar += sqrt((target_candidate.at<float>(i,0))*(target_model.at<float>(i,0)));
	}

	dist = sqrt(1-p_bar);

	return dist;

}

