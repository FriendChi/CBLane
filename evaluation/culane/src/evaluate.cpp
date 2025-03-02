/*************************************************************************
	> File Name: evaluate.cpp
	> Author: Xingang Pan, Jun Li
	> Mail: px117@ie.cuhk.edu.hk
	> Created Time: 2016年07月14日 星期四 18时28分45秒
 ************************************************************************/

#include "counter.hpp"
#include "spline.hpp"
#if __linux__
#include <unistd.h>
#elif _MSC_VER
#include "getopt.h"
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

void help(void)
{
	cout<<"./evaluate [OPTIONS]"<<endl;
	cout<<"-h                  : print usage help"<<endl;
	cout<<"-a                  : directory for annotation files (default: /data/driving/eval_data/anno_label/)"<<endl;
	cout<<"-d                  : directory for detection files (default: /data/driving/eval_data/predict_label/)"<<endl;
	cout<<"-i                  : directory for image files (default: /data/driving/eval_data/img/)"<<endl;
	cout<<"-l                  : list of images used for evaluation (default: /data/driving/eval_data/img/all.txt)"<<endl;
	cout<<"-w                  : width of the lanes (default: 10)"<<endl;
	cout<<"-t                  : threshold of iou (default: 0.4)"<<endl;
	cout<<"-c                  : cols (max image width) (default: 1920)"<<endl;
	cout<<"-r                  : rows (max image height) (default: 1080)"<<endl;
	cout<<"-s                  : show visualization"<<endl;
	cout<<"-f                  : start frame in the test set (default: 1)"<<endl;
}


void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes, float x_factor, float y_factor);
void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane);

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

int main(int argc, char **argv)
{
    // 初始化一些默认参数
    string anno_dir = "/data/driving/eval_data/anno_label/";  // 标注车道线文件目录
    string detect_dir = "/data/driving/eval_data/predict_label/";  // 检测车道线文件目录
    string im_dir = "/data/driving/eval_data/img/";  // 图像文件目录
    string list_im_file = "/data/driving/eval_data/img/all.txt";  // 存储所有图像文件名的文件路径
    string output_file = "./output.txt";  // 结果输出文件
    int width_lane = 10;  // 车道线宽度
    double iou_threshold = 0.4;  // IoU 阈值
    int im_width = 1920;  // 图像宽度
    int im_height = 1080;  // 图像高度
    int oc;  // 用于解析命令行参数的变量
    bool show = false;  // 是否显示图像
    int frame = 1;  // 帧编号
    double x_factor = 1.0;  // x 方向缩放因子
    double y_factor = 1.0;  // y 方向缩放因子

	// 使用 getopt 解析命令行参数，允许用户覆盖默认参数
    while((oc = getopt(argc, argv, "ha:d:i:l:w:t:c:r:sf:o:x:y:")) != -1)
    {
        switch(oc)
        {
            case 'h':  // 显示帮助信息
                help();
                return 0;
            case 'a':  // 标注文件目录
                anno_dir = optarg;
                break;
            case 'd':  // 检测结果文件目录
                detect_dir = optarg;
                break;
            case 'i':  // 图像文件目录
                im_dir = optarg;
                break;
            case 'l':  // 包含所有图像文件名的文件路径
                list_im_file = optarg;
                break;
            case 'w':  // 车道线宽度
                width_lane = atoi(optarg);
                break;
            case 't':  // IoU 阈值
                iou_threshold = atof(optarg);
                break;
            case 'c':  // 图像宽度
                im_width = atoi(optarg);
                break;
            case 'r':  // 图像高度
                im_height = atoi(optarg);
                break;
            case 's':  // 是否显示图像
                show = true;
                break;
            case 'f':  // 帧编号
                frame = atoi(optarg);
                break;
            case 'o':  // 结果输出文件路径
                output_file = optarg;
                break;
            case 'x':  // x 方向缩放因子
                x_factor = atof(optarg);
                break;
            case 'y':  // y 方向缩放因子
                y_factor = atof(optarg);
                break;
        }
    }


	cout<<"------------Configuration---------"<<endl;
	cout<<"anno_dir: "<<anno_dir<<endl;
	cout<<"detect_dir: "<<detect_dir<<endl;
	cout<<"im_dir: "<<im_dir<<endl;
	cout<<"list_im_file: "<<list_im_file<<endl;
	cout<<"width_lane: "<<width_lane<<endl;
	cout<<"iou_threshold: "<<iou_threshold<<endl;
	cout<<"im_width: "<<im_width<<endl;
	cout<<"im_height: "<<im_height<<endl;
	cout<<"x_factor: "<<x_factor<<endl;
	cout<<"y_factor: "<<y_factor<<endl;
	cout<<"-----------------------------------"<<endl;
	cout<<"Evaluating the results..."<<endl;
	// this is the max_width and max_height

	// 如果车道线宽度无效，输出错误并退出
	if(width_lane<1)
	{
		cerr<<"width_lane must be positive"<<endl;
		help();
		return 1;
	}

	// 尝试打开存储图像文件名的文件，失败则报错并退出
	ifstream ifs_im_list(list_im_file, ios::in);
	if(ifs_im_list.fail())
	{
		cerr<<"Error: file "<<list_im_file<<" not exist!"<<endl;
		return 1;
	}

    // 创建一个计数器，用于统计车道线匹配情况 (TP, FP, FN 等)
    Counter counter(im_width, im_height, iou_threshold, width_lane);

    vector<int> anno_match;  // 标注匹配结果的占位符
    string sub_im_name;  // 存储每个图像文件的名称
    vector<string> filelists;  // 存储所有图像文件的列表

    // 从图像列表文件中逐行读取图像文件名
    while (getline(ifs_im_list, sub_im_name)) {
        filelists.push_back(sub_im_name);  // 将每个图像文件名存入 filelists
    }
    ifs_im_list.close();  // 关闭文件

    // 创建一个存储匹配结果的容器 tuple_lists
    vector<tuple<vector<int>, long, long, long, long>> tuple_lists;
    tuple_lists.resize(filelists.size());  // 根据文件列表的大小调整大小

    // 使用 OpenMP 并行处理每个图像文件的车道线匹配
    #pragma omp parallel for
    for (int i = 0; i < filelists.size(); i++)
    {

    

        // 获取图像文件名
        auto sub_im_name = filelists[i];
		// cout<<'sub_im_name:'<<sub_im_name<<endl;
        string full_im_name = im_dir + sub_im_name;  // 完整的图像文件路径
        string sub_txt_name = sub_im_name.substr(0, sub_im_name.find_last_of(".")) + ".lines.txt";  // 车道线标注文件名
	        // 从最后一个斜杠之后的位置提取文件名
	    // 查找最后一个斜杠的位置
    	size_t lastSlashPos = sub_txt_name.find_last_of('/');
    	string filename = sub_txt_name.substr(lastSlashPos + 1);
	string anno_file_name = anno_dir + filename;  // 完整的标注文件路径
        string detect_file_name = detect_dir + sub_txt_name;  // 完整的检测文件路径
		// cout<<"detect_file_name "<<detect_file_name<<endl<<"anno_file: "<<anno_file_name<<endl;
        // 读取标注车道线和检测车道线，按缩放因子缩放
        vector<vector<Point2f>> anno_lanes;  // 标注车道线
        vector<vector<Point2f>> detect_lanes;  // 检测车道线

		// if (fileExists(anno_file_name)) {
			
		// 	std::cout << "检测文件存在: " << anno_file_name << std::endl;
						
		// } else {
		// 	std::cout << "标注文件不存在: " << anno_file_name << std::endl;
		// }

		// if (fileExists(detect_file_name)) {
			
		// 	std::cout << "检测文件存在: " << detect_file_name << std::endl;
			
		// } else {
		// 	std::cout << "检测文件不存在: " << detect_file_name << std::endl;
		// }

        read_lane_file(anno_file_name, anno_lanes, x_factor, y_factor);  // 读取标注车道线
        read_lane_file(detect_file_name, detect_lanes, x_factor, y_factor);  // 读取检测车道线

        // 统计当前图像的 TP, FP, FN 等，并存入 tuple_lists 中
        tuple_lists[i] = counter.count_im_pair(anno_lanes, detect_lanes);
    }

    // 初始化 TP、FP、TN、FN 的统计变量
    long tp = 0, fp = 0, tn = 0, fn = 0;
    
    // 遍历所有图像的统计结果，累加 TP、FP 和 FN
    for (auto result : tuple_lists) {
        tp += get<1>(result);
        fp += get<2>(result);
        // tn = get<3>(result);  // TN 未使用
        fn += get<4>(result);
    }

    // 将累积的 TP、FP 和 FN 设置到 counter 对象中
    counter.setTP(tp);
    counter.setFP(fp);
    counter.setFN(fn);

    // 计算精准度（Precision）、召回率（Recall）和 F1 分数
    double precision = counter.get_precision();
    double recall = counter.get_recall();
    double F = 2 * precision * recall / (precision + recall);

	cerr<<"finished process file"<<endl;
	cout<<"precision: "<<precision<<endl;
	cout<<"recall: "<<recall<<endl;
	cout<<"Fmeasure: "<<F<<endl;
	cout<<"----------------------------------"<<endl;

	ofstream ofs_out_file;
	ofs_out_file.open(output_file, ios::out);
	ofs_out_file<<"file: "<<output_file<<endl;
	ofs_out_file<<"tp: "<<counter.getTP()<<" fp: "<<counter.getFP()<<" fn: "<<counter.getFN()<<endl;
	ofs_out_file<<"precision: "<<precision<<endl;
	ofs_out_file<<"recall: "<<recall<<endl;
	ofs_out_file<<"Fmeasure: "<<F<<endl<<endl;
	ofs_out_file.close();
	return 0;
}

void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes, float x_factor, float y_factor)
{
	lanes.clear();
	ifstream ifs_lane(file_name, ios::in);
	if(ifs_lane.fail())
	{
		return;
	}

	string str_line;
	while(getline(ifs_lane, str_line))
	{
		vector<Point2f> curr_lane;
		stringstream ss;
		ss<<str_line;
		double x,y;
		while(ss>>x>>y)
		{
			curr_lane.push_back(Point2f(x* x_factor, y* y_factor));
		}
		lanes.push_back(curr_lane);
	}

	ifs_lane.close();
}

void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane)
{
	Mat img = imread(full_im_name, 1);
	Mat img2 = imread(full_im_name, 1);
	vector<Point2f> curr_lane;
	vector<Point2f> p_interp;
	Spline splineSolver;
	Scalar color_B = Scalar(255, 0, 0);
	Scalar color_G = Scalar(0, 255, 0);
	Scalar color_R = Scalar(0, 0, 255);
	Scalar color_P = Scalar(255, 0, 255);
	Scalar color;
	for (int i=0; i<anno_lanes.size(); i++)
	{
		curr_lane = anno_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		if (anno_match[i] >= 0)
		{
			color = color_G;
		}
		else
		{
			color = color_G;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	bool detected;
	for (int i=0; i<detect_lanes.size(); i++)
	{
		detected = false;
		curr_lane = detect_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		for (int n=0; n<anno_lanes.size(); n++)
		{
			if (anno_match[n] == i)
			{
				detected = true;
				break;
			}
		}
		if (detected == true)
		{
			color = color_B;
		}
		else
		{
			color = color_R;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	namedWindow("visualize", 1);
	imshow("visualize", img);
	namedWindow("visualize2", 1);
	imshow("visualize2", img2);
}
