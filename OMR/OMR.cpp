#pragma once
#include "OMRKnnDescription.h"
#include <iostream>
#include <vector>
#include <math.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

/*垂直的侵蝕與膨脹，做完會抓到大部分的音符，少部分會殘缺*/
Mat SpecifyVerticalAxis(Mat& source)
{
	Mat result = source.clone();
	int verticalsize = result.rows / 800;
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
	erode(result, result, verticalStructure, Point(-1, -1));
	dilate(result, result, verticalStructure, Point(-1, -1));
	return result;
}

/*用水平的侵蝕與膨脹找出水平線*/
Mat SpecifyHorizontalAxis(Mat& source)
{
	Mat result = source.clone();
	int horizontalsize = result.cols / 10;
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));
	erode(result, result, horizontalStructure, Point(-1, -1));
	dilate(result, result, horizontalStructure, Point(-1, -1));
	return result;
}

/*把距離太靠近或太遠的水平線移除*/
void AdjustStaffLine(vector<Vec4i>& lines) {
	int number = 0;
	int lineSize = (int)lines.size();

	while (number < lineSize - 1) {
		if (number % 5 == 4) {
			int interval1 = 90 - (lines[number + 1][1] - lines[number][1]);

			//兩譜相間至少90(並沒有很嚴謹)
			if (interval1 > 1) {
				lines.erase(lines.begin() + number + 1);
				lineSize--;
			}
			else number++;
		}
		else {
			int interval = 14 - (lines[number + 1][1] - lines[number][1]);
			//距離太近的兩線刪下面那條
			if (interval > 1) {
				lines.erase(lines.begin() + number + 1);
				lineSize--;
			}
			//距離太遠會試著在跟下一條線比，如果可能是斷點就補線，否則刪上面那條
			else if (interval < -1) {
				lines.erase(lines.begin() + number);
				lineSize--;
			}
			else number++;
		}
	}
}

/*找出足夠長的水平線視為五線譜，並依照y軸5個一組分類，若水平線數量不為5的倍數則視為失敗*/
void FindStaffAndBorder(Mat& source, vector<Vec4i>& outputLines, vector<Vec2i>& horizontalBorders, Vec2i& verticalBorders) {
	verticalBorders[0] = source.cols >> 2;
	verticalBorders[1] = verticalBorders[0];
	HoughLinesP(source, outputLines, 1, CV_PI / 2, 10, source.rows >> 1, source.rows / 10);
	//用y來排序
	sort(outputLines.begin(), outputLines.end(),
		[](const Vec4i & a, const Vec4i & b) -> bool
	{
		return a[1] < b[1];
	});
	AdjustStaffLine(outputLines);
	if (outputLines.size() % 5 != 0) throw "error while finding staff (incorrect number of Horizotal line)";
	int lineSize = (int)outputLines.size() / 5;

	if (lineSize == 0) throw "can no find any Staff !!";
	else if (lineSize == 1) {
		horizontalBorders.push_back(Vec2i(0, source.cols));
		return;
	}

	//upperbound偷偷放寬到1.5，之後應該會改回來，正確的寫法應該為使用ROI的中心判斷，目前是用左上和右下判斷，如果稍微太高超過會失效
	int lowerBound = (outputLines[4][1] + outputLines[5][1]) >> 1;
	int upperBound = outputLines[0][1] - ((lowerBound - outputLines[4][1]) * 1.5);
	horizontalBorders.push_back(Vec2i(upperBound, lowerBound));
	for (int i = 1; i < lineSize - 1; i++) {
		upperBound = (outputLines[(i - 1) * 5 + 4][1] + outputLines[i * 5][1]) >> 1;
		lowerBound = (outputLines[i * 5 + 4][1] + outputLines[(i + 1) * 5][1]) >> 1;
		horizontalBorders.push_back(Vec2i(upperBound, lowerBound));
	}
	upperBound = (outputLines[(lineSize - 2) * 5 + 4][1] + outputLines[(lineSize - 1) * 5][1]) >> 1;
	lowerBound = outputLines[(lineSize - 1) * 5 + 4][1] + (outputLines[(lineSize - 1) * 5][1] - upperBound);
	horizontalBorders.push_back(Vec2i(upperBound, lowerBound));

	for (Vec4i line : outputLines) {
		if (line[0] < verticalBorders[0]) verticalBorders[0] = line[0];
		if (line[2] > verticalBorders[0]) verticalBorders[1] = line[2];
	}
}

/*輸入的兩個矩形，輸出一個較大的矩形將兩個矩形框住*/
Rect CombineRect(Rect rect1, Rect rect2) {
	int tl_x = min(rect1.tl().x, rect2.tl().x);

	int tl_y = min(rect1.tl().y, rect2.tl().y);
	int br_x = max(rect1.br().x, rect2.br().x);
	int br_y = max(rect1.br().y, rect2.br().y);
	return Rect(Point(tl_x, tl_y), Point(br_x, br_y));
}

int main()
{
	OMRKnnDescription knn("C:\\Users\\Mystia\\Downloads\\train\\training-set", "C:\\Users\\Mystia\\Downloads\\train\\note");
	char* text = new char[64];
	//載入灰階(單通道)
	Mat source = imread("C:\\Users\\Mystia\\Downloads\\12321.png", CV_LOAD_IMAGE_GRAYSCALE);
	int size = source.rows * source.cols;
	imshow("src", ~source);
	//五線譜有陰影，所以不同閥值可以濾出不同粗細的線，線條的粗細在後面找ROI的時候會有差
	Mat binaryThin(source.rows, source.cols, CV_8U);
	threshold(~source, binaryThin, 150, 255, THRESH_BINARY);
	Mat binaryThick(source.rows, source.cols, CV_8U);
	threshold(~source, binaryThick, 15, 255, THRESH_BINARY);

	Mat vertical = SpecifyVerticalAxis(binaryThin);
	Mat verticalCopy;
	vertical.copyTo(verticalCopy);

	Mat horizontal = SpecifyHorizontalAxis(binaryThick);
	///Mat ttttt;
	///resize(horizontal, ttttt, Size(horizontal.cols / 2.2, horizontal.rows / 2.2));
	///imshow("horizontal", ttttt);

	Mat hhh(horizontal.rows, horizontal.cols, CV_8UC3);
	cvtColor(horizontal, hhh, CV_GRAY2BGR);
	vector<Vec4i> lines;
	vector<Vec2i> lineHorizontalBorder;
	Vec2i lineVerticalBorder;
	FindStaffAndBorder(horizontal, lines, lineHorizontalBorder, lineVerticalBorder);
	//把左邊五線譜外的圖用反轉的原圖取代，這樣大括弧就不會斷掉了
	for (int i = 0; i < source.rows; i++) {
		for (int j = 0; j < lineVerticalBorder[0]; j++) {
			int position = j + i * source.cols;
			vertical.data[position] = ~source.data[position];
		}
	}
	///imshow("vertical", vertical);

	std::cout << "共有 : " << lines.size() << " 個五線譜" << std::endl;
	int number = 0;
	int counter = 0;
	for (Vec4i sline : lines) {
		if (counter >= 5) {
			counter = 0;
			number++;
		}
		std::cout << "第" << (number + 1) << "個五線譜，第" << (counter + 1) << "條線的y為" << sline[1] << std::endl;
		counter++;
		line(hhh, Point(sline[0], sline[1]), Point(sline[2], sline[3]), Scalar(0, 0, 255), 3);

		Point textPA = Point(sline[0] - 25, sline[1]);
		sprintf_s(text, 64, "%d", counter + number * 5);
		putText(hhh, text, textPA, FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, Scalar(0, 0, 255), 2, 8, false);
	}

	for (Vec2i bound : lineHorizontalBorder) {
		line(hhh, Point(0, bound[0]), Point(hhh.cols, bound[0]), Scalar(255, 0, 255), 2);
		line(hhh, Point(0, bound[1]), Point(hhh.cols, bound[1]), Scalar(0, 255, 0), 15);
	}
	///resize(hhh, hhh, Size(hhh.cols, hhh.rows));
	///imshow("horizontal2", hhh);
	/*********************************/

	/*尋找countour並尋找bounding box*/
	Mat colorImg;
	cvtColor(vertical, colorImg, CV_GRAY2BGR);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//先侵蝕再膨脹，因為現在是黑白反轉的狀態，音符是白色的
	//dilate(vertical, vertical, Mat(), Point(-1, -1), 1);
	//erode(vertical, vertical, Mat(), Point(-1, -1), 1);
	imshow("vertical", vertical);

	findContours(vertical, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	vector<vector<Point> > contours_poly(contours.size());
	/*用水平border分類成數個向量，再按照x排序*/
	vector<Rect>* orderedROI = new vector<Rect>[lineHorizontalBorder.size()];
	//把cotours最近似矩形後丟進vector
	int lineBorderSize = (int)lineHorizontalBorder.size();
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		Rect tmp = boundingRect(Mat(contours_poly[i]));

		for (int i = 0; i < lineBorderSize; i++) {
			//border的[0]為上限、[1]為下限，越上方數字越小
			if (lineHorizontalBorder[i][0] < tmp.tl().y && tmp.tl().y < lineHorizontalBorder[i][1]) {
				if (lineHorizontalBorder[i][0] < tmp.br().y && tmp.br().y < lineHorizontalBorder[i][1]) {
					orderedROI[i].push_back(tmp);
					break;
				}
				//有可能一個ROI橫跨兩個譜或以上，所以加這個判斷，但也最都只能處理橫跨兩個的，再多要再改
				else if ((i + 1) < lineBorderSize) {
					orderedROI[i].push_back(tmp);
					orderedROI[i + 1].push_back(tmp);
					break;
				}
			}
		}
	}

	for (int i = 0; i < lineBorderSize; i++) {
		sort(orderedROI[i].begin(), orderedROI[i].end(),
			[](const Rect & a, const Rect & b) -> bool
		{
			return a.tl().x < b.tl().x;
		});
	}

	//把面積過小的連通物件去跟前面或後面最接近且較大的物件合併
	for (int i = 0; i < lineHorizontalBorder.size(); i++) {
		//大小事會變動的，所以不能先算size
		for (int j = 0; j < orderedROI[i].size(); j++) {
			//if (q != orderedROI[i].size()) std::cout << 666;
			//如果後面的tl(左上)或bl(左下)跟自己交疊，就跟後面的ROI合併
			if (j < orderedROI[i].size() - 1 &&
				(orderedROI[i][j].contains(orderedROI[i][j + 1].tl()) || orderedROI[i][j].contains(Point(orderedROI[i][j + 1].tl().x, orderedROI[i][j + 1].tl().y + orderedROI[i][j + 1].height >> 1)))) {
				//先在第j個位置插入，再刪除j+1的rect兩次
				orderedROI[i].insert(orderedROI[i].begin() + j, 1, CombineRect(orderedROI[i][j], orderedROI[i][j + 1]));
				orderedROI[i].erase(orderedROI[i].begin() + j + 1);
				orderedROI[i].erase(orderedROI[i].begin() + j + 1);
			}
			//如果前面的br(右下)或tr(右上)跟自己交疊，就跟前面的ROI合併
			else if (j > 0 &&
				(orderedROI[i][j].contains(orderedROI[i][j - 1].br()) || orderedROI[i][j].contains(Point(orderedROI[i][j - 1].tl().x, orderedROI[i][j - 1].tl().y - orderedROI[i][j - 1].height >> 1)))) {
				//先在第j-1個位置插入，再刪除j的rect兩次
				orderedROI[i].insert(orderedROI[i].begin() + j - 1, 1, CombineRect(orderedROI[i][j], orderedROI[i][j - 1]));
				orderedROI[i].erase(orderedROI[i].begin() + j);
				orderedROI[i].erase(orderedROI[i].begin() + j);
				j--;
			}

		}
		//std::cout << orderedROI[i].size() << std::endl;
	}

	/*********************************/

	/*將剛剛找到的bounding box畫上標示，並註記為第幾排的第幾個，橫跨多欄的會再經過的欄都出現*/
	/// Draw polygonal contour + bonding rects + circles
	Scalar color = Scalar(0, 0, 255);
	Scalar fontColor = Scalar(255, 255, 0);
	for (int i = 0; i < lineHorizontalBorder.size(); i++)
	{
		for (int j = 0; j < orderedROI[i].size(); j++) {
			//drawContours(colorImg, contours_poly, i, color, 2, 8, vector<Vec4i>(), 0, Point());
			rectangle(colorImg, orderedROI[i][j].tl(), orderedROI[i][j].br(), color, 2, 8, 0);
			Point textP = Point(orderedROI[i][j].tl().x, orderedROI[i][j].tl().y - 10);
			Point textPA = Point(orderedROI[i][j].tl().x, orderedROI[i][j].tl().y - 20);

			sprintf_s(text, 64, "%d - %d", i, j);
			putText(colorImg, text, textP, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, fontColor, 0.5, 8, false);
			/*if (i == 0 && (j == 8 || j == 9 || j == 10 || j == 29)) {
				std::cout << j << std::endl;
				std::cout << " x = " << orderedROI[i][j].tl().x << " y = " << orderedROI[i][j].tl().y << std::endl;
				std::cout << " x = " << orderedROI[i][j].br().x << " y = " << orderedROI[i][j].br().y << std::endl;
			}*/
		}
	}

	/*********************************/


	for (int i = 0; i < lineHorizontalBorder.size(); i++) {
		for (int j = 0; j < orderedROI[i].size(); j++) {
			Mat sampleROI;
			verticalCopy(orderedROI[i][j]).copyTo(sampleROI);
			float r = knn.FindNearestSymbol(~sampleROI);
			if (r == -1) {
				Rect resizeROI(Point(orderedROI[i][j].tl().x - 5, orderedROI[i][j].tl().y - 5), Point(orderedROI[i][j].br().x + 5, orderedROI[i][j].br().y + 5));
				verticalCopy(resizeROI).copyTo(sampleROI);
				Mat sampleROICopy = sampleROI.clone();

				vector<Vec4i> noteLines;
				HoughLinesP(sampleROI, noteLines, 0.1, CV_PI, 45, 10, 10);

				for (Vec4i sline : noteLines) {
					line(sampleROICopy, Point(sline[0], sline[1]), Point(sline[2], sline[3]), Scalar(0), 2);
				}
				if (i == 2 && j == 27)imshow("gdfg", sampleROICopy);

				std::stringstream ss;
				ss << "C:\\Users\\Mystia\\Downloads\\zzzzzz\\train" << i << " " << j;

				findContours(sampleROICopy, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

				contours_poly.resize(contours.size());
				std::cout << "size = " << noteLines.size() << " : " << i << " , " << j << std::endl;
				for (int k = 0; k < contours.size(); k++)
				{
					std::stringstream st;
					st << ss.str() << " " << k << ".png";
					approxPolyDP(Mat(contours[k]), contours_poly[k], 3, true);
					Rect tmp = boundingRect(Mat(contours_poly[k]));
					Mat trainData;
					sampleROI(tmp).copyTo(trainData);
					if (tmp.height > 5 && tmp.width > 5)
						imwrite(st.str(), ~trainData);
				}

				//imwrite(ss.str(), );
				//imshow(ss.str(), sampleCopy);
			}
			//std::cout << i << " - " << j << "seam like  " << knn.GetSymbolName(r) << std::endl;
		}
	}


	//imshow("note", sampleCopy);


	/*for (int i = 0; i < lineHorizontalBorder.size(); i++) {
		for (int j = 0; j < orderedROI[i].size(); j++) {
			verticalCopy(orderedROI[i][j]).copyTo(sampleROI);
			float r = knn.FindNearestSymbol(~sampleROI);
			std::cout << i << " - " << j << "seam like  " << knn.GetSymbolName(r) << std::endl;
		}
	}*/
	//imwrite("C:\\Users\\Mystia\\Downloads\\data.png", colorImg);
	//verticalCopy(orderedROI[1][30]).copyTo(sampleROI);
	//imwrite("C:\\Users\\Mystia\\Downloads\\train0.png", ~sampleROI);
	//verticalCopy(orderedROI[1][21]).copyTo(sampleROI);
	//imwrite("C:\\Users\\Mystia\\Downloads\\train1.png", ~sampleROI);
	//verticalCopy(orderedROI[2][12]).copyTo(sampleROI);
	//imwrite("C:\\Users\\Mystia\\Downloads\\train2.png", ~sampleROI);
	//verticalCopy(orderedROI[2][24]).copyTo(sampleROI);
	//imwrite("C:\\Users\\Mystia\\Downloads\\train3.png", ~sampleROI);
	//verticalCopy(orderedROI[3][23]).copyTo(sampleROI);
	//imwrite("C:\\Users\\Mystia\\Downloads\\train4.png", ~sampleROI);
	//verticalCopy(orderedROI[0][56]).copyTo(sampleROI);
	//imwrite("C:\\Users\\Mystia\\Downloads\\train5.png", ~sampleROI);

	delete[] orderedROI;
	delete[] text;
	/// Show in a window
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", colorImg);

	waitKey();
	return 0;
}
