#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void saveXYZ(const char* filename, Mat* mat, Mat* img)
{
    const double max_z = 1000;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat->rows; y++)
    {
        for(int x = 0; x < mat->cols; x++)
        {
            Vec3f point = mat->at<cv::Vec3f>(y,x);
	    Vec3f color = img->at<cv::Vec3b>(y,x);
            if(fabs(point(2) - max_z) < FLT_EPSILON || fabs(point(2)) > max_z)
		continue;
            fprintf(fp, "%f %f %f %i %i %i\n", point(0), point(1), point(2), static_cast<int>(color(2)), static_cast<int>(color(1)), static_cast<int>(color(0)));
        }
    }
    fclose(fp);

    //    cvSave("temp_xyz.yml", mat);
}

int main(int argc, char** argv )
{

    std::vector<Point2f> pointBuf;
    std::vector<std::vector<Point2f> > imagePointsLeft;
    std::vector<std::vector<Point2f> > imagePointsRight;
    std::vector<Point3f> objectBuf;
    std::vector<std::vector<Point3f> > objectPoints;

    // Génération du pattern de référence
    float squareSize = 1;
    Size boardSize(4, 11);
    for( int i = 0; i < boardSize.height; i++ )
	for( int j = 0; j < boardSize.width; j++ )
	    objectBuf.push_back(Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));
	    
    // Détection mires caméra gauche
    int successes = 0;
    for(int i=0; i<20; i++)
    {
        Mat acqImageGray;
        char imageName[256];
        sprintf(imageName, "./output/left_%02d.jpg", i);
        acqImageGray = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        imshow("imGray", acqImageGray);

	pointBuf.clear();
	    
	int found = 0;

	found = findCirclesGrid(acqImageGray, boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);

        // Affichage et sauvegarde des points
	if(found){
	    drawChessboardCorners( acqImageGray, boardSize, Mat(pointBuf), found );
	    imshow("imCorners", acqImageGray);
	    waitKey(10);

	    imagePointsLeft.push_back(pointBuf);
	    objectPoints.push_back(objectBuf);

	    successes++;
	}
    }
    cout << "Mires caméra gauche : " << successes << endl;

    // Détection mires caméra droite
    successes = 0;
    for(int i=0; i<20; i++)
    {
        Mat acqImageGray;
        char imageName[256];
        sprintf(imageName, "./output/right_%02d.jpg", i);
        acqImageGray = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
        imshow("imGray", acqImageGray);

        pointBuf.clear();
            
        int found = 0;

	found = findCirclesGrid(acqImageGray, boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);


        // Affichage et sauvegarde des points
        if(found){

            drawChessboardCorners( acqImageGray, boardSize, Mat(pointBuf), found );
            imshow("imCorners", acqImageGray);
            cvWaitKey(10);

            imagePointsRight.push_back(pointBuf);

            successes++;
        }
    }

    // Etalonnage des caméras
    cout << "Mires caméra droite : " << successes << endl;
    cout << "Etalonnage caméras..." << endl;

    Mat cameraMatrixLeft = Mat(3, 3, CV_32FC1);
    Mat cameraMatrixRight = Mat(3, 3, CV_32FC1);
    Mat distCoeffsLeft, distCoeffsRight;
    vector<Mat> rvecsLeft, rvecsRight;
    vector<Mat> tvecsLeft, tvecsRight;
    double rmsLeft, rmsRight;
    Size imageSize(640, 480);

    rmsLeft = calibrateCamera(objectPoints, imagePointsLeft, imageSize, cameraMatrixLeft, 
			  distCoeffsLeft, rvecsLeft, tvecsLeft);
    rmsRight = calibrateCamera(objectPoints, imagePointsRight, imageSize, cameraMatrixRight, 
			  distCoeffsRight, rvecsRight, tvecsRight);
    
    std::cout << "cameraMatrixLeft : " << cameraMatrixLeft << std::endl;
    std::cout << "distCoeffsLeft : " << distCoeffsLeft << std::endl;
    std::cout << "rmsLeft : " << rmsLeft << std::endl;

    std::cout  << endl;

    std::cout << "cameraMatrixRight : " << cameraMatrixRight << std::endl;
    std::cout << "distCoeffsRight : " << distCoeffsRight << std::endl;
    std::cout << "rmsRight : " << rmsRight << std::endl;


    // Stéréo calibration
    Mat R, T, E, F;
    double rms;
    rms = stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight, cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, imageSize, R, T, E, F);

    std::cout << "rms: " << rms << std::endl;
    std::cout << "R : " << R << std::endl;
    std::cout << "T : " << T << std::endl;


    // Stéréo rectification
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrixLeft, distCoeffsLeft, cameraMatrixRight, distCoeffsRight, imageSize, R, T, R1, R2, P1, P2, Q);// CV_CALIB_ZERO_DISPARITY

    // Rectification des images de test
    // Remarque : bien vérifier les chemins image
    Mat img0 = imread( "./data/left_rodeco1.jpg", CV_LOAD_IMAGE_COLOR);
    Mat img1 = imread( "./data/right_rodeco1.jpg", CV_LOAD_IMAGE_COLOR);

    if( img0.empty() || img1.empty() )
	cout << "Unable to load img0 or img1. Please check path." << endl;

    // Conversion rbg->gray
    Mat img0gray;
    Mat img1gray;
    cvtColor(img0, img0gray, CV_BGR2GRAY);
    cvtColor(img1, img1gray, CV_BGR2GRAY);

    Mat map0x, map0y, map1x, map1y;
    Mat imgU0, imgU1;
    initUndistortRectifyMap(cameraMatrixLeft, distCoeffsLeft, R1,
      P1,
      imageSize, CV_32FC1, map0x, map0y);
    initUndistortRectifyMap(cameraMatrixRight, distCoeffsRight, R2,
      P2,
      imageSize, CV_32FC1, map1x, map1y);

    remap(img0gray, imgU0, map0x, map0y, INTER_LINEAR);
    remap(img1gray, imgU1, map1x, map1y, INTER_LINEAR);
    imshow("Image View1", imgU0);
    imshow("Image View2", imgU1);
    cvWaitKey(0);
    /********** Affichage ***********/
    Mat canvas;
    int w = 640;
    int h = 480;
    canvas.create(h, w*2, CV_8UC3);

    Mat imgU0c, imgU1c;
    cvtColor(imgU0, imgU0c, CV_GRAY2RGB);
    cvtColor(imgU1, imgU1c, CV_GRAY2RGB);

    Mat canvasPart = canvas(Rect(0, 0, w, h));
    imgU0c.copyTo(canvasPart);
    Rect vroi1(cvRound(validRoi[0].x), cvRound(validRoi[0].y),
              cvRound(validRoi[0].width), cvRound(validRoi[0].height));
    rectangle(canvasPart, vroi1, Scalar(0,255,0), 3, 8);

    canvasPart = canvas(Rect(w, 0, w, h));
    imgU1c.copyTo(canvasPart);
    Rect vroi2(cvRound(validRoi[1].x), cvRound(validRoi[1].y),
              cvRound(validRoi[1].width), cvRound(validRoi[1].height));
    rectangle(canvasPart, vroi2, Scalar(0,255,0), 3, 8);

    for( int j = 0; j < canvas.rows; j += 16 )
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 0, 255), 1, 8);

    imshow("stereo1", canvas);
    cvWaitKey(0);

    /*********** Disparités ***********/

    Mat disp;
    Mat disp8;
    StereoSGBM sgbm;
    sgbm.SADWindowSize = 5;
    sgbm.numberOfDisparities = 112;
    sgbm.preFilterCap = 4;
    sgbm.minDisparity = 0;//-64
    sgbm.uniquenessRatio = 1;
    sgbm.speckleWindowSize = 150;
    sgbm.speckleRange = 2;
    sgbm.disp12MaxDiff = 10;
    sgbm.fullDP = false;
    sgbm.P1 = 600;
    sgbm.P2 = 2400;

    sgbm(imgU0, imgU1, disp);
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

    cv::Mat imgDisparity32F = Mat( disp.rows, disp.cols, CV_32F );
    disp.convertTo( imgDisparity32F, CV_32F, 1./16);
    cv::Mat Q_32F;
    Q.convertTo(Q_32F, CV_32F);

    imshow("stereo2", disp8);
    imwrite( "disp.jpg", disp8 );
    cvWaitKey(0);

    Mat xyz = Mat(disp.rows, disp.cols, CV_32FC3);
    reprojectImageTo3D(imgDisparity32F, xyz, Q, 1);
    saveXYZ("./stereo_xyzrgb.xyz", &xyz, &img0);

    std::string filename = "Q.xml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "Q" << Q; 

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
