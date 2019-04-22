#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    VideoCapture cap(0); // Ouverture du framegrabber de la caméra
    if(!cap.isOpened())  // Vérification
        return -1;

    Mat acqImageGray; // Image 8 bits

    int successes = 0;

    std::vector<Point2f> pointBuf; // Vecteur des points détectés dans l'image
    std::vector<std::vector<Point2f> > imagePoints; // Vecteur de vecteurs des points détectés dans l'image
    std::vector<Point3f> objectBuf; // vecteur des points de référence
    std::vector<std::vector<Point3f> > objectPoints; // Vecteur de vecteurs des points de référence
    Size imageSize;

    // Génération du pattern de référence
    float squareSize = 1;
    Size boardSize(4, 11);
    for( int i = 0; i < boardSize.height; i++ )
	for( int j = 0; j < boardSize.width; j++ )
	    objectBuf.push_back(Point3f(float((2*j + i % 2)*squareSize), float(i*squareSize), 0));

    while(1)
    {
        Mat frame;
	cap >> frame; // Capture d'une nouvelle image
        cvtColor(frame, acqImageGray, CV_BGR2GRAY);
        imshow("imGray", acqImageGray);
        if(waitKey(10) != -1){

            //Find chessboard
            pointBuf.clear();
            
	    bool found = false;
	    found = findCirclesGrid(frame, boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);

            // Affichage et sauvegarde des points
            if(found){

                drawChessboardCorners( frame, boardSize, Mat(pointBuf), found );
                imshow("imCorners", frame);
                cvWaitKey(10);

                imagePoints.push_back(pointBuf);
                objectPoints.push_back(objectBuf);

                successes++;
            }
        }

        if( successes == 10)
	{
	   imageSize = frame.size();
	   break;
	}
    }

    cout << "Etalonnage avec: " << successes << " images" << endl;

    // Varaibles résultat
    Mat cameraMatrix = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    double rms = 0;
    // Etalonnage caméra
    rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, 
			  distCoeffs, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5|CV_CALIB_FIX_K6);


    std::cout << "rms : " << rms << std::endl;
    std::cout << "intr : " << cameraMatrix << std::endl;
    std::cout << "dist : " << distCoeffs << std::endl;

    // Rectification
    Mat R;
    Mat map1, map2;
    Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size(640, 480), 1, Size(640, 480), 0);
    initUndistortRectifyMap(cameraMatrix, distCoeffs, R,
      newCameraMatrix,
      imageSize, CV_32FC1, map1, map2);


    while(1)
    {
        Mat frame;
        cap >> frame; // capture nouvelle image
        Mat rectFrame; // Varible pour l'image rectifiée


	remap(frame, rectFrame, map1, map2, INTER_LINEAR);


	// Affichage de l'image rectifiée
        imshow("Image View", rectFrame);
        if(cvWaitKey(30) >= 0) break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

