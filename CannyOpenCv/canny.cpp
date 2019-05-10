#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define SIZE 5

using namespace std;
using namespace cv;

Mat gn, imgFinal;
int tL, tH;

void imagePadded(Mat& imageIn, Mat& imageOut){

  int pad = floor(SIZE/2);

  imageOut = Mat::zeros(imageIn.rows + 2 * pad, imageIn.cols + 2 * pad, imageIn.type());

  for(int i=pad; i<imageOut.rows-pad; i++){
    for(int j=pad; j<imageOut.cols-pad; j++){
      imageOut.at<uchar>(i,j) = imageIn.at<uchar>(i-pad, j-pad);
    }
  }
}

void gaussianFilter(Mat&imageOut, Mat& imgGaussian){

  int pad = floor(SIZE/2);

  imgGaussian = Mat::zeros(imageOut.rows, imageOut.cols, imageOut.type());

  Mat kernelGauss = Mat::zeros(SIZE, SIZE, CV_32FC1); //filtro per gauss

  short sigma = 3; //angolo orientazione gradiente

  //inserisco i coefficienti di Gauss
  for(int i=-pad; i<=pad; i++){
    for(int j=-pad; j<=pad; j++){
      //calcolo i coefficienti con la formula di Gauss
      kernelGauss.at<float>(i+pad, j+pad) = exp( -((pow(i, 2)/(2*sigma)) + (pow(j, 2)/ (2*sigma)) ));
    }
  }

  float min = kernelGauss.at<float>(0, 0); //prendo il valore più piccolo

  //costante moltiplicativa
  float K = 1/min;

  for(int i=-pad; i<=pad; i++){
    for(int j=-pad; j<=pad; j++){
      //normalizzo i coefficienti approssimando i valori a quelli più vicini
      kernelGauss.at<float>(i+pad, j+pad) = round(kernelGauss.at<float>(i+pad, j+pad)*K);
    }
  }

  //prendo la somma dei coefficienti del filtro Gaussiano
  float sum = 0;

  for(int i=-pad; i<=pad; i++){
    for(int j=-pad; j<=pad; j++){

      sum += kernelGauss.at<float>(i+pad, j+pad);
    }
  }


  float accumGauss;

  for(int i=pad; i<imageOut.rows-pad; i++){
    for(int j=pad; j<imageOut.cols-pad; j++){

      accumGauss = 0;

      for(int r=-pad; r<=pad; r++){
        for(int c=-pad; c<=pad; c++){
          //casto i valori float ad int
          accumGauss += (int)imageOut.at<uchar>(i+r, j+c)*kernelGauss.at<float>(r+pad, c+pad);
        }
      }

      imgGaussian.at<uchar>(i, j) = round(accumGauss/sum);
    }
  }
}

void sobelFilter(Mat& imgGaussian, Mat& gx, Mat& gy, Mat& magnitudo, Mat& alfa){

  int pad = floor(SIZE/2);

  int padSobel = floor(3/2);

  Mat xSobel = (Mat_<char>(3, 3) <<   -1, 0, 1,
                                      -2, 0, 2,
                                      -1, 0, 1
);

  Mat ySobel = (Mat_<char>(3, 3)<<    1, 2, 1,
                                      0, 0, 0,
                                      -1, -2,-1
);

gx = Mat::zeros(imgGaussian.rows, imgGaussian.cols, imgGaussian.type());

gy = Mat::zeros(imgGaussian.rows, imgGaussian.cols, imgGaussian.type());

int xAcc, yAcc;

magnitudo = Mat::zeros(imgGaussian.rows, imgGaussian.cols, CV_32FC1);     //magnitudo gradiente (in float)

alfa = Mat::zeros(imgGaussian.rows, imgGaussian.cols, CV_32FC1);          //direzione gradiente (in float)

for(int i=pad; i<imgGaussian.rows-pad; i++){
  for(int j=pad; j<imgGaussian.cols-pad; j++){

    xAcc = yAcc = 0;

    for(int k=-padSobel; k<=padSobel; k++){
      for(int l=-padSobel; l<=padSobel; l++){

        xAcc += imgGaussian.at<uchar>(i+k, j+l)*xSobel.at<char>(k+padSobel, l+padSobel);
        yAcc += imgGaussian.at<uchar>(i+k, j+l)*ySobel.at<char>(k+padSobel, l+padSobel);
      }
    }

    //faccio in modo che non si ottengano valori negativi
    //dato che alcuni coefficienti di Sobel sono negativi
    xAcc = abs(xAcc);
    yAcc = abs(yAcc);

    //effettuo un processo simile alla sogliatura, per fare in modo
    //che i valori dei gradienti non superino il valore massimo (255)

    xAcc = (xAcc>255) ? 255 : xAcc;
    yAcc = (yAcc>255) ? 255 : yAcc;

    //salvo i gradienti
    gx.at<uchar>(i-pad, j-pad) = xAcc;
    gy.at<uchar>(i-pad, j-pad) = yAcc;

    //calcolo magnitudo e orientazione secondo le formule
    //l'angolo è arctg(gy/gx)*180/pigreco
    alfa.at<float>(i-pad, j-pad) = (atan2(yAcc, xAcc)*180)/M_PI;
    magnitudo.at<float>(i-pad, j-pad) = xAcc + yAcc;
  }
}

}

//La soppressione dei non-massimi porta all'eliminazione degli edge ambigui (quelli con valori piccoli)
//lavora sulla magnitudo prendendo in considerazione la direzione del gradiente (alfa)
//l'immagine finale (gn) sarà l'immagine con soltanto gli edge sicuri

void nonMaximaSuppression(Mat& gn, Mat& magnitudo, Mat& alfa){

  int pad = floor(SIZE/2);

  int dir = 0;

  float angle = 0;

  gn = Mat::zeros(magnitudo.rows, magnitudo.cols, magnitudo.type());

  //scorriamo la matrice della direzione (alfa)

  for(int i=pad; i<gn.rows-pad; i++){
    for(int j=pad; j<gn.cols-pad; j++){
      //prendiamo in esame la direzione di ogni pixel
      //salvandoci di ognuno la direzione
      angle = alfa.at<float>(i, j);

      //l'angolo è ortogonale all'edge (formano un angolo di 90 gradi)

      //se l'angolo è in uno di questi range, allora è orizzontale -> edge verticale
      if(-22.5 <= angle <= 22.5 || 157.5 <= angle <= 180 || -180 <= angle <= -157.5){
        dir = 0; //verticale
      }
      //l'angolo è verticale -> edge orizzontale
      else if(67.5 <= angle <= 112.5 || -112.5 <= angle <= -67.5){
        dir = 90; //orizzontale
      }
      //l'angolo va da in alto a destra a in basso a sinistra -> edge diagonale opposta
      else if(-157.5 <= angle <= -112.5 || 22.5 <= angle <= 67.5){
        dir = 45;
      }
      //l'angolo va da in alto a sinistra a in basso a destra -> edge diagonale opposta
      else if(-67.5 <= angle <= -22.5 || 112.5 <= angle <= 157.5){
        dir = 135;
      }

      //si controllano i due valori più vicini ad ogni pixel in esame per sopprimere i non-massimi
      if(dir == 0){ //edge verticale
        if(magnitudo.at<float>(i, j) >= magnitudo.at<float>(i, j-1) && magnitudo.at<float>(i, j) >= magnitudo.at<float>(i, j+1)) {
          gn.at<float>(i, j) = magnitudo.at<float>(i, j);
        }
      }
      else if(dir == 90){ //edge orizzontale
        if(magnitudo.at<float>(i, j) >= magnitudo.at<float>(i-1, j) && magnitudo.at<float>(i, j) >= magnitudo.at<float>(i+1, j)) {
          gn.at<float>(i, j) = magnitudo.at<float>(i, j);
        }
      }
      else if(dir == 45){ //obliquo : top-sx -> bottom-dx
        if(magnitudo.at<float>(i ,j) >= magnitudo.at<float>(i-1, j-1) && magnitudo.at<float>(i, j) >= magnitudo.at<float>(i+1, j+1)) {
          gn.at<float>(i, j) = magnitudo.at<float>(i, j);
        }
      }
      else if(dir == 135){ //obliquo : top-dx -> bottom-sx
        if(magnitudo.at<float>(i, j) >= magnitudo.at<float>(i+1, j-1) && magnitudo.at<float>(i, j) >= magnitudo.at<float>(i-1, j+1)) {
          gn.at<float>(i, j) = magnitudo.at<float>(i, j);
        }
      }

    }
  }

}

void thresholdHysteresis(Mat& gn, int tL, int tH){

  int pad = floor(SIZE/2);

  Mat gnL = Mat::zeros(gn.rows, gn.cols, gn.type());
  Mat gnH = Mat::zeros(gn.rows, gn.cols, gn.type());

  imgFinal = Mat::zeros(gn.rows, gn.cols, CV_8UC1);

  for(int i=pad; i<gn.rows-pad; i++){
    for(int j=pad; j<gn.cols-pad; j++){
      //scorro l'immagine per ricavarmi tutti gli edge forti
      if(gn.at<float>(i, j) >= tH){
        gnH.at<float>(i, j) = 255;
      }

      //controllo se ci sono edge deboli che rispettano questa condizione
      //e li salvo
      else if(gn.at<float>(i, j) < tH && gn.at<float>(i, j) >= tL){
        gnL.at<float>(i, j) = 255;
      }
    }
  }

  //scorro nella matrice degli edge forti
  //per controllare se ci sono edge deboli vicino ai forti presi in considerazione

  for(int i=pad; i<gnH.rows-pad; i++){
    for(int j=pad; j<gnH.cols-pad; j++){

      //se l'ij-esimo considerato ha un edge forte non nullo
      if(gnH.at<float>(i, j) != 0){
        //controllo se ci sono degli edge deboli nel suo vicinato non nulli
        //con il metodo della 8 connettività (cerco l'intorno del pixel considerato)
        for(int k=-1; k<=1; k++){
          for(int l=-1; l<=1; l++){
            //se il pixel considerato non è nullo
            if(gnL.at<float>(i+k, j+l) != 0){
              //l'edge debole diventa forte
              gnH.at<float>(i+k, j+l) = 255;
            }
            //altrimenti viene ignorato
          }
        }
      }
      imgFinal.at<uchar>(i, j) = round(gnH.at<float>(i, j));
    }
  }
  imshow("Hysteresis Image", imgFinal);

}

void minThreshold(int inputValue, void*){
  tL = inputValue;

  thresholdHysteresis(gn, tL, tH);
}

void maxThreshold(int inputValue, void*){
  tH = inputValue;
  thresholdHysteresis(gn, tL, tH);
}

int main(int argc, char** argv) {

  Mat imageIn = imread(argv[1], 0);

  if(imageIn.empty()){
    cout << "Error. No image found" << endl;
    return -1;
  }

  namedWindow("Hysteresis Image", WINDOW_AUTOSIZE);
  imshow("Lena Input", imageIn);

  Mat imageOut;

  //Paddiamo l'immagine
  imagePadded(imageIn, imageOut);

  imshow("Padded Image", imageOut);


  //formiamo un'immagine sfocata convolvendo l'immagine con il filtro Gaussiano
  Mat imgGaussian;

  gaussianFilter(imageOut, imgGaussian);

  imshow("Image Gaussian", imgGaussian);

  //usiamo il filtro di sobel per calcolare il gradiente
  Mat imgSobel;

  Mat gx;
  Mat gy;

  Mat magnitudo = Mat::zeros(imageOut.rows, imageOut.cols, CV_32FC1);     //magnitudo gradiente (in float)

  Mat alfa = Mat::zeros(imageOut.rows, imageOut.cols, CV_32FC1);          //direzione gradiente (in float)

  sobelFilter(imgGaussian, gx, gy, magnitudo, alfa);

  imshow("Sobel gx", gx);

  imshow("Sobel gy", gy);

  nonMaximaSuppression(gn, magnitudo, alfa);

  imshow("Non-Maxima Suppression Image", gn);

  int threshold;

  createTrackbar( "Min Threshold: ", "Hysteresis Image", &threshold, 255, minThreshold);
  createTrackbar( "Max Threshold: ", "Hysteresis Image", &threshold, 255, maxThreshold);

  waitKey(0);
  return 0;
}
