#include <iostream>
#include <vector>
#include<opencv2>
using namespace std;
vector<vector<unsigned char> & noneMaxmiumsuppression(const vector<vector<unsigned char>> & magnitude,const vector<vector<float>> & angle){
    h=magnitude.size();
    w=magnitude[0].siae;
    vector<vector<unsigned char>>nms(h,vector<unsigned char>(w))
    for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            unsigned char mag=magnitude[y][x];
            float dir=angle[y][x];
            if(0<dir<22.5||157.5<dir<180){
                mag1=magnitude[x-1][y];
                mag2=magnitude[x+1][y];
            }
            if(22.5<dir<67.5){
                mag1=magnitude[x+1][y+1];
                mag2=magnitude[x-1][y-1];
            }
            if(67.5<dir<112.5){
                mag1=magnitude[x][y-1];
                mag2=magnitude[x][y+1];
            }
            if(112.5<dir<157.5){
                mag1=magnitude[x+1][y-1];
                mag1=magnitude[x-1][y+1];

            }
            if(mag>=mag1&&mag>=mag2){
                nms[y][x]=mag;
                return;
            }
            nms[i][j]=0;
        }
    }
    return nms;

}
vector<vector<unsigned char>> doubleThreshold(const vector<vector<unsigned char>>& nms){
    int h=nms.height;
    int w=nms.width;
    int threshold1;
    int threshold2;
    vector<unsigned char> result(h,vector<unsigned char>(w));
    for (int y=0;y<h;y++){
        for(int x=0;x<w;x++){
            if(nms[y][x]<threshold1){
                result[y][x]=255;
            }
            else if(nms[y][x<threshold2]){
                result[y][x]=75;
            }
            else{
                result[y][x]=0;
            }
        }
    }  
}
int main(){
   Mat image=imread(image.jpg);
   Mat gray;
   cvtCOLOR(image,gray,COLOR_BGR2GRAY);
   Mat blur;
   guassianBlur(gray,blur,(5,5),0);
   Mat binary;
   threshole(blur,binary,100,255,THRESHOLE_BINARY);
   Mat kernel=getStructingElement(MORPH_RECT(3,3));
   morphologtEx(binary,binary,MORPH_OPEN,kernel)
   morphologtEx(binary,binary,MORPH_CLOSE,kernel)
   vector<vector<Point> contours;
   findContours(binary,contours,RETR_EXTERNEL,CHAIN_APPROX_SIMPLE);
        for(int i=0;i<coutour.size();i++){
            area=contourerea(contours[i]);
            if(area<500) continue;
        Rect rect=boundingRect(contours[i]);
        h=rect.height;
        w=rect.width;
        ratio=h/w;
        if(0.5<ratio<2){
            rectangle(image,rect,Scalar(0,0,255),2)
        }
    
   }
}

