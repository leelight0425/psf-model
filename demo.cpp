#include<Eigen/Dense>
#include<vector>
using namespace std;
MatrixXd img2cols( const MatrixXd& image,int pad,int stride,int ksize){
    int h=image.rows();
    int w=image.cols():
    int out_h=(h+2*pad-ksize)/stride+1;
    int out_w=(w+2*pad-ksize)/stride+1;
    int N=out_w*out_h;
    int col=0;
    MatrixXd img_pad=MatrixXd::Zero(out_h,out_w);
    img_pad(pad,pad,h,w)=image;
    MatrixXd cols(ksize*ksize,N);
    for(int i=0;i<out_h;i++){
        for(int j=0;j<out_w;j++){
            int x=i*stride;
            int y=i*stride;
            int idx=0;
            for(int dy=0;dy<ksize;dy++){
                for(int dx=0;dx<ksize;dx++){
                    xx=dx+x;
                    yy=dy+y;    
                    col[idx][col]=img_pad[yy][xx];

                }
            }
        }
    }
    return cols;
}
MatrixXd conv(const MatrixXd& image,const MatrixXd& kernel,int pad,int stride,int ksize){
    MatrixXd W=img2cols(image,pad,stride,ksize);
    int out_h=(h+2*pad-ksize)/stride+1;
    int out_w=(w+2*pad-ksize)/stride+1;
    int N=out_w*out_h;
    MatrixXd A=kernel.reshpe(1,ksize*ksize);
    MatrixXd res=A*W;
    MatrixXd ans=Map<MatrixXd>(res.data(),out_h,out_w);
    return ans;
    
}