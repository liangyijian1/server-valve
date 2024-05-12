//
// Created by leihao on 2024/5/10.
//

#ifndef SERVER__UTILS_H
#define SERVER__UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/core/utils/logger.hpp>


using namespace cv;
using namespace std;

#define VISION_TOLERANCE 0.0000001
#define MATCH_CANDIDATE_NUM 5
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)


struct s_TemplData
{
    vector<Mat> vecPyramid;
    vector<Scalar> vecTemplMean;
    vector<double> vecTemplNorm;
    vector<double> vecInvArea;
    vector<bool> vecResultEqual1;
    bool bIsPatternLearned;
    int iBorderColor;

    void clear()
    {
        vector<Mat>().swap(vecPyramid);
        vector<double>().swap(vecTemplNorm);
        vector<double>().swap(vecInvArea);
        vector<Scalar>().swap(vecTemplMean);
        vector<bool>().swap(vecResultEqual1);
    }
    void resize(int iSize)
    {
        vecTemplMean.resize(iSize);
        vecTemplNorm.resize(iSize, 0);
        vecInvArea.resize(iSize, 1);
        vecResultEqual1.resize(iSize, false);
    }
    s_TemplData()
    {
        bIsPatternLearned = false;

        iBorderColor = 0;
    }
};

struct s_MatchParameter
{
    Point2d pt;
    double dMatchScore;
    double dMatchAngle;
    //Mat matRotatedSrc;
    Rect rectRoi;
    double dAngleStart;
    double dAngleEnd;
    RotatedRect rectR;
    Rect rectBounding;
    bool bDelete;

    double vecResult[3][3];//for subpixel
    int iMaxScoreIndex;//for subpixel
    bool bPosOnBorder;
    Point2d ptSubPixel;
    double dNewAngle;

    s_MatchParameter(Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
    {
        pt = ptMinMax;
        dMatchScore = dScore;
        dMatchAngle = dAngle;

        bDelete = false;
        dNewAngle = 0.0;

        bPosOnBorder = false;
    }
    s_MatchParameter()
    {
        double dMatchScore = 0;
        double dMatchAngle = 0;
    }
    ~s_MatchParameter()
    {

    }
};

struct s_SingleTargetMatch
{
    Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
    double dMatchedAngle;
    double dMatchScore;
};

struct s_BlockMax
{
    struct Block
    {
        Rect rect;
        double dMax;
        Point ptMaxLoc;
        Block()
        {}
        Block(Rect rect_, double dMax_, Point ptMaxLoc_)
        {
            rect = rect_;
            dMax = dMax_;
            ptMaxLoc = ptMaxLoc_;
        }
    };
    s_BlockMax()
    {}
    vector<Block> vecBlock;
    Mat matSrc;
    s_BlockMax(Mat matSrc_, Size sizeTemplate)
    {
        matSrc = matSrc_;
        int iBlockW = sizeTemplate.width * 2;
        int iBlockH = sizeTemplate.height * 2;

        int iCol = matSrc.cols / iBlockW;
        bool bHResidue = matSrc.cols % iBlockW != 0;

        int iRow = matSrc.rows / iBlockH;
        bool bVResidue = matSrc.rows % iBlockH != 0;

        if (iCol == 0 || iRow == 0)
        {
            vecBlock.clear();
            return;
        }

        vecBlock.resize(iCol * iRow);
        int iCount = 0;
        for (int y = 0; y < iRow; y++)
        {
            for (int x = 0; x < iCol; x++)
            {
                Rect rectBlock(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
                vecBlock[iCount].rect = rectBlock;
                minMaxLoc(matSrc(rectBlock), 0, &vecBlock[iCount].dMax, 0, &vecBlock[iCount].ptMaxLoc);
                vecBlock[iCount].ptMaxLoc += rectBlock.tl();
                iCount++;
            }
        }
        if (bHResidue && bVResidue)
        {
            Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
            Block blockRight;
            blockRight.rect = rectRight;
            minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
            blockRight.ptMaxLoc += rectRight.tl();
            vecBlock.push_back(blockRight);

            Rect rectBottom(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
            Block blockBottom;
            blockBottom.rect = rectBottom;
            minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
            blockBottom.ptMaxLoc += rectBottom.tl();
            vecBlock.push_back(blockBottom);
        }
        else if (bHResidue)
        {
            Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
            Block blockRight;
            blockRight.rect = rectRight;
            minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
            blockRight.ptMaxLoc += rectRight.tl();
            vecBlock.push_back(blockRight);
        }
        else
        {
            Rect rectBottom(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
            Block blockBottom;
            blockBottom.rect = rectBottom;
            minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
            blockBottom.ptMaxLoc += rectBottom.tl();
            vecBlock.push_back(blockBottom);
        }
    }
    void UpdateMax(Rect rectIgnore)
    {
        if (vecBlock.size() == 0)
            return;
        //找出所有跟rectIgnore交集的block
        int iSize = vecBlock.size();
        for (int i = 0; i < iSize; i++)
        {
            Rect rectIntersec = rectIgnore & vecBlock[i].rect;
            //無交集
            if (rectIntersec.width == 0 && rectIntersec.height == 0)
                continue;
            minMaxLoc(matSrc(vecBlock[i].rect), 0, &vecBlock[i].dMax, 0, &vecBlock[i].ptMaxLoc);
            vecBlock[i].ptMaxLoc += vecBlock[i].rect.tl();
        }
    }
    void GetMaxValueLoc(double& dMax, Point& ptMaxLoc)
    {
        int iSize = vecBlock.size();
        if (iSize == 0)
        {
            minMaxLoc(matSrc, 0, &dMax, 0, &ptMaxLoc);
            return;
        }
        int iIndex = 0;
        dMax = vecBlock[0].dMax;
        for (int i = 1; i < iSize; i++)
        {
            if (vecBlock[i].dMax >= dMax)
            {
                iIndex = i;
                dMax = vecBlock[i].dMax;
            }
        }
        ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
    }
};

class _utils {
public:
    _utils()
            : m_iMaxPos(1),
              m_dMaxOverlap(0),
              m_dScore(0.5),
              m_iMinReduceArea(256),
              m_dToleranceAngle(10.0),
              m_bStopLayer1(false),
              th(0.85)
    {}
    int m_iMaxPos;
    double m_dMaxOverlap;
    float th;
    double m_dScore;
    int m_iMinReduceArea;
    double m_dToleranceAngle;
    string m_strExecureTime;
    vector<s_SingleTargetMatch> m_vecSingleTargetData;
    bool m_bStopLayer1;

public:
    cv::Mat m_matSrc;
    cv::Mat m_matDst;
    s_TemplData m_TemplData;
    bool Match();
    int GetTopLayer(Mat* matTempl, int iMinDstLength);
    void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD);
    int IM_Conv_SIMD (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength);
    int _mm_hsum_epi32 (__m128i V);
    void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer);
    void FilterWithScore(vector<s_MatchParameter>* vec, double dScore);
    void FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap);
    Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle);
    Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle);
    Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap);
    Point GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax);
    void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI);
    void SortPtWithCenter(vector<Point2f>& vecSort);
    void LearnPattern();
};


#endif //SERVER__UTILS_H
