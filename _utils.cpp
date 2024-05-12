//
// Created by leihao on 2024/5/10.
//

#include "_utils.h"

bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs) { return  lhs.dMatchScore > rhs.dMatchScore; }
bool comparePtWithAngle(const pair<Point2f, double> lhs, const pair<Point2f, double> rhs) { return lhs.second < rhs.second; }

bool _utils::Match()
{
    if (m_matSrc.empty() || m_matDst.empty())
        return false;
    if ((m_matDst.cols < m_matSrc.cols && m_matDst.rows > m_matSrc.rows) || (m_matDst.cols > m_matSrc.cols && m_matDst.rows < m_matSrc.rows))
        return false;
    if (m_matDst.size().area() > m_matSrc.size().area())
        return false;
    LearnPattern();
    int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
    vector<Mat> vecMatSrcPyr;
    buildPyramid(m_matSrc, vecMatSrcPyr, iTopLayer);

    s_TemplData* pTemplData = &m_TemplData;
    double dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iTopLayer].cols, pTemplData->vecPyramid[iTopLayer].rows)) * R2D;

    vector<double> vecAngles;
    if (m_dToleranceAngle < VISION_TOLERANCE)
        vecAngles.push_back(0.0);
    else
    {
        for (double dAngle = 0; dAngle < m_dToleranceAngle + dAngleStep; dAngle += dAngleStep)
            vecAngles.push_back(dAngle);
        for (double dAngle = -dAngleStep; dAngle > -m_dToleranceAngle - dAngleStep; dAngle -= dAngleStep)
            vecAngles.push_back(dAngle);
    }

    int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
    Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

    int iSize = (int)vecAngles.size();
    //vector<s_MatchParameter> vecMatchParameter (iSize * (m_iMaxPos + MATCH_CANDIDATE_NUM));
    vector<s_MatchParameter> vecMatchParameter;
    //Caculate lowest score at every layer
    vector<double> vecLayerScore(iTopLayer + 1, m_dScore);
    for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
        vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

    Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
    for (int i = 0; i < iSize; i++)
    {
        Mat matRotatedSrc, matR = getRotationMatrix2D(ptCenter, vecAngles[i], 1);
        Mat matResult;
        Point ptMaxLoc;
        double dValue, dMaxVal;
        double dRotate = clock();
        Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), pTemplData->vecPyramid[iTopLayer].size(), vecAngles[i]);

        float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
        float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
        matR.at<double>(0, 2) += fTranslationX;
        matR.at<double>(1, 2) += fTranslationY;
        warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar(pTemplData->iBorderColor));

        MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, false);


        minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
        if (dMaxVal < vecLayerScore[iTopLayer])
            continue;
        vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
        for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
        {
            ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, m_dMaxOverlap);
            if (dValue < vecLayerScore[iTopLayer])
                break;
            vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
        }

    }
    sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);


    int iMatchSize = (int)vecMatchParameter.size();
    int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;

    int iStopLayer = m_bStopLayer1 ? 1 : 0; //设置为1时：粗匹配，牺牲精度提升速度。
    vector<s_MatchParameter> vecAllResult;
    for (int i = 0; i < (int)vecMatchParameter.size(); i++)
    {
        double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
        Point2f ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

        double dAngleStep = atan(2.0 / max(iDstW, iDstH)) * R2D;//min改為max
        vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
        vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

        if (iTopLayer <= iStopLayer)
        {
            vecMatchParameter[i].pt = Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
            vecAllResult.push_back(vecMatchParameter[i]);
        }
        else
        {
            for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
            {
                dAngleStep = atan(2.0 / max(pTemplData->vecPyramid[iLayer].cols, pTemplData->vecPyramid[iLayer].rows)) * R2D;//min改為max
                vector<double> vecAngles;
                double dMatchedAngle = vecMatchParameter[i].dMatchAngle;
                if (m_dToleranceAngle < VISION_TOLERANCE)
                    vecAngles.push_back(0.0);
                else
                    for (int i = -1; i <= 1; i++)
                        vecAngles.push_back(dMatchedAngle + dAngleStep * i);

                Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
                iSize = (int)vecAngles.size();
                vector<s_MatchParameter> vecNewMatchParameter(iSize);
                int iMaxScoreIndex = 0;
                double dBigValue = -1;
                for (int j = 0; j < iSize; j++)
                {
                    Mat matResult, matRotatedSrc;
                    double dMaxValue = 0;
                    Point ptMaxLoc;
                    GetRotatedROI(vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size(), ptLT * 2, vecAngles[j], matRotatedSrc);
                    MatchTemplate(matRotatedSrc, pTemplData, matResult, iLayer, true);
                    minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
                    vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecAngles[j]);

                    if (vecNewMatchParameter[j].dMatchScore > dBigValue)
                    {
                        iMaxScoreIndex = j;
                        dBigValue = vecNewMatchParameter[j].dMatchScore;
                    }
                    if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
                        vecNewMatchParameter[j].bPosOnBorder = true;
                    if (!vecNewMatchParameter[j].bPosOnBorder)
                    {
                        for (int y = -1; y <= 1; y++)
                            for (int x = -1; x <= 1; x++)
                                vecNewMatchParameter[j].vecResult[x + 1][y + 1] = matResult.at<float>(ptMaxLoc + Point(x, y));
                    }
                }
                if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
                    break;

                double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

                Point2f ptPaddingLT = ptRotatePt2f(ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f(3, 3);
                Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);

                pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);
                if (iLayer == iStopLayer)
                {
                    vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
                    vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
                }
                else
                {
                    vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
                    vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
                    vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
                    ptLT = pt;
                }
            }

        }
    }
    FilterWithScore(&vecAllResult, m_dScore);

    iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
    iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

    for (int i = 0; i < (int)vecAllResult.size(); i++)
    {
        Point2f ptLT, ptRT, ptRB, ptLB;
        double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
        ptLT = vecAllResult[i].pt;
        ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
        ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
        ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
        //紀錄旋轉矩形
        vecAllResult[i].rectR = RotatedRect(ptLT, ptRT, ptRB);
    }
    FilterWithRotatedRect(&vecAllResult, CV_TM_CCOEFF_NORMED, m_dMaxOverlap);
    sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

    iMatchSize = (int)vecAllResult.size();
    if (vecAllResult.size() == 0)
        return false;
    int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

    for (int i = 0; i < iMatchSize; i++)
    {
        s_SingleTargetMatch sstm;
        double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

        sstm.ptLT = vecAllResult[i].pt;

        sstm.ptRT = Point2d(sstm.ptLT.x + iW * cos(dRAngle), sstm.ptLT.y - iW * sin(dRAngle));
        sstm.ptLB = Point2d(sstm.ptLT.x + iH * sin(dRAngle), sstm.ptLT.y + iH * cos(dRAngle));
        sstm.ptRB = Point2d(sstm.ptRT.x + iH * sin(dRAngle), sstm.ptRT.y + iH * cos(dRAngle));
        sstm.ptCenter = Point2d((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
        sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
        sstm.dMatchScore = vecAllResult[i].dMatchScore;

        if (sstm.dMatchedAngle < -180)
            sstm.dMatchedAngle += 360;
        if (sstm.dMatchedAngle > 180)
            sstm.dMatchedAngle -= 360;
        m_vecSingleTargetData.push_back(sstm);

        if (i + 1 == m_iMaxPos)
            break;
    }

    for (int i = 0; i < (int)m_vecSingleTargetData.size(); i++)
    {
        s_SingleTargetMatch sstm = m_vecSingleTargetData[i];
        cout << "Matched Score : " << sstm.dMatchScore << '\n';
        if (sstm.dMatchScore > 0.84)
        {
            cout << "OK" << '\n';
        }
        else {
            cout << "NG" << '\n';
        }
    }
    return (int)m_vecSingleTargetData.size();
}

void _utils::MatchTemplate(Mat& matSrc, s_TemplData* pTemplData, Mat& matResult, int iLayer, bool bUseSIMD)
{

    matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);
    CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
}

void _utils::CCOEFF_Denominator(Mat& matSrc, s_TemplData* pTemplData, Mat& matResult, int iLayer)
{
    if (pTemplData->vecResultEqual1[iLayer])
    {
        matResult = Scalar::all(1);
        return;
    }
    double* q0 = 0, * q1 = 0, * q2 = 0, * q3 = 0;
    Mat sum, sqsum;
    integral(matSrc, sum, sqsum, CV_64F);
    q0 = (double*)sqsum.data;
    q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
    q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
    q3 = q2 + pTemplData->vecPyramid[iLayer].cols;
    double* p0 = (double*)sum.data;
    double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
    double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows * sum.step);
    double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;
    int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
    int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;
    double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
    double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
    double dInvArea = pTemplData->vecInvArea[iLayer];

    int i, j;
    for (i = 0; i < matResult.rows; i++)
    {
        float* rrow = matResult.ptr<float>(i);
        int idx = i * sumstep;
        int idx2 = i * sqstep;

        for (j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1)
        {
            double num = rrow[j], t;
            double wndMean2 = 0, wndSum2 = 0;

            t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
            wndMean2 += t * t;
            num -= t * dTemplMean0;
            wndMean2 *= dInvArea;
            t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
            wndSum2 += t;
            //t = std::sqrt (MAX (wndSum2 - wndMean2, 0)) * dTemplNorm;
            double diff2 = MAX(wndSum2 - wndMean2, 0);
            if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wndSum2))
                t = 0; // avoid rounding errors
            else
                t = std::sqrt(diff2) * dTemplNorm;
            if (fabs(num) < t)
                num /= t;
            else if (fabs(num) < t * 1.125)
                num = num > 0 ? 1 : -1;
            else
                num = 0;

            rrow[j] = (float)num;
        }
    }
}

void _utils::FilterWithScore(vector<s_MatchParameter>* vec, double dScore) {
    sort(vec->begin(), vec->end(), compareScoreBig2Small);
    int iSize = vec->size(), iIndexDelete = iSize + 1;
    for (int i = 0; i < iSize; i++)
    {
        if ((*vec)[i].dMatchScore < dScore)
        {
            iIndexDelete = i;
            break;
        }
    }
    if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
        return;
    vec->erase(vec->begin() + iIndexDelete, vec->end());
    return;
}

void _utils::FilterWithRotatedRect(vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap) {
    int iMatchSize = (int)vec->size();
    RotatedRect rect1, rect2;
    for (int i = 0; i < iMatchSize - 1; i++)
    {
        if (vec->at(i).bDelete)
            continue;
        for (int j = i + 1; j < iMatchSize; j++)
        {
            if (vec->at(j).bDelete)
                continue;
            rect1 = vec->at(i).rectR;
            rect2 = vec->at(j).rectR;
            vector<Point2f> vecInterSec;
            int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);
            if (iInterSecType == INTERSECT_NONE)//無交集
                continue;
            else if (iInterSecType == INTERSECT_FULL) //一個矩形包覆另一個
            {
                int iDeleteIndex;
                if (iMethod == CV_TM_SQDIFF)
                    iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
                else
                    iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
                vec->at(iDeleteIndex).bDelete = true;
            }
            else//交點 > 0
            {
                if (vecInterSec.size() < 3)//一個或兩個交點
                    continue;
                else
                {
                    int iDeleteIndex;
                    //求面積與交疊比例
                    SortPtWithCenter(vecInterSec);
                    double dArea = contourArea(vecInterSec);
                    double dRatio = dArea / rect1.size.area();
                    //若大於最大交疊比例，選分數高的
                    if (dRatio > dMaxOverLap)
                    {
                        if (iMethod == CV_TM_SQDIFF)
                            iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
                        else
                            iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
                        vec->at(iDeleteIndex).bDelete = true;
                    }
                }
            }
        }
    }
}

int _utils::GetTopLayer(Mat* matTempl, int iMinDstLength) {
    int iTopLayer = 0;
    int iMinReduceArea = iMinDstLength * iMinDstLength;
    int iArea = matTempl->cols * matTempl->rows;
    while (iArea > iMinReduceArea)
    {
        iArea /= 4;
        iTopLayer++;
    }
    return iTopLayer;
}

Size _utils::GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle) {
    double dRAngle_radian = dRAngle * D2R;
    Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
    Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
    Point2f ptLT_R = ptRotatePt2f(Point2f(ptLT), ptCenter, dRAngle_radian);
    Point2f ptLB_R = ptRotatePt2f(Point2f(ptLB), ptCenter, dRAngle_radian);
    Point2f ptRB_R = ptRotatePt2f(Point2f(ptRB), ptCenter, dRAngle_radian);
    Point2f ptRT_R = ptRotatePt2f(Point2f(ptRT), ptCenter, dRAngle_radian);

    float fTopY = max(max(ptLT_R.y, ptLB_R.y), max(ptRB_R.y, ptRT_R.y));
    float fBottomY = min(min(ptLT_R.y, ptLB_R.y), min(ptRB_R.y, ptRT_R.y));
    float fRightX = max(max(ptLT_R.x, ptLB_R.x), max(ptRB_R.x, ptRT_R.x));
    float fLeftX = min(min(ptLT_R.x, ptLB_R.x), min(ptRB_R.x, ptRT_R.x));

    if (dRAngle > 360)
        dRAngle -= 360;
    else if (dRAngle < 0)
        dRAngle += 360;

    if (fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
    {
        return Size(sizeSrc.height, sizeSrc.width);
    }
    else if (fabs(dRAngle) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
    {
        return sizeSrc;
    }

    double dAngle = dRAngle;

    if (dAngle > 0 && dAngle < 90)
    {
        ;
    }
    else if (dAngle > 90 && dAngle < 180)
    {
        dAngle -= 90;
    }
    else if (dAngle > 180 && dAngle < 270)
    {
        dAngle -= 180;
    }
    else if (dAngle > 270 && dAngle < 360)
    {
        dAngle -= 270;
    }
    else//Debug
    {
        cout << "Unkown" << '\n';
    }

    float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
    float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

    int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
    int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

    Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

    bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
                      || (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
                          || sizeDst.area() > sizeRet.area());
    if (bWrongSize)
        sizeRet = Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

    return sizeRet;
}

Point2f _utils::ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle) {
    double dWidth = ptOrg.x * 2;
    double dHeight = ptOrg.y * 2;
    double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

    double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
    double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

    dY = -dY + dHeight;
    return Point2f((float)dX, (float)dY);
}

Point _utils::GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap) {
    int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
    int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);

    rectangle(matResult, Rect(iStartX, iStartY, 2 * sizeTemplate.width * (1 - dMaxOverlap), 2 * sizeTemplate.height * (1 - dMaxOverlap)), Scalar(-1), FILLED);

    Point ptNewMaxLoc;
    minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
    return ptNewMaxLoc;
}

Point _utils::GetNextMaxLoc(Mat& matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap,
                            s_BlockMax& blockMax) {
    int iStartX = int(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
    int iStartY = int(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
    Rect rectIgnore(iStartX, iStartY, int(2 * sizeTemplate.width * (1 - dMaxOverlap))
            , int(2 * sizeTemplate.height * (1 - dMaxOverlap)));

    rectangle(matResult, rectIgnore, Scalar(-1), FILLED);
    blockMax.UpdateMax(rectIgnore);
    Point ptReturn;
    blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
    return ptReturn;
}

void _utils::GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI) {
    double dAngle_radian = dAngle * D2R;
    Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
    Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
    Size sizePadding(size.width + 6, size.height + 6);
    Mat rMat = getRotationMatrix2D(ptC, dAngle, 1);
    rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
    rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
    warpAffine(matSrc, matROI, rMat, sizePadding);

}

void _utils::SortPtWithCenter(vector<Point2f>& vecSort) {
    int iSize = (int)vecSort.size();
    Point2f ptCenter;
    for (int i = 0; i < iSize; i++)
        ptCenter += vecSort[i];
    ptCenter /= iSize;
    Point2f vecX(1, 0);
    vector<pair<Point2f, double>> vecPtAngle(iSize);
    for (int i = 0; i < iSize; i++)
    {
        vecPtAngle[i].first = vecSort[i];//pt
        Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
        float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
        float fDot = vec1.x;

        if (vec1.y < 0)//若點在中心的上方
        {
            vecPtAngle[i].second = acos(fDot / fNormVec1) * R2D;
        }
        else if (vec1.y > 0)//下方
        {
            vecPtAngle[i].second = 360 - acos(fDot / fNormVec1) * R2D;
        }
        else//點與中心在相同Y
        {
            if (vec1.x - ptCenter.x > 0)
                vecPtAngle[i].second = 0;
            else
                vecPtAngle[i].second = 180;
        }

    }
    sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
    for (int i = 0; i < iSize; i++)
        vecSort[i] = vecPtAngle[i].first;
}

void _utils::LearnPattern() {

    m_TemplData.clear();
    m_iMinReduceArea = 256;
    int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
    buildPyramid(m_matDst, m_TemplData.vecPyramid, iTopLayer);
    s_TemplData* templData = &m_TemplData;
    templData->iBorderColor = mean(m_matDst).val[0] < 128 ? 255 : 0;
    int iSize = templData->vecPyramid.size();
    templData->resize(iSize);
    for (int i = 0; i < iSize; i++)
    {
        double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
        Scalar templMean, templSdv;
        double templNorm = 0, templSum2 = 0;
        meanStdDev(templData->vecPyramid[i], templMean, templSdv);
        templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

        if (templNorm < DBL_EPSILON)
        {
            templData->vecResultEqual1[i] = true;
        }
        templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];
        templSum2 /= invArea;
        templNorm = std::sqrt(templNorm);
        templNorm /= std::sqrt(invArea); // care of accuracy here
        templData->vecInvArea[i] = invArea;
        templData->vecTemplMean[i] = templMean;
        templData->vecTemplNorm[i] = templNorm;
    }
    templData->bIsPatternLearned = true;

}