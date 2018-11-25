#include "HMSpringImgProcess.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\highgui.hpp>
#include <set>
#include <array>
#include "HMImage.h"
#include "HMLutTable.h"
#include "log.h"
#include <assert.h>
using namespace cv;
using namespace std;

HMSpringImgProcess &HMSpringImgProcess::inst()
{
    static HMSpringImgProcess s_this;
    return s_this;
}

HMSpringImgProcess::HMSpringImgProcess()
{

}

bool HMSpringImgProcess::setAlgoParameter(const algoParameter_t &param)
{
    if (param.nCellWidth < 0 || param.nCellWidth > 800 ||
        param.nCellHeight < 0 || param.nCellHeight > 300 ||
        param.nContourCountThreshold < 0 || param.nContourCountThreshold > 100 ||
        param.nInterval < 0 || param.nInterval > 4 ||
        param.nHysteresisHigh < 0 || param.nHysteresisHigh > 200 ||
        param.nHysteresisLow < 0 || param.nHysteresisLow > 200 ||
        param.nHysteresisLow < param.nHysteresisHigh ||
        param.nCheckWidth < 0 || param.nCheckWidth > 600 ||
        param.nCheckHeight < 0 || param.nCheckHeight > 200)
    {
        LogInfo() << "algorithm parameter is out of range";
        return false;
    }

    m_algoParam = param;

    return true;
}

bool HMSpringImgProcess::analyzeSpringImg(pImage_t img, vSprings_t &vRet)
{
    //1. sanity check
    if (!img || img->getDeviceImg().elemSize1() != 2) return false;


    m_pImg = img;
    cv::Mat  mImg = img->getDeviceImg().clone();



    //2. Gaussian to filer noise
    double sigma = 0;
    GaussianBlur(mImg, mImg, cv::Size(3, 3), sigma);


    //here is the parameter
    int nCellHeight = m_algoParam.nCellHeight;
    int nCellWidth = m_algoParam.nCellWidth;

    const int nCheckWidth       = m_algoParam.nCheckWidth;
    const int nCheckHeight      = m_algoParam.nCheckHeight;
    const int nInterval         = m_algoParam.nInterval;;
    const int nHysteresisLow    = m_algoParam.nHysteresisLow;;
    const int nHysteresisHigh   = m_algoParam.nHysteresisHigh;;
    const int nContourCountThreshold = m_algoParam.nContourCountThreshold;




    struct lineStatistic_t
    {
        lineStatistic_t()
        {
            nSum.fill(0);
        }
        enum eType
        {
            eWhite,
            eSpring,
            ePlate,
            eBlack,
            eOther,
            eSpringBar,
            eMax,
        };

        void clear()
        {
            nSum.fill(0);
            for (int i = eWhite; i < eMax; ++i)
            {
                vSegLen[i].clear();
                vSegCoor[i].clear();
            }
        }

        vector<int> vSegLen[eMax];
        vector<pair<int, int>> vSegCoor[eMax];

        array<int, eMax> nSum;

        int getMaxLen(eType e)
        {
            if (vSegLen[e].empty()) return -1;
            else return *std::max_element(vSegLen[e].begin(), vSegLen[e].end());
        }
    };

    grayRange_t plateBG     = m_algoParam.plateBG;
    grayRange_t whiteBG     = m_algoParam.whiteBG;
    grayRange_t SpringBG    = m_algoParam.SpringBG;
    grayRange_t SpringBarBG = m_algoParam.SpringBarBG;
    grayRange_t blackBG     = m_algoParam.blackBG;

    auto fnSaveImg = [](const string &imgName, cv::Mat &img)
    {
        const string sImgFolder = "e:\\spring_img";
        string sPath = fmt::format("{}\\{}.png", sImgFolder, imgName);
        cv::imwrite(sPath, img);
    };





    auto fnInrange = [](int nVal, const grayRange_t &range) ->bool
    {
        return (nVal >= range.nLow && nVal <= range.nHigh);
    };

    auto fnLineStatistic = [&](int nPos, cv::Mat &subimg, bool bY, lineStatistic_t &status, int nStart = -1, int nEnd = -1)
    {
        cv::Mat mRoi;
        int		nThisLen[lineStatistic_t::eMax] = { 0 };
        int		nStartRange = 0;
        int		nVals[4000] = { -1 };
        ushort	nVal = 0;

        status.clear();


        auto fnProcessPix = [&](ushort nVal, int nIdx)
        {
            lineStatistic_t::eType eThis;
            if (fnInrange(nVal, whiteBG))
            {
                eThis = lineStatistic_t::eWhite;
            }
            else if (fnInrange(nVal, plateBG))
            {
                eThis = lineStatistic_t::ePlate;
            }
            else if (fnInrange(nVal, SpringBG))
            {
                eThis = lineStatistic_t::eSpring;
            }
            else if (fnInrange(nVal, blackBG))
            {
                eThis = lineStatistic_t::eBlack;
            }
            else if (fnInrange(nVal, SpringBarBG))
            {
                eThis = lineStatistic_t::eSpringBar;
            }
            else
            {
                eThis = lineStatistic_t::eOther;
            }

            int eArr[] = { lineStatistic_t::eWhite, lineStatistic_t::eSpring, lineStatistic_t::ePlate, lineStatistic_t::eBlack, lineStatistic_t::eOther, lineStatistic_t::eSpringBar };
            eArr[eThis] = -1;

            nThisLen[eThis]++;
            status.nSum[eThis]++;
            for (auto e : eArr)
            {
                if (e == -1) continue;
                if (nThisLen[e] != 0)
                {
                    status.vSegLen[e].push_back(nThisLen[e]);
                    status.vSegCoor[e].push_back(make_pair(nStartRange, nIdx - 1));
                    nStartRange = nIdx;
                }
                nThisLen[e] = 0;
            }
        };
        if (bY)
        {
            mRoi = cv::Mat(subimg, cv::Rect(0, nPos, subimg.cols, 1));
            if (nStart == -1)  nStart = 0;
            if (nEnd == -1) nEnd = subimg.cols - 1;
            nStartRange = nStart;

            for (int i = nStart; i <= nEnd; ++i)
            {
                if (subimg.elemSize1() == 2)
                {
                    nVal = mRoi.at<ushort>(0, i);
                }
                else
                {
                    nVal = mRoi.at<uchar>(0, i);
                }
                fnProcessPix(nVal, i);
                nVals[i] = nVal;
            }

        }
        else
        {
            if (nStart == -1)  nStart = 0;
            if (nEnd == -1) nEnd = subimg.rows - 1;
            nStartRange = nStart;

            mRoi = cv::Mat(subimg, cv::Rect(nPos, 0, 1, subimg.rows));
            for (int i = nStart; i < nEnd; ++i)
            {
                if (subimg.elemSize1() == 2)
                {
                    nVal = mRoi.at<ushort>(i, 0);
                }
                else
                {
                    nVal = mRoi.at<uchar>(i, 0);

                }
                fnProcessPix(nVal, i);
                nVals[i] = nVal;

            }
        }
        int eArr[] = { lineStatistic_t::eWhite, lineStatistic_t::eSpring, lineStatistic_t::ePlate, lineStatistic_t::eBlack, lineStatistic_t::eOther, lineStatistic_t::eSpringBar };
        for (auto e : eArr)
        {
            if (nThisLen[e] != 0)
            {
                status.vSegLen[e].push_back(nThisLen[e]);
                status.vSegCoor[e].push_back(make_pair(nStartRange, nEnd));

            }
        }

    };

    auto fnMove2FirstNotPlate = [&](int &nPos, int nInter, cv::Mat &subimg, bool bY)
    {
        bool bFindPlateLine = false;
        int nPixs = bY ? subimg.cols : subimg.rows;

        while (1)
        {
            if (bY)
            {
                if (nInter > 0 && nPos >= subimg.rows) return;
                if (nInter < 0 && nPos < 0) return;
            }
            else
            {
                if (nInter > 0 && nPos >= subimg.cols) return;
                if (nInter < 0 && nPos < 0) return;
            }
            lineStatistic_t status;
            fnLineStatistic(nPos, subimg, bY, status);

            if (!bFindPlateLine && status.getMaxLen(lineStatistic_t::ePlate) > nPixs * 0.75)
            {
                bFindPlateLine = true;
            }
            if (bFindPlateLine && status.getMaxLen(lineStatistic_t::ePlate) < nPixs * 0.5)
            {
                nPos -= nInter;
                break;
            }

            nPos += nInter;
        }
    };
    auto fnMove2FirstPlate = [&](int &nPos, int nInter, cv::Mat &subimg, bool bY)
    {
        int nPixs = bY ? subimg.cols : subimg.rows;

        while (1)
        {
            if (bY)
            {
                if (nInter > 0 && nPos >= subimg.rows) return;
                if (nInter < 0 && nPos < 0) return;
            }
            else
            {
                if (nInter > 0 && nPos >= subimg.cols) return;
                if (nInter < 0 && nPos < 0) return;
            }

            lineStatistic_t status;
            fnLineStatistic(nPos, subimg, bY, status);

            if (status.getMaxLen(lineStatistic_t::ePlate) > nPixs * 0.75)
            {
                break;
            }
            nPos += nInter;
        }
    };


    //find center core image

    {
        int nStartX = 0;
        int nEndX = mImg.cols - 1;
        int nStartY = 0;
        int nEndY = mImg.rows - 1;

        fnMove2FirstNotPlate(nStartX, nInterval, mImg, false);
        fnMove2FirstNotPlate(nEndX, -nInterval, mImg, false);
        fnMove2FirstNotPlate(nStartY, nInterval, mImg, true);
        fnMove2FirstNotPlate(nEndY, -nInterval, mImg, true);

        LogInfo() << fmt::format("center core image coor: x1 = {}, y1 = {}, x2 = {}, y2 = {}", nStartX, nStartY, nEndX, nEndY);

        if ((nEndX - nStartX) < 500 || (nEndY - nStartY) < 500)
        {
            LogError() << fmt::format("calculated center core image is too short, width = {}, height = {}", nEndX - nStartX, nEndY - nStartY);
            return false;
        }
        mImg = cv::Mat(mImg, cv::Rect(nStartX, nStartY, nEndX - nStartX, nEndY - nStartY)).clone();
    }
    vector<int> vRows;
    vector<int> vCols;


    //find all rows
    {
        vRows.push_back(0);
        int nRowStart = 0;
        while (nRowStart < mImg.rows)
        {
            nRowStart += nCellHeight / 2;
            fnMove2FirstPlate(nRowStart, nInterval, mImg, true);

            if (nRowStart < mImg.rows)  vRows.push_back(nRowStart);
            else
            {
                vRows.push_back(mImg.rows - 1);
                break;
            }
            fnMove2FirstNotPlate(nRowStart, nInterval, mImg, true);
            if (nRowStart < mImg.rows)  vRows.push_back(nRowStart);
        }

        if (vRows.size() < 4)
        {
            LogError() << fmt::format("calculated calculate rows's size is too little: {}", vRows.size());
            return false;
        }
    }
    //find all cols
    {
        vCols.push_back(0);
        int nColStart = 0;
        while (nColStart < mImg.cols)
        {
            nColStart += nCellWidth / 2;
            fnMove2FirstPlate(nColStart, nInterval, mImg, false);

            if (nColStart < mImg.cols)  vCols.push_back(nColStart);
            else
            {
                vCols.push_back(mImg.cols - 1);
                break;
            }
            fnMove2FirstNotPlate(nColStart, nInterval, mImg, false);
            if (nColStart < mImg.cols)  vCols.push_back(nColStart);
        }

        if (vCols.size() < 4)
        {
            LogError() << fmt::format("calculated calculate cols's size is too little: {}", vRows.size());
            return false;
        }
    }
    {
        cv::Mat mMatLine = mImg.clone();
        for (auto nPos : vCols)
        {
            cv::line(mMatLine, cv::Point(nPos, 0), cv::Point(nPos + 1, mImg.rows - 1), cv::Scalar(65535), 3);
        }

        for (auto nPos : vRows)
        {
            cv::line(mMatLine, cv::Point(0, nPos), cv::Point(mImg.cols - 1, nPos + 1), cv::Scalar(65535), 3);
        }

        fnSaveImg("cut", mMatLine);
    }


    // loop over each cell
    for (size_t nIdxY = 0; nIdxY < vRows.size(); nIdxY += 2)
    {
        for (size_t nIdxX = 0; nIdxX < vCols.size(); nIdxX += 2)
        {
            singleSpring item;

            int nCellRealWidth = vCols[nIdxX + 1] - vCols[nIdxX];
            int nCellRealHeight = vRows[nIdxY + 1] - vRows[nIdxY];
            int nXStart = vCols[nIdxX];
            int nYStart = vRows[nIdxY];
            //			int nXEnd = nXStart + nCellRealWidth;
            //			int nYEnd = nYStart + nCellRealHeight;
            int nCalArea = 5;        //calculate 1/5  from center
            cv::Mat imgCal = cv::Mat(mImg, cv::Rect(nXStart + nCellRealWidth / 2 - nCellRealWidth / nCalArea / 2,
                nYStart + nCellRealHeight / 2 - nCellRealHeight / nCalArea / 2, nCellRealWidth / nCalArea, nCellRealHeight / nCalArea));


            double dNorm = cv::norm(imgCal, NORM_L1) / imgCal.rows / imgCal.cols;
            int    nMinNorm = dNorm - 8000;
            int    nMaxNorm = dNorm + 5000;
            grayRange_t grayRange = { nMinNorm, nMaxNorm };
            grayRange_t validGrayRange = { dNorm - 2000, dNorm + 6000 };
            if (fnInrange(dNorm, grayRange))
            {
                item.img = cv::Mat(mImg, cv::Rect(nXStart, nYStart, nCellRealWidth, nCellRealHeight)).clone();

                item.nCol = nIdxX / 2;
                item.nRow = nIdxY / 2;
                cv::rectangle(mImg, cv::Rect(nXStart, nYStart, nCellRealWidth, nCellRealHeight), cv::Scalar(65535), 3);
                fnSaveImg(fmt::format("{}_{}_raw", item.nCol, item.nRow), item.img);

                //calculate the slope

                int nTopMark = 0;
                int nBottomMark = item.img.rows - 1;
                int nLeftMark = 0;
                int nRightMark = item.img.cols - 1;


                {
                    auto  fnGetRotateDegree = [](const cv::Point &pt1, const cv::Point &pt2)->double
                    {
                        double  dy = pt1.y - pt2.y;
                        double  dx = pt1.x - pt2.x;
                        float   angle = atan2(dy, dx);

                        double  dDegree = 180 / 3.1415926 * angle;
                        double  dFinal = 0;
                        if (dDegree > 0) dFinal = (dDegree - 180);
                        else dFinal = (180 + dDegree);

                        return dFinal;
                    };

                    //use 3 vertical line to calculate slope, also used to get the vertical center
                    lineStatistic_t statusCenter;
                    lineStatistic_t statusLeft;
                    lineStatistic_t statusRight;

                    //use 2 horizontal line inside spring bar to calculate the horizontal center
                    lineStatistic_t statusUpbar;
                    lineStatistic_t statusBotbar;


                    //get 3 vertical line statistic, center, left 1/6, right 1/6
                    fnLineStatistic(nCellRealWidth / 2, item.img, false, statusCenter);
                    fnLineStatistic(nCellRealWidth / 2 - nCellRealWidth / 6, item.img, false, statusLeft);
                    fnLineStatistic(nCellRealWidth / 2 + nCellRealWidth / 6, item.img, false, statusRight);



                    //assume the bar segments is all bigger that 2
                    assert(statusCenter.vSegCoor[lineStatistic_t::eSpringBar].size() >= 2 &&
                        statusLeft.vSegCoor[lineStatistic_t::eSpringBar].size() >= 2 &&
                        statusRight.vSegCoor[lineStatistic_t::eSpringBar].size() >= 2);

                    vector < pair<int, int>> *LeftSpringBar = &statusLeft.vSegCoor[lineStatistic_t::eSpringBar];
                    vector < pair<int, int>> *CenterSpringBar = &statusCenter.vSegCoor[lineStatistic_t::eSpringBar];
                    vector < pair<int, int>> *RightSpringBar = &statusRight.vSegCoor[lineStatistic_t::eSpringBar];


                    //get the bar center point
                    pair<int, int> firstBar = LeftSpringBar->at(0);
                    cv::Point ptLeft(nCellRealWidth / 2 - nCellRealWidth / 6, firstBar.first + (firstBar.second - firstBar.first) / 2);

                    firstBar = CenterSpringBar->at(0);
                    cv::Point ptCenter(nCellRealWidth / 2, firstBar.first + (firstBar.second - firstBar.first) / 2);

                    firstBar = RightSpringBar->at(0);
                    cv::Point ptRight(nCellRealWidth / 2 + nCellRealWidth / 6, firstBar.first + (firstBar.second - firstBar.first) / 2);


                    //get the slope base on th 3 point
                    double dDegree1 = fnGetRotateDegree(ptLeft, ptCenter);
                    double dDegree2 = fnGetRotateDegree(ptCenter, ptRight);

                    //average the slope
                    double dDegree = (dDegree1 + dDegree2) / 2;

                    //rotate the image base on slope
                    cv::Mat dst;
                    cv::Point2f pc(item.img.cols / 2., item.img.rows / 2.);
                    cv::Mat r = cv::getRotationMatrix2D(pc, dDegree, 1.0);
                    cv::warpAffine(item.img, dst, r, item.img.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(65535));

                    fnSaveImg(fmt::format("{}_{}_raw_rotate", item.nCol, item.nRow), dst);

                    //assign image!!!!!!!!!!
                    item.img = dst;

                    //since image maybe rotated, now get the static again
                    fnLineStatistic(nCellRealWidth / 2, item.img, false, statusCenter);
                    fnLineStatistic(nCellRealWidth / 2 - nCellRealWidth / 6, item.img, false, statusLeft);
                    fnLineStatistic(nCellRealWidth / 2 + nCellRealWidth / 6, item.img, false, statusRight);

                    //now need to get the area of valid spring

                    //get the vertical center
                    int nY_upBorder = (LeftSpringBar->at(0).second + CenterSpringBar->at(0).second + RightSpringBar->at(0).second) / 3;
                    int nY_bomBorder = (LeftSpringBar->at(LeftSpringBar->size() - 1).first + CenterSpringBar->at(CenterSpringBar->size() - 1).first + RightSpringBar->at(RightSpringBar->size() - 1).first) / 3;
                    int nY_Center = nY_upBorder + (nY_bomBorder - nY_upBorder) / 2;


                    //get top and bottom bar center horizontal line ,used to get if certain corner is missing
                    int nUpBarCenterY = CenterSpringBar->at(0).first + (CenterSpringBar->at(0).second - CenterSpringBar->at(0).first) / 2;
                    int nBomBarCenterY = CenterSpringBar->at(CenterSpringBar->size() - 1).first + (CenterSpringBar->at(CenterSpringBar->size() - 1).second - CenterSpringBar->at(CenterSpringBar->size() - 1).first) / 2;
                    fnLineStatistic(nUpBarCenterY, item.img, true, statusUpbar);
                    fnLineStatistic(nBomBarCenterY, item.img, true, statusBotbar);


                    //sort the bar segment, to find the longest one
                    auto itMaxUp = std::max_element(statusUpbar.vSegCoor[lineStatistic_t::eSpringBar].begin(), statusUpbar.vSegCoor[lineStatistic_t::eSpringBar].end(),
                        [](const pair<int, int> &a, const pair<int, int> &b) {
                        return (a.second - a.first) < (b.second - b.first);
                    });
                    auto itMaxBom = std::max_element(statusBotbar.vSegCoor[lineStatistic_t::eSpringBar].begin(), statusBotbar.vSegCoor[lineStatistic_t::eSpringBar].end(),
                        [](const pair<int, int> &a, const pair<int, int> &b) {
                        return (a.second - a.first) < (b.second - b.first);
                    });

                    //get the horizontal center
                    int nX_UPCenter = (*itMaxUp).first + ((*itMaxUp).second - (*itMaxUp).first) / 2;
                    int nX_BomCenter = (*itMaxBom).first + ((*itMaxBom).second - (*itMaxBom).first) / 2;
                    int nX_Center = (nX_UPCenter + nX_BomCenter) / 2;

                    //base on vertical and horizontal center, get the crop image used to analyze

                    nTopMark = nY_Center - nCheckHeight / 2;
                    nBottomMark = nY_Center + nCheckHeight / 2;
                    nLeftMark = nX_Center - nCheckWidth / 2;
                    nRightMark = nX_Center + nCheckWidth / 2;

                    cv::Mat matCrop = cv::Mat(item.img, cv::Rect(nLeftMark, nTopMark, nCheckWidth, nCheckHeight)).clone();
                    fnSaveImg(fmt::format("{}_{}_raw_crop", item.nCol, item.nRow), matCrop);

                    //now check 4 corner to see if any missing

                    auto fnGetNorm = [](cv::Mat &img, cv::Point &pt, int inter)->int
                    {
                        cv::Mat imgCalculate = cv::Mat(img, cv::Rect(pt.x - inter / 2, pt.y - inter / 2, inter, inter));
                        double dNorm = cv::norm(imgCalculate, NORM_L1) / imgCalculate.rows / imgCalculate.cols;
                        return (int)dNorm;
                    };

                    cv::Point ptCornerTL((*itMaxUp).first - m_algoParam.nDistanceFromBar2Corner, CenterSpringBar->at(0).first);
                    cv::Point ptCornerTR((*itMaxUp).second + m_algoParam.nDistanceFromBar2Corner, CenterSpringBar->at(0).first);

                    cv::Point ptCornerBL((*itMaxBom).first - m_algoParam.nDistanceFromBar2Corner, CenterSpringBar->at(CenterSpringBar->size() - 1).first);
                    cv::Point ptCornerBR((*itMaxBom).second + m_algoParam.nDistanceFromBar2Corner, CenterSpringBar->at(CenterSpringBar->size() - 1).first);

                    int nValTL = fnGetNorm(item.img, ptCornerTL, 4);
                    int nValTR = fnGetNorm(item.img, ptCornerTR, 4);
                    int nValBL = fnGetNorm(item.img, ptCornerBL, 4);
                    int nValBR = fnGetNorm(item.img, ptCornerBR, 4);

                    if (nValTL > validGrayRange.nHigh) item.mpCornerXY[singleSpring::eTL] = ptCornerTL;
                    if (nValTR > validGrayRange.nHigh) item.mpCornerXY[singleSpring::eTR] = ptCornerTR;
                    if (nValBL > validGrayRange.nHigh) item.mpCornerXY[singleSpring::eBL] = ptCornerBL;
                    if (nValBR > validGrayRange.nHigh) item.mpCornerXY[singleSpring::eBR] = ptCornerBR;

                    if (item.mpCornerXY.size())
                    {
                        LogInfo() << fmt::format("spring : {}_{} has missing corner", item.nCol, item.nRow);
                    }
                }



                //maThred includes all valid spring and potential defects(bright) 
                cv::Mat matThred;
                cv::inRange(item.img, cv::Scalar(validGrayRange.nLow), cv::Scalar(validGrayRange.nHigh), matThred);


                //crop to analysis area
                cv::Mat(matThred, cv::Rect(0, 0, matThred.cols, nTopMark + 1)).setTo(Scalar(0));
                cv::Mat(matThred, cv::Rect(0, nBottomMark, matThred.cols, matThred.rows - nBottomMark)).setTo(Scalar(0));
                cv::Mat(matThred, cv::Rect(0, 0, nLeftMark + 1, matThred.rows)).setTo(Scalar(0));
                cv::Mat(matThred, cv::Rect(nRightMark, 0, matThred.cols - nRightMark, matThred.rows)).setTo(Scalar(0));
                fnSaveImg(fmt::format("{}_{}_thred", item.nCol, item.nRow), matThred);


                //matRawMasked contains the analysis raw area
                cv::Mat matRawMasked(item.img.rows, item.img.cols, item.img.type(), Scalar(0));
                item.img.copyTo(matRawMasked, matThred);
                fnSaveImg(fmt::format("{}_{}_raw_mask", item.nCol, item.nRow), matRawMasked);

                //imgThredExcludeHole is used to exclude hole(dark) in further sobel image, because both hole(dark) and defect(bright) are shown bright in sobel image
                cv::Mat imgThredExcludeHole;
                cv::inRange(matRawMasked, cv::Scalar(dNorm), cv::Scalar(validGrayRange.nHigh), imgThredExcludeHole);
                fnSaveImg(fmt::format("{}_{}_thredExcludeHole", item.nCol, item.nRow), imgThredExcludeHole);


                //morphologyGrad is used to exclude border in sobel image
                cv::Mat element5(5, 5, CV_8U, cv::Scalar(1));
                cv::Mat morphologyGrad;
                morphologyEx(matThred, morphologyGrad, MORPH_GRADIENT, element5);
                fnSaveImg(fmt::format("{}_{}_thread_grad", item.nCol, item.nRow), morphologyGrad);

                //reverse morphologyGrad
                cv::bitwise_not(morphologyGrad, morphologyGrad);
                fnSaveImg(fmt::format("{}_{}_thread_grad_reverse", item.nCol, item.nRow), morphologyGrad);


                //use HMImage to generate a lut image which contains valid area
                HMImage image;
                image.setImg(matRawMasked, false);
                image.getLut()->setGammaRange(validGrayRange.nLow, validGrayRange.nHigh);
                cv::Mat showImg = image.getShowImg();
                fnSaveImg(fmt::format("{}_{}_lut", item.nCol, item.nRow), showImg);


                //generate sobel image
                cv::Mat sobelX;
                cv::Mat sobelY;
                cv::Scharr(showImg, sobelX, CV_16S, 1, 0);
                cv::Scharr(showImg, sobelY, CV_16S, 0, 1);
                cv::Mat sobel;
                //compute the L1 norm
                sobel = abs(sobelX) + abs(sobelY);

                cv::Mat sobelImage16;
                sobel.convertTo(sobelImage16, CV_16U);

                fnSaveImg(fmt::format("{}_{}_sobel16", item.nCol, item.nRow), sobelImage16);


                //exclude the border from sobel image
                cv::Mat sobelImageMaskExcludeBorder16;
                sobelImage16.copyTo(sobelImageMaskExcludeBorder16, morphologyGrad);
                fnSaveImg(fmt::format("{}_{}_sobel_mask_exclude_border16", item.nCol, item.nRow), sobelImageMaskExcludeBorder16);

                //exclude the hole(dark area)
                cv::Mat sobleImageExcludeBorderAndHole;
                sobelImageMaskExcludeBorder16.copyTo(sobleImageExcludeBorderAndHole, imgThredExcludeHole);
                fnSaveImg(fmt::format("{}_{}_sobel_mask_exclude_border_and_hole16", item.nCol, item.nRow), sobleImageExcludeBorderAndHole);


                auto fnHysteresisInter = [](cv::Mat &imgSrc, cv::Mat &imgDst, int nLowThre, int nHighThre, int nRow, int nCol)->bool
                {
                    bool bChanged = false;
                    for (int nInterRow = nRow - 1; nInterRow <= nRow + 1; ++nInterRow)
                    {
                        for (int nInterCol = nCol - 1; nInterCol <= nCol + 1; ++nInterCol)
                        {
                            if (nInterRow < 0 || nInterRow >= imgSrc.rows || nInterCol < 0 || nInterCol >= imgSrc.cols) continue;
                            if (nInterRow == nRow && nInterCol == nCol) continue;
                            if (imgDst.at<uchar>(nInterRow, nInterCol) == 255) continue;

                            ushort nVal = imgSrc.at<ushort>(nInterRow, nInterCol);
                            if (nVal >= nLowThre && nVal < nHighThre)
                            {
                                imgDst.at<uchar>(nInterRow, nInterCol) = 255;
                                bChanged = true;
                            }
                        }
                    }
                    return bChanged;
                };


                auto fnThresholdHysteresis = [&](cv::Mat &imgSrc, cv::Mat &imgDst, int nLowThre, int nHighThre)
                {
                    imgDst = cv::Mat::zeros(imgSrc.rows, imgSrc.cols, CV_8U);
                    cv::inRange(imgSrc, cv::Scalar(nHighThre), cv::Scalar(65535), imgDst);

                    int nLoop = 0;
                    while (1)
                    {
                        bool bChanged = false;

                        for (int nRow = 1; nRow < imgDst.rows - 1; ++nRow)
                        {
                            for (int nCol = 1; nCol < imgDst.cols - 1; ++nCol)
                            {
                                if (imgDst.at<uchar>(nRow, nCol) == 255)
                                {
                                    if (fnHysteresisInter(imgSrc, imgDst, nLowThre, nHighThre, nRow, nCol))
                                        bChanged = true;
                                }
                            }
                        }

                        if (!bChanged)
                            break;
                        nLoop++;
                    }

                };


                //generate Hysteresis image based on sobleImageExcludeBorderAndHole
                cv::Mat imgHysteresis;
                fnThresholdHysteresis(sobleImageExcludeBorderAndHole, imgHysteresis, nHysteresisLow, nHysteresisHigh);
                fnSaveImg(fmt::format("{}_{}_imgHysteresis", item.nCol, item.nRow), imgHysteresis);


                cv::findContours(imgHysteresis,
                    item.vContours, // a vector of contours
                    CV_RETR_EXTERNAL, // retrieve the external contours 
                    CV_CHAIN_APPROX_NONE); // all pixels of each contours

                auto itErase = std::remove_if(item.vContours.begin(), item.vContours.end(), [&](const std::vector<cv::Point> &vPts) {
                    return vPts.size() < (size_t)nContourCountThreshold;
                });
                item.vContours.erase(itErase, item.vContours.end());

                item.imgContour = item.img.clone();
                for (auto &ct : item.vContours)
                {
                    cv::Rect r0 = cv::boundingRect(cv::Mat(ct));
                    cv::rectangle(item.imgContour, r0, cv::Scalar(65535), 2);
                    fnSaveImg(fmt::format("{}_{}_contour", item.nCol, item.nRow), item.imgContour);

                }
                vRet.push_back(item);
            }
        }
    }
    return true;
}
