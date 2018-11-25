#ifndef HM_SPRING_IMG_PROCESS_H
#define HM_SPRING_IMG_PROCESS_H
#include <memory>
#include <map>
#include <vector>
#include <opencv2/core/mat.hpp>
class HMImage;
class HMSpringImgProcess
{
public:
    static HMSpringImgProcess &inst();
    //structure definition

    struct grayRange_t
    {
        void setRange(int l, int h)
        {
            nLow    = l;
            nHigh   = h;
        }
        int nLow;
        int nHigh;
    };

    struct algoParameter_t
    {
        algoParameter_t()
        {
            nCellHeight     = 170;
            nCellWidth      = 520;

            nCheckWidth     = 190 * 2;
            nCheckHeight    = 40 * 2;

            nInterval       = 2;
            nHysteresisLow  = 40;
            nHysteresisHigh = 80;

            nContourCountThreshold  = 10;
            nDistanceFromBar2Corner = 25;

            plateBG.setRange(13000, 17000);
            whiteBG.setRange(20000, 60000);
            SpringBG.setRange(10000, 12999);
            SpringBarBG.setRange(3000, 9999);
            blackBG.setRange(0, 10);
        }
        //here is the parameter
        int nCellHeight;
        int nCellWidth;

        int nCheckWidth;
        int nCheckHeight;
        int nInterval;
        int nHysteresisLow;
        int nHysteresisHigh;
        int nContourCountThreshold;
        int nDistanceFromBar2Corner;


        grayRange_t plateBG;
        grayRange_t whiteBG;
        grayRange_t SpringBG ;
        grayRange_t SpringBarBG ;
        grayRange_t blackBG;

    };
    struct singleSpring
    {
		singleSpring()
		{
			bDefect = false;
			nConerDfects = 0;
		}

        bool  isDefect() {
            return vContours.size() || mpCornerXY.size();
        }
		enum eCornerDefect
		{
			eTL = 0x01,
			eTR = 0x02,
			eBL = 0x04,
			eBR = 0x08,
		};
		using vContours_t	= std::vector<std::vector<cv::Point>>;
        using vRects_t      = std::vector<cv::Rect>;
        using mpCornerXY_t  = std::map<eCornerDefect, cv::Point>;
		cv::Mat			img;
		cv::Mat			imgContour;
		vContours_t		vContours;
		vRects_t		vDefectRects;
		int				nConerDfects;
        int				nCol;
        int				nRow;
        bool			bDefect;
        mpCornerXY_t    mpCornerXY;
    };

    //type declaration
    using pImage_t = std::shared_ptr<HMImage>;
    using vSprings_t = std::vector<singleSpring>;

    //method
    HMSpringImgProcess();
    algoParameter_t getAlgoParameter() { return m_algoParam; }
    bool            setAlgoParameter(const algoParameter_t&);
    bool            analyzeSpringImg(pImage_t img, vSprings_t &);
private:
    pImage_t        m_pImg;
    algoParameter_t m_algoParam;
};

#endif
