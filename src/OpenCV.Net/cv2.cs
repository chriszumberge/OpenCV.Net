using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Structure;

namespace OpenCV.Net
{
    public static class Cv2
    {
        /// <summary>
        /// Draws a simple or filled circle with given center and radius. The circle is clipped by ROI rectangle.
        /// </summary>
        /// <param name="img">Image where the circle is drawn</param>
        /// <param name="center">Center of the circle</param>
        /// <param name="radius">Radius of the circle.</param>
        /// <param name="color">Color of the circle</param>
        /// <param name="thickness">Thickness of the circle outline if positive, otherwise indicates that a filled circle has to be drawn</param>
        /// <param name="lineType">Line type</param>
        /// <param name="shift">Number of fractional bits in the center coordinates and radius value</param>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html?highlight=cv2.circle#cv2.circle
        /// </remarks>
        public static void Circle(IInputOutputArray img, Point center, int radius, MCvScalar color, int thickness = 1, Emgu.CV.CvEnum.LineType lineType = Emgu.CV.CvEnum.LineType.EightConnected, int shift = 0)
        {
            // No change needed, just wrapping to complete the api
            CvInvoke.Circle(img, center, radius, color, thickness, lineType, shift);
        }

        /// <summary>
        /// Draws a simple or filled circle with given center and radius. The circle is clipped by ROI rectangle.
        /// </summary>
        /// <param name="img">Image where the circle is drawn</param>
        /// <param name="center">Center of the circle</param>
        /// <param name="radius">Radius of the circle.</param>
        /// <param name="color">Color of the circle</param>
        /// <param name="thickness">Thickness of the circle outline if positive, otherwise indicates that a filled circle has to be drawn</param>
        /// <param name="lineType">Line type</param>
        /// <param name="shift">Number of fractional bits in the center coordinates and radius value</param>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html?highlight=cv2.circle#cv2.circle
        /// </remarks>
        public static void Circle(IInputOutputArray img, Point center, int radius, Color color, int thickness = 1, Emgu.CV.CvEnum.LineType lineType = Emgu.CV.CvEnum.LineType.EightConnected, int shift = 0)
        {
            // No change needed, just wrapping to complete the api
            CvInvoke.Circle(img, center, radius, color.ToMCvSCalar(), thickness, lineType, shift);
        }

        /// <summary>
        /// Calculates area of the whole contour or contour section. 
        /// </summary>
        /// <param name="contour">Input vector of 2D points (contour vertices), stored in std::vector or Mat. </param>
        /// <param name="oriented">Oriented area flag. If it is true, the function returns a signed area value, depending on the contour orientation (clockwise or counter-clockwise).
        /// Using this feature you can determine orientation of a contour by taking the sign of an area. 
        /// By default, the parameter is false, which means that the absolute value is returned.</param>
        /// <returns>The area of the whole contour or contour section</returns>

        public static double ContourArea(IInputArray contour, bool oriented = false)
        {
            // No change needed, just wrapping to complete the api
            return CvInvoke.ContourArea(contour, oriented);
        }

        /// <summary>
        /// Converts input image from one color space to another. The function ignores colorModel and channelSeq fields of IplImage header, so the source image color space should be specified correctly (including order of the channels in case of RGB space, e.g. BGR means 24-bit format with B0 G0 R0 B1 G1 R1 ... layout, whereas RGB means 24-bit format with R0 G0 B0 R1 G1 B1 ... layout). 
        /// </summary>
        /// <param name="src">The source 8-bit (8u), 16-bit (16u) or single-precision floating-point (32f) image</param>
        /// <param name="code">Color conversion operation that can be specifed using CV_src_color_space2dst_color_space constants </param>
        /// <param name="dstCn">Number of channels in the output image; if the parameter is 0, the number of the channels is derived automatically from src and code .</param>
        /// <returns>The output image of the same data type as the source one. The number of channels may be different</returns>
        public static IOutputArray CvtColor(IInputArray src, ColorConversion code, int dstCn = 0)
        {
            Mat dst = new Mat();
            CvInvoke.CvtColor(src, dst, code, dstCn);
            return dst;
        }

        /// <summary>
        /// Dilates the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the maximum is taken
        /// The function supports the in-place mode. Dilation can be applied several (iterations) times. In case of color image each channel is processed independently
        /// </summary>
        /// <param name="src">Source image</param>
        /// <param name="element">Structuring element used for erosion. If it is IntPtr.Zero, a 3x3 rectangular structuring element is used</param>
        /// <param name="iterations">Number of times erosion is applied</param>
        /// <param name="borderType">Pixel extrapolation method</param>
        /// <param name="borderValue">Border value in case of a constant border </param>
        /// <param name="anchor">Position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.</param>
        /// <returns>Output image of the same size and type as <code>src</code>.</returns>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
        /// C++: void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )
        /// Python: cv2.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) → dst
        /// C: void cvDilate(const CvArr* src, CvArr* dst, IplConvKernel* element = NULL, int iterations = 1 )
        /// Python: cv.Dilate(src, dst, element=None, iterations=1) → None
        /// </remarks>
        public static IOutputArray Dilate(IInputArray src, IInputArray element = null, Point? anchor = null, int iterations = 1, Emgu.CV.CvEnum.BorderType borderType = BorderType.Constant, MCvScalar? borderValue = null)
        {
            if (!anchor.HasValue)
                anchor = new Point(-1, -1);
            if (!borderValue.HasValue)
                borderValue = CvInvoke.MorphologyDefaultBorderValue;

            Mat dst = new Mat();
            CvInvoke.Dilate(src, dst, element, anchor.Value, iterations, borderType, borderValue.Value);
            return dst;
        }

        /// <summary>
        /// Erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken:
        /// dst=erode(src,element):  dst(x,y)=min((x',y') in element)) src(x+x',y+y')
        /// The function supports the in-place mode. Erosion can be applied several (iterations) times. In case of color image each channel is processed independently.
        /// </summary>
        /// <param name="src">Input image; the number of channels can be arbitrary, but the depth should be one of <code>CV_8U, CV_16U, CV_16S, CV_32F`</code> or <code>``CV_64F.</code></param>
        /// <param name="element">Structuring element used for erosion. If it is <code>IntPtr.Zero</code>, a 3x3 rectangular structuring element is used.</param>
        /// <param name="anchor">Position of the anchor within the element; default value <code>(-1, -1)</code> means that the anchor is at the element center.</param>
        /// <param name="iterations">Number of times erosion is applied.</param>
        /// <param name="borderType">Pixel extrapolation method <seealso cref="BorderInterpolate"/></param>
        /// <param name="borderValue">Border value in case of a constant border, use Constant for default</param>
        /// <returns>Output image of the same size and type as <code>src</code>.</returns>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=erode#erode
        /// C++: void erode(InputArray src, OutputArray dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )
        /// Python: cv2.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) → dst
        /// C: void cvErode(const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1)
        /// Python: cv.Erode(src, dst, element=None, iterations=1) → None
        /// </remarks>
        public static IOutputArray Erode(IInputArray src, IInputArray element = null, Point? anchor = null, int iterations = 1, Emgu.CV.CvEnum.BorderType borderType = BorderType.Constant, MCvScalar? borderValue = null)
        {
            if (!anchor.HasValue)
                anchor = new Point(-1, -1);
            if (!borderValue.HasValue)
                borderValue = CvInvoke.MorphologyDefaultBorderValue;

            Mat dst = new Mat();

            CvInvoke.Erode(src, dst, element, anchor.Value, iterations, borderType, borderValue.Value);

            return dst;
        }

        /// <summary>
        /// Retrieves contours from the binary image and returns the number of retrieved contours. The pointer firstContour is filled by the function. It will contain pointer to the first most outer contour or IntPtr.Zero if no contours is detected (if the image is completely black). Other contours may be reached from firstContour using h_next and v_next links. The sample in cvDrawContours discussion shows how to use contours for connected component detection. Contours can be also used for shape analysis and object recognition - see squares.c in OpenCV sample directory
        /// The function modifies the source image content
        /// </summary>
        /// <param name="image">The source 8-bit single channel image. Non-zero pixels are treated as 1s, zero pixels remain 0s - that is image treated as binary. To get such a binary image from grayscale, one may use cvThreshold, cvAdaptiveThreshold or cvCanny. The function modifies the source image content</param>
        /// <param name="hierarchy">Optional output vector, containing information about the image topology.</param>
        /// <param name="mode">Retrieval mode</param>
        /// <param name="method">Approximation method (for all the modes, except CV_RETR_RUNS, which uses built-in approximation). </param>
        /// <returns>Detected contours. Each contour is stored as a vector of points.</returns>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours
        /// C++: void findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point())
        /// C++: void findContours(InputOutputArray image, OutputArrayOfArrays contours, int mode, int method, Point offset = Point())
        /// Python: cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]]) → contours, hierarchy
        /// C: int cvFindContours(CvArr* image, CvMemStorage* storage, CvSeq** first_contour, int header_size = sizeof(CvContour), int mode = CV_RETR_LIST, int method = CV_CHAIN_APPROX_SIMPLE, CvPoint offset = cvPoint(0, 0))
        /// Python: cv.FindContours(image, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=(0, 0)) → contours
        /// </remarks>
        public static VectorOfVectorOfPoint FindContours(IInputOutputArray src, Emgu.CV.CvEnum.RetrType mode = RetrType.List, ChainApproxMethod method = ChainApproxMethod.ChainApproxSimple)
        {
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(src, contours, null, mode, method);
            return contours;
        }

        /// <summary>
        /// Blurs an image using a Gaussian filter.
        /// </summary>
        /// <param name="src">Input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.</param>
        /// <param name="ksize">Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd. Or, they can be zero’s and then they are computed from sigma* .</param>
        /// <param name="sigmaX">Gaussian kernel standard deviation in X direction.</param>
        /// <param name="sigmaY">Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height , respectively (see getGaussianKernel() for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.</param>
        /// <param name="borderType">Pixel extrapolation method</param>
        /// <returns>Output image of the same size and type as src.</returns>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#gaussianblur
        /// C++: void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )
        /// Python: cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) → dst
        /// </remarks>
        public static IOutputArray GaussianBlur(IInputArray src, Size ksize, double sigmaX, double sigmaY = 0, BorderType borderType = BorderType.Reflect101)
        {
            Mat dst = new Mat();
            CvInvoke.GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType);
            return dst;
        }


        /// <summary>
        /// Performs range check for every element of the input array:
        /// dst(I)=lower(I)_0 &lt;= src(I)_0 &lt;= upper(I)_0
        /// For single-channel arrays,
        /// dst(I)=lower(I)_0 &lt;= src(I)_0 &lt;= upper(I)_0 &amp;&amp;
        /// lower(I)_1 &lt;= src(I)_1 &lt;= upper(I)_1
        /// For two-channel arrays etc.
        /// dst(I) is set to 0xff (all '1'-bits) if src(I) is within the range and 0 otherwise. All the arrays must have the same type, except the destination, and the same size (or ROI size)
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="lower">The lower values stored in an image of same type &amp; size as <paramref name="src"/></param>
        /// <param name="upper">The upper values stored in an image of same type &amp; size as <paramref name="src"/></param>
        /// <returns>The resulting mask</returns>
        public static IOutputArray InRange(IInputArray src, IInputArray lower, IInputArray upper)
        {
            Mat mask = new Mat();
            CvInvoke.InRange(src, lower, upper, mask);
            return mask;
        }

        /// <summary>
        /// Performs range check for every element of the input array:
        /// dst(I)=lower(I)_0 &lt;= src(I)_0 &lt;= upper(I)_0
        /// For single-channel arrays,
        /// dst(I)=lower(I)_0 &lt;= src(I)_0 &lt;= upper(I)_0 &amp;&amp;
        /// lower(I)_1 &lt;= src(I)_1 &lt;= upper(I)_1
        /// For two-channel arrays etc.
        /// dst(I) is set to 0xff (all '1'-bits) if src(I) is within the range and 0 otherwise. All the arrays must have the same type, except the destination, and the same size (or ROI size)
        /// </summary>
        /// <param name="src">The source image</param>
        /// <param name="lower">The lower value of color to mask</param>
        /// <param name="upper">The upper value of color to mask</param>
        /// <returns>The resulting mask</returns>
        public static IOutputArray InRange(IInputArray src, Color lower, Color upper)
        {
            return Cv2.InRange(src, lower.ToIInputArray(), upper.ToIInputArray());
        }

        /// <summary>
        /// Draws the line segment between pt1 and pt2 points in the image. The line is clipped by the image or ROI rectangle. For non-antialiased lines with integer coordinates the 8-connected or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased lines are drawn using Gaussian filtering.
        /// </summary>
        /// <param name="img">The image</param>
        /// <param name="pt1">First point of the line segment</param>
        /// <param name="pt2">Second point of the line segment</param>
        /// <param name="color">Line color</param>
        /// <param name="thickness">Line thickness. </param>
        /// <param name="lineType">Type of the line:
        /// 8 (or 0) - 8-connected line.
        /// 4 - 4-connected line.
        /// CV_AA - antialiased line. 
        /// </param>
        /// <param name="shift">Number of fractional bits in the point coordinates</param>
        public static void Line(IInputOutputArray img, Point pt1, Point pt2, MCvScalar color, int thickness = 1, Emgu.CV.CvEnum.LineType lineType = Emgu.CV.CvEnum.LineType.EightConnected, int shift = 0)
        {
            // No change needed, just wrapping to complete the api
            CvInvoke.Line(img, pt1, pt2, color, thickness, lineType, shift);
        }

        /// <summary>
        /// Draws the line segment between pt1 and pt2 points in the image. The line is clipped by the image or ROI rectangle. For non-antialiased lines with integer coordinates the 8-connected or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased lines are drawn using Gaussian filtering.
        /// </summary>
        /// <param name="img">The image</param>
        /// <param name="pt1">First point of the line segment</param>
        /// <param name="pt2">Second point of the line segment</param>
        /// <param name="color">Line color</param>
        /// <param name="thickness">Line thickness. </param>
        /// <param name="lineType">Type of the line:
        /// 8 (or 0) - 8-connected line.
        /// 4 - 4-connected line.
        /// CV_AA - antialiased line. 
        /// </param>
        /// <param name="shift">Number of fractional bits in the point coordinates</param>
        public static void Line(IInputOutputArray img, Point pt1, Point pt2, Color color, int thickness = 1, Emgu.CV.CvEnum.LineType lineType = Emgu.CV.CvEnum.LineType.EightConnected, int shift = 0)
        {
            // No change needed, just wrapping to complete the api
            CvInvoke.Line(img, pt1, pt2, color.ToMCvSCalar(), thickness, lineType, shift);
        }

        /// <summary>
        /// Finds the minimal circumscribed circle for 2D point set using iterative algorithm. It returns nonzero if the resultant circle contains all the input points and zero otherwise (i.e. algorithm failed)
        /// </summary>
        /// <param name="points">Sequence or array of 2D points</param>
        ///<returns>The minimal circumscribed circle for 2D point set</returns>
        public static CircleF MinEnclosingCircle(IInputArray points)
        {
            // No change needed, just wrapping to complete the api
            return CvInvoke.MinEnclosingCircle(points);
        }

        /// <summary>
        /// Calculates spatial and central moments up to the third order and writes them to moments. The moments may be used then to calculate gravity center of the shape, its area, main axises and various shape characeteristics including 7 Hu invariants.
        /// </summary>
        /// <param name="arr">Image (1-channel or 3-channel with COI set) or polygon (CvSeq of points or a vector of points)</param>
        /// <param name="binaryImage">(For images only) If the flag is true, all the zero pixel values are treated as zeroes, all the others are treated as 1s</param>
        /// <returns>The moment</returns>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#cv2.moments
        /// </remarks>
        public static MCvMoments Moments(IInputArray arr, bool binaryImage = false)
        {
            // No change needed, just wrapping to complete the api
            return CvInvoke.Moments(arr, binaryImage);
        }

        /// <summary>
        /// Renders the text in the image with the specified font and color. The printed text is clipped by ROI rectangle. Symbols that do not belong to the specified font are replaced with the rectangle symbol.
        /// </summary>
        /// <param name="img">Input image</param>
        /// <param name="text">String to print</param>
        /// <param name="org">Coordinates of the bottom-left corner of the first letter</param>
        /// <param name="fontFace">Font type.</param>
        /// <param name="fontScale">Font scale factor that is multiplied by the font-specific base size.</param>
        /// <param name="color">Text color</param>
        /// <param name="thickness">Thickness of the lines used to draw a text.</param>
        /// <param name="lineType">Line type</param>
        /// <param name="bottomLeftOrigin">When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.</param>
        /// <remarks>
        /// http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html?highlight=cv2.circle#puttext
        /// C++: void putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, Scalar color, int thickness=1, int lineType=8, bool bottomLeftOrigin=false )
        /// Python: cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) → None
        /// C: void cvPutText(CvArr* img, const char* text, CvPoint org, const CvFont* font, CvScalar color)
        /// Python: cv.PutText(img, text, org, font, color) → None
        /// </remarks>
        public static void PutText(IInputOutputArray img, String text, Point org, Emgu.CV.CvEnum.FontFace fontFace, double fontScale, MCvScalar color, int thickness = 1, Emgu.CV.CvEnum.LineType lineType = Emgu.CV.CvEnum.LineType.EightConnected, bool bottomLeftOrigin = false)
        {
            // No change needed, just wrapping to complete the api
            CvInvoke.PutText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);
        }

        /// <summary>
        /// Renders the text in the image with the specified font and color. The printed text is clipped by ROI rectangle. Symbols that do not belong to the specified font are replaced with the rectangle symbol.
        /// </summary>
        /// <param name="img">Input image</param>
        /// <param name="text">String to print</param>
        /// <param name="org">Coordinates of the bottom-left corner of the first letter</param>
        /// <param name="fontFace">Font type.</param>
        /// <param name="fontScale">Font scale factor that is multiplied by the font-specific base size.</param>
        /// <param name="color">Text color</param>
        /// <param name="thickness">Thickness of the lines used to draw a text.</param>
        /// <param name="lineType">Line type</param>
        /// <param name="bottomLeftOrigin">When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.</param>
        public static void PutText(IInputOutputArray img, String text, Point org, Emgu.CV.CvEnum.FontFace fontFace, double fontScale, Color color, int thickness = 1, Emgu.CV.CvEnum.LineType lineType = Emgu.CV.CvEnum.LineType.EightConnected, bool bottomLeftOrigin = false)
        {
            // No change needed, just wrapping to complete the api
            CvInvoke.PutText(img, text, org, fontFace, fontScale, color.ToMCvSCalar(), thickness, lineType, bottomLeftOrigin);
        }
    }
}
