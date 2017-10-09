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
    }
}
