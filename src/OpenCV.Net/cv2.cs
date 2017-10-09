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
        /// Erodes the source image using the specified structuring element that determines the shape of a pixel neighborhood over which the minimum is taken:
        /// dst=erode(src,element):  dst(x,y)=min((x',y') in element)) src(x+x',y+y')
        /// The function supports the in-place mode. Erosion can be applied several (iterations) times. In case of color image each channel is processed independently.
        /// </summary>
        /// <param name="src">Source image. </param>
        /// <param name="element">Structuring element used for erosion. If it is IntPtr.Zero, a 3x3 rectangular structuring element is used.</param>
        /// <param name="iterations">Number of times erosion is applied.</param>
        /// <param name="borderType">Pixel extrapolation method</param>
        /// <param name="borderValue">Border value in case of a constant border, use Constant for default</param>
        /// <param name="anchor">Position of the anchor within the element; default value (-1, -1) means that the anchor is at the element center.</param>
        /// <returns>Destination image</returns>
        public static IOutputArray Erode(IInputArray src, IInputArray kernel = null, Point? anchor = null, int iterations = 1, Emgu.CV.CvEnum.BorderType borderType = BorderType.Constant, MCvScalar? borderValue = null)
        {
            if (!anchor.HasValue)
                anchor = new Point(-1, -1);
            if (!borderValue.HasValue)
                borderValue = CvInvoke.MorphologyDefaultBorderValue;

            Mat dst = new Mat();

            CvInvoke.Erode(src, dst, kernel, anchor.Value, iterations, borderType, borderValue.Value);

            return dst;
        }
    }
}
