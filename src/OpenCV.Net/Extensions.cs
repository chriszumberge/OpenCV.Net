using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;

namespace OpenCV.Net
{
    public static class Extensions
    {
        public static IInputArray ToIInputArray(this Color color)
        {
            int r = color.R;
            int g = color.G;
            int b = color.B;

            return new Matrix<int>(new int[] { r, g, b });
        }

        //public static implicit operator MCvScalar(Color c) => new MCvScalar(c.R, c.G, c.B);
        public static MCvScalar ToMCvSCalar(this Color color)
        {
            return new Bgr(color).MCvScalar;
        }
    }
}
