using Emgu.CV;
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
    }
}
