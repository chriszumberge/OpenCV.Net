# OpenCV.Net

OpenCV.Net is a reimplementation of [Emgu CV](https://github.com/emgucv/emgucv), a cross-platform .Net wrapper to the OpenCV image processing library.

OpenCV.Net reimplements the methods of the Emgu CV library, but conforms them to the [OpenCV API Reference](http://docs.opencv.org/2.4/modules/refman.html) 
style so that they can be more easily used with the python documentation and examples.

Most notably this means adding return types and optional parameters.
However, this also includes adding method overloads that accept common C# classes, for example
OpenCV uses the `MCvScalar` class to define colors when drawing on frames.
OpenCV.Net offers an overload of the method that uses `System.Drawing.Color`.



### Example
The [OpenCV API Documentation defines erode](http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=erode#erode)
in Python as:
```Python
cv2.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) → dst
```


EmguCV defines the method adhering to the C++ documentation as:
```C#
public static void Erode(IInputArray src, IOutputArray dst, IInputArray element, Point anchor, int iterations, CvEnum.BorderType borderType, MCvScalar borderValue)
```

But OpenCV.Net wraps the EmguCV method as:
```C#
public static IOutputArray Erode(IInputArray src, IInputArray kernel = null, Point? anchor = null, int iterations = 1, Emgu.CV.CvEnum.BorderType borderType = BorderType.Constant, MCvScalar? borderValue = null)
```

so that when a C# developer sees this line in a python tutorial
```Python
output = cv2.erode(mask, None, iterations=2)
```
they can simply write:
```C#
Mat output = Cv2.Erode(src, null, iterations = 2);
```
in their C# class.