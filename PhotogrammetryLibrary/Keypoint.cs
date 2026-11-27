namespace PhotogrammetryLibrary;

public readonly struct Keypoint(double x, double y, double s, double orientation)
{
	public readonly double X = x;
	public readonly double Y = y;
	public readonly double S = s;
	public readonly double Orientation = orientation;
}