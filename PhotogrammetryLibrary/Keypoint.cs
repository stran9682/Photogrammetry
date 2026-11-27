namespace PhotogrammetryLibrary;

public readonly record struct Keypoint(double X, double Y, double S, double Orientation);

public readonly record struct Coordinate(int Layer, int Row, int Column);