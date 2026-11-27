using PhotogrammetryLibrary;

namespace Photogrammetry.Tests;

public class UtilsTests
{
	[Theory]
	[InlineData(1, 1, 1, true)] // typical valid interior point
	[InlineData(2, 5, 2, true)] // middle valid point
	[InlineData(0, 5, 2, false)] // row on top edge
	[InlineData(9, 5, 2, false)] // row on bottom edge (rows = 10)
	[InlineData(5, 0, 2, false)] // col on left edge
	[InlineData(5, 9, 2, false)] // col on right edge (cols = 10)
	[InlineData(5, 5, 0, false)] // layer 0 (lowest layer)
	[InlineData(5, 5, 4, false)] // layer 4 (highest layer, count = 5)
	[InlineData(-1, 5, 2, false)] // negative row
	[InlineData(5, 10, 2, false)] // out of bound col
	public void IsWithinBounds_ValidatesBoundaryCorrectly(int row, int col, int layer, bool expected)
	{
		var dogOctave = TestHelpers.CreateDogOctave();
		var coordinate = new Coordinate(layer, row, col);

		var result = Utils.IsWithinBounds(dogOctave, coordinate);

		Assert.Equal(expected, result);
	}

	[Fact]
	public void ApproximateGradient_UniformOctave_ReturnsZeroGradient()
	{
		var dogOctave = TestHelpers.CreateDogOctave(5, 10, 10, 42f);
		var coordinate = new Coordinate(2, 5, 5);

		var gradient = Utils.ApproximateGradient(dogOctave, coordinate);

		Assert.Equal(3, gradient.Length);
		Assert.Equal(0.0, gradient[0], 6); // dx (col)
		Assert.Equal(0.0, gradient[1], 6); // dy (row)
		Assert.Equal(0.0, gradient[2], 6); // ds (layer)
	}

	[Fact]
	public void ApproximateGradient_LinearFields_ComputesCorrectDerivatives()
	{
		var dogOctave = TestHelpers.CreateDogOctave();

		// Populate with f(row, col, layer) = 2*col + 4*row + 8*layer
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
			dogOctave[l][r, c] = 2f * c + 4f * r + 8f * l;
		
		var coordinate = new Coordinate(2, 4, 4);
		var gradient = Utils.ApproximateGradient(dogOctave, coordinate);

		// dx = ImageScale * 0.5 * (f(c+1) - f(c-1)) = ImageScale * 0.5 * (2*2) = 2 * ImageScale
		var expectedDx = 2.0 * Utils.ImageScale;
		// dy = ImageScale * 0.5 * (f(r+1) - f(r-1)) = ImageScale * 0.5 * (4*2) = 4 * ImageScale
		var expectedDy = 4.0 * Utils.ImageScale;
		// ds = ImageScale * 0.5 * (f(l+1) - f(l-1)) = ImageScale * 0.5 * (8*2) = 8 * ImageScale
		var expectedDs = 8.0 * Utils.ImageScale;

		Assert.Equal(expectedDx, gradient[0], 6);
		Assert.Equal(expectedDy, gradient[1], 6);
		Assert.Equal(expectedDs, gradient[2], 6);
	}

	[Fact]
	public void ApproximateHessian_ConstantOctave_ReturnsZeroHessian()
	{
		var dogOctave = TestHelpers.CreateDogOctave(5, 10, 10, 15f);
		var coordinate = new Coordinate(2, 3, 3);

		var hessian = Utils.ApproximateHessian(dogOctave, coordinate);

		for (var i = 0; i < 3; i++)
		for (var j = 0; j < 3; j++)
			Assert.Equal(0.0, hessian[i, j], 6);
	}

	[Fact]
	public void ApproximateHessian_IsSymmetric()
	{
		var dogOctave = TestHelpers.CreateDogOctave();
		// Assign pseudo-random or arbitrary values
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
			dogOctave[l][r, c] = (float)Math.Sin(l + 1) * 10f + r * c;
		
		var coordinate = new Coordinate(2, 4, 4);
		var hessian = Utils.ApproximateHessian(dogOctave, coordinate);

		Assert.Equal(hessian[0, 1], hessian[1, 0], 6);
		Assert.Equal(hessian[0, 2], hessian[2, 0], 6);
		Assert.Equal(hessian[1, 2], hessian[2, 1], 6);
	}

	[Fact]
	public void ApproximateHessian_QuadraticFunction_ComputesCorrectSecondDerivatives()
	{
		var dogOctave = TestHelpers.CreateDogOctave();

		// f(r, c, l) = 3*c^2 + 5*r^2 + 7*l^2 + 2*c*r
		for (var l = 0; l < 5; l++)
		for (var r = 0; r < 10; r++)
		for (var c = 0; c < 10; c++)
			dogOctave[l][r, c] = 3f * c * c + 5f * r * r + 7f * l * l + 2f * c * r;

		var coordinate = new Coordinate(2, 4, 4);
		var hessian = Utils.ApproximateHessian(dogOctave, coordinate);

		// dxx = ImageScale * (3*(c+1)^2 - 6*c^2 + 3*(c-1)^2) = ImageScale * 6
		var expectedDxx = 6.0 * Utils.ImageScale;
		// dyy = ImageScale * (5*(r+1)^2 - 10*r^2 + 5*(r-1)^2) = ImageScale * 10
		var expectedDyy = 10.0 * Utils.ImageScale;
		// dss = ImageScale * (7*(l+1)^2 - 14*l^2 + 7*(l-1)^2) = ImageScale * 14
		var expectedDss = 14.0 * Utils.ImageScale;
		// dxy = ImageScale * 0.25 * [ 2*(c+1)(r+1) - 2*(c-1)(r+1) - 2*(c+1)(r-1) + 2*(c-1)(r-1) ] = ImageScale * 2
		var expectedDxy = 2.0 * Utils.ImageScale;

		Assert.Equal(expectedDxx, hessian[0, 0], 6);
		Assert.Equal(expectedDyy, hessian[1, 1], 6);
		Assert.Equal(expectedDss, hessian[2, 2], 6);
		Assert.Equal(expectedDxy, hessian[0, 1], 6);
		Assert.Equal(0.0, hessian[0, 2], 6); // dxs is 0
		Assert.Equal(0.0, hessian[1, 2], 6); // dys is 0
	}

	[Fact]
	public void CalculateOffset_SingularHessian_ReturnsNull()
	{
		// All-zero Hessian has determinant 0
		double[,] zeroHessian =
		{
			{ 0, 0, 0 },
			{ 0, 0, 0 },
			{ 0, 0, 0 }
		};
		double[] gradient = [1.0, 1.0, 1.0];

		var offset = Utils.CalculateOffset(zeroHessian, gradient);

		Assert.Null(offset);
	}

	[Fact]
	public void CalculateOffset_DiagonalHessian_SolvesNewtonStepCorrectly()
	{
		// For H = diag(2, 4, 8) and g = [1, 2, 4]:
		// H * x = -g  =>  x = -H^-1 * g
		// x (col)   = -1 / 2 = -0.5
		// y (row)   = -2 / 4 = -0.5
		// s (layer) = -4 / 8 = -0.5
		// CalculateOffset returns [rowOffset, colOffset, scaleOffset] = [y, x, s]
		double[,] hessian =
		{
			{ 2.0, 0.0, 0.0 },
			{ 0.0, 4.0, 0.0 },
			{ 0.0, 0.0, 8.0 }
		};
		// gradient is [dx, dy, ds] = [1, 2, 4]
		double[] gradient = [1.0, 2.0, 4.0];

		var offset = Utils.CalculateOffset(hessian, gradient);

		Assert.NotNull(offset);
		Assert.Equal(-0.5, offset[0], 6); // rowOffset (y)
		Assert.Equal(-0.5, offset[1], 6); // colOffset (x)
		Assert.Equal(-0.5, offset[2], 6); // scaleOffset (s)
	}

	[Fact]
	public void CalculateOffset_PreservesCoordinateOrdering()
	{
		// To distinguish row vs col offset:
		// Let H = diag(2, 10, 5) where index 0=x (col), 1=y (row), 2=s (layer)
		// g = [2, 10, 5]
		// Then x = -2/2 = -1, y = -10/10 = -1, s = -5/5 = -1
		// Now let's differentiate the solutions:
		// x = -0.2 (gx = 0.4), y = -0.7 (gy = 7.0), s = -0.1 (gs = 0.5)
		double[,] hessian =
		{
			{ 2.0, 0.0, 0.0 },
			{ 0.0, 10.0, 0.0 },
			{ 0.0, 0.0, 5.0 }
		};
		double[] gradient = [0.4, 7.0, 0.5];

		var offset = Utils.CalculateOffset(hessian, gradient);

		Assert.NotNull(offset);
		// row offset corresponds to y: -7.0 / 10.0 = -0.7
		Assert.Equal(-0.7, offset[0], 5);
		// col offset corresponds to x: -0.4 / 2.0 = -0.2
		Assert.Equal(-0.2, offset[1], 5);
		// scale offset corresponds to s: -0.5 / 5.0 = -0.1
		Assert.Equal(-0.1, offset[2], 5);
	}
}